# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Adapted from Transformers PR #48249 under the Apache License, Version 2.0.
# See LICENSES/transformers-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.

"""Greedy text/audio scheduling, following Transformers PR #48249.

The native engine advances its own LFM2 cache. Each audio event produces eight
codes, then feeds their summed embedding into the next native decoding step.
"""

import torch
from transformers import Lfm2AudioForConditionalGeneration
from transformers.models.lfm2_audio.modeling_lfm2_audio import Lfm2AudioGenerateOutput

from ..runtime import depth_graph


class AudioState:
    def __init__(self, source):
        model = Lfm2AudioForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16, device_map="cpu"
        ).eval()
        for name in (
            "lfm",
            "conformer",
            "audio_adapter_norm",
            "audio_adapter_linear_1",
            "audio_adapter_linear_2",
        ):
            setattr(model.model, name, torch.nn.Identity())
        self.model = model.cuda()
        self.config = model.config
        self.next_embedding = torch.zeros(
            self.config.text_config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        self.use_audio = torch.zeros((), dtype=torch.bool, device="cuda")
        self.active = False
        self.graph = None

    def reset(self, mode):
        if mode not in ("sequential", "interleaved"):
            raise ValueError("generation_mode must be 'sequential' or 'interleaved'.")
        self.mode = mode
        self.modality = 1
        self.left = self.config.interleaved_n_text
        self.text_done = False
        self.events = []
        self.use_audio.fill_(False)
        self.active = True

    def embed(self, input_ids, embeddings):
        mask = (input_ids == self.config.audio_token_id) & self.use_audio
        return torch.where(mask[:, None], self.next_embedding, embeddings)

    @torch.inference_mode()
    def step(self, hidden, logits):
        if not self.active:
            return logits, None
        if logits.shape[0] != 1:
            raise ValueError("Native audio adapters support exactly one request at a time.")
        if self.mode == "interleaved":
            self.left -= 1
        event = None
        if self.modality == 1:
            token = int(logits.argmax(-1).item())
            self.use_audio.fill_(False)
            if self.mode != "interleaved" or token != self.config.eos_token_id:
                event = [1, token]
            if token == self.config.eos_token_id:
                self.active = False
            elif self.mode == "sequential" and token == self.config.audio_start_token_id:
                self.modality = 3
            elif self.mode == "interleaved":
                if token == self.config.text_end_token_id:
                    self.text_done = True
                if self.left == 0 or self.text_done:
                    self.modality = 3
                    self.left = self.config.interleaved_n_audio
        else:
            if self.graph is None:
                self.graph = depth_graph(lambda h: self.model._sample_audio_frame(h, 0.0, 1), hidden)
            frame = self.graph(hidden).clone()
            if int(frame[0].item()) == self.config.audio_eos_token_id:
                frame.fill_(self.config.audio_eos_token_id)
                self.modality = 1
            elif self.mode == "interleaved" and self.left == 0 and not self.text_done:
                self.modality = 1
                self.left = self.config.interleaved_n_text
            offset_frame = frame + self.model.model.get_codebook_offsets(frame.device)
            self.next_embedding.copy_(self.model.model.audio_embedding(offset_frame).sum(0))
            self.use_audio.fill_(True)
            # A transport token reserves one native cache position for this audio frame.
            logits.fill_(-float("inf"))
            logits[:, self.config.audio_token_id] = 0
            event = [3, *frame.tolist()]
        if event is not None:
            self.events.append(event)
        return logits, event


def output_tensors(events, device="cuda"):
    events = [event for event in events if event is not None]
    text = [event[1] for event in events if event[0] == 1]
    audio = [event[1:] for event in events if event[0] == 3]
    return Lfm2AudioGenerateOutput(
        sequences=torch.tensor([text], dtype=torch.long, device=device),
        audio_codes=torch.tensor(audio, dtype=torch.long, device=device).reshape(-1, 8).T[None],
        modalities=torch.tensor([[event[0] for event in events]], dtype=torch.long, device=device),
    )
