"""The reference generation loop, with an injectable exact text projection.

Adapted from Transformers PR #48249. See THIRD_PARTY_NOTICES.md. The original
loop is used if the installed implementation differs from the audited source.
"""

import hashlib
import inspect
import textwrap

import torch
from transformers.models.lfm2_audio.modeling_lfm2_audio import (
    AUDIO_OUTPUT_MODALITY,
    TEXT_MODALITY,
    Lfm2AudioForConditionalGeneration,
    Lfm2AudioGenerateOutput,
)

SOURCE_HASH = "f78b65a818a7f99041a19cec0328805b39f8dd5fb2553462c7ad6dbcb2d679ce"


def matches_reference(function):
    reference = Lfm2AudioForConditionalGeneration.generate
    if getattr(function, "__func__", None) is not reference:
        return False
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(reference)))
    except (OSError, TypeError):
        return False
    return hashlib.sha256(source.encode()).hexdigest() == SOURCE_HASH


@torch.no_grad()
def generate(
    self,
    project,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features: torch.FloatTensor | None = None,
    input_features_attention_mask: torch.Tensor | None = None,
    modality_ids: torch.LongTensor | None = None,
    audio_codes: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    max_new_tokens: int = 256,
    generation_mode: str = "sequential",
    text_temperature: float | None = None,
    text_top_k: int | None = None,
    audio_temperature: float | None = None,
    audio_top_k: int | None = None,
    **kwargs,
) -> Lfm2AudioGenerateOutput:
    """Generate text tokens and 8-codebook audio frames from a single prompt."""
    if generation_mode not in {"sequential", "interleaved"}:
        raise ValueError("`generation_mode` must be either 'sequential' or 'interleaved'.")

    input_embeddings, _, _ = self.model._prepare_inputs_embeds(
        input_ids,
        attention_mask,
        input_features,
        input_features_attention_mask,
        modality_ids,
        audio_codes,
        inputs_embeds,
    )
    if input_embeddings.shape[0] != 1:
        raise ValueError("LFM2-Audio generation currently supports a batch size of 1.")

    current_input = input_embeddings
    current_modality = TEXT_MODALITY
    modality_left = self.config.interleaved_n_text
    text_done = False
    past_key_values = None
    generated_text = []
    generated_audio = []
    generated_modalities = []
    generation_attention_mask = attention_mask
    if generation_attention_mask is not None and bool(generation_attention_mask.bool().all()):
        generation_attention_mask = None

    for _ in range(max_new_tokens):
        outputs = self.model.lfm(
            inputs_embeds=current_input,
            attention_mask=generation_attention_mask if past_key_values is None else None,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        hidden_state = outputs.last_hidden_state[0, -1]
        past_key_values = outputs.past_key_values

        if generation_mode == "interleaved":
            modality_left -= 1

        if current_modality == TEXT_MODALITY:
            text_logits = project(hidden_state)
            next_token = self._sample(text_logits, temperature=text_temperature, top_k=text_top_k)
            token_id = int(next_token.item())
            # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B treats interleaved EOS as a non-yielded stream terminator.
            if generation_mode == "interleaved" and token_id == self.config.eos_token_id:
                break
            generated_text.append(next_token)
            generated_modalities.append(TEXT_MODALITY)

            # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 7 (`<|im_end|>`) to finish a response.
            if token_id == self.config.eos_token_id:
                break
            # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 128 to switch sequential generation to audio.
            if generation_mode == "sequential" and token_id == self.config.audio_start_token_id:
                current_modality = AUDIO_OUTPUT_MODALITY
            elif generation_mode == "interleaved":
                # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses token 130 to mark the end of interleaved text.
                if token_id == self.config.text_end_token_id:
                    text_done = True
                if modality_left == 0 or text_done:
                    current_modality = AUDIO_OUTPUT_MODALITY
                    modality_left = self.config.interleaved_n_audio
            current_input = self.get_input_embeddings()(next_token)[None]
        else:
            next_frame = self._sample_audio_frame(
                hidden_state, temperature=audio_temperature, top_k=audio_top_k
            )
            # CODEPATH: LiquidAI/LFM2.5-Audio-1.5B uses Mimi code 2048 to mark the end of generated audio.
            if int(next_frame[0].item()) == self.config.audio_eos_token_id:
                next_frame = torch.full_like(next_frame, self.config.audio_eos_token_id)
                current_modality = TEXT_MODALITY
            elif generation_mode == "interleaved" and modality_left == 0 and not text_done:
                current_modality = TEXT_MODALITY
                modality_left = self.config.interleaved_n_text

            generated_audio.append(next_frame)
            generated_modalities.append(AUDIO_OUTPUT_MODALITY)
            offset_frame = next_frame + self.model.get_codebook_offsets(next_frame.device)
            current_input = self.model.audio_embedding(offset_frame).sum(0)[None, None]

    if generated_text:
        sequences = torch.cat(generated_text).unsqueeze(0)
    else:
        sequences = torch.empty((1, 0), dtype=torch.long, device=input_embeddings.device)
    if generated_audio:
        output_audio_codes = torch.stack(generated_audio, dim=-1).unsqueeze(0)
    else:
        output_audio_codes = torch.empty(
            (1, self.config.codebooks, 0), dtype=torch.long, device=input_embeddings.device
        )
    modalities = torch.tensor(
        generated_modalities, dtype=torch.long, device=input_embeddings.device
    ).unsqueeze(0)
    return Lfm2AudioGenerateOutput(
        sequences=sequences,
        audio_codes=output_audio_codes,
        modalities=modalities,
    )
