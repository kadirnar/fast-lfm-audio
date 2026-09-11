import torch
from sglang.srt.models.lfm2 import Lfm2ForCausalLM as NativeLfm2

from ..audio import AudioState


class Lfm2ForCausalLM(NativeLfm2):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        source = getattr(config, "fast_lfm_audio_source", None)
        self.audio_state = AudioState(source) if source is not None else None

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        if self.audio_state is None:
            return super().forward(input_ids, positions, forward_batch, input_embeds, **kwargs)
        if input_embeds is None:
            input_embeds = self.audio_state.embed(input_ids, self.model.embed_tokens(input_ids))
        hidden = self.model(input_ids, positions, forward_batch, input_embeds)
        output = self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)
        # The sampling hook consumes this after graph replay; it is not sent to the client.
        output.hidden_states = hidden[-1:]
        return output


EntryClass = Lfm2ForCausalLM
