"""Owned CPU audio chunks, with exact buffering or opt-in prefix decoding."""

import torch


class BufferedAudioStream:
    """Deliver exact waveform chunks once the final decoder shape is known.

    This preserves full-decode bits, but cannot reduce generation latency: no
    samples are sent before the original full waveform has been computed.
    """

    def __init__(self, callback, chunk_samples):
        self.callback = callback
        self.chunk_samples = chunk_samples

    def finish(self, waveform):
        for start in range(0, waveform.shape[-1], self.chunk_samples):
            self.callback(
                waveform[:, start : start + self.chunk_samples].to(
                    device="cpu", dtype=torch.float32, copy=True
                )
            )
        return waveform


class PrefixAudioStream:
    """Deliver owned CPU chunks while retaining the samples actually delivered.

    Each decode includes all preceding codes, so convolution/attention context is
    retained. Different decoder bucket sizes can change floating-point rounding,
    including with FP32 arithmetic. Use BufferedAudioStream for full-decode bits.
    """

    def __init__(self, decoder, callback, chunk_frames, eos_token_id):
        self.decoder = decoder
        self.callback = callback
        self.chunk_frames = chunk_frames
        self.eos_token_id = eos_token_id
        self.frames = []
        self.chunks = []
        self.samples_emitted = 0
        self.ended = False

    def append(self, frame):
        # Match the model's first-codebook EOS rule before decoding any frame.
        if int(frame[0].item()) == self.eos_token_id:
            self.ended = True
            return
        if self.ended:
            raise ValueError("Audio frames after EOS cannot be streamed as one utterance.")
        if bool(((frame < 0) | (frame >= self.eos_token_id)).any()):
            raise ValueError("Invalid audio code in streamed frame.")
        self.frames.append(frame.clone())
        if len(self.frames) % self.chunk_frames == 0:
            codes = torch.stack(self.frames, dim=-1).unsqueeze(0)
            self._emit(self.decoder(codes))

    def _emit(self, waveform):
        if waveform.shape[-1] <= self.samples_emitted:
            return
        chunk = waveform[:, self.samples_emitted :].clone()
        self.chunks.append(chunk)
        self.samples_emitted = waveform.shape[-1]
        # The synchronous copy makes the callback an actual playable-audio
        # boundary. A consumer may retain or modify its private copy.
        self.callback(chunk.to(device="cpu", dtype=torch.float32, copy=True))

    def finish(self, waveform):
        self._emit(waveform)
        if self.samples_emitted != waveform.shape[-1]:
            raise ValueError("Streamed audio length differs from the final code sequence.")
        # Return exactly the samples sent to the consumer, including early
        # prefixes; do not silently replace them with a later full decode.
        return torch.cat(self.chunks, dim=-1) if self.chunks else waveform

    def finish_codes(self, codes, samples_per_frame):
        """Decode only an undelivered tail, using the same prefix decoder."""
        expected_samples = codes.shape[-1] * samples_per_frame
        if self.samples_emitted > expected_samples:
            raise ValueError("Streamed audio length differs from the final code sequence.")
        if self.samples_emitted < expected_samples:
            return self.finish(self.decoder(codes))
        if self.chunks:
            return torch.cat(self.chunks, dim=-1)
        return torch.empty((1, 0), device=codes.device, dtype=torch.float32)
