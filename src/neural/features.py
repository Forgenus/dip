from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio


def build_symmetric_pair_features(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.dim() != 3 or right.dim() != 3:
        raise ValueError("mel tensors must have shape [batch, n_mels, frames]")
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("mel tensors must have matching batch and n_mels dimensions")

    left, right = _align_feature_frames(left, right)
    if left.shape[-1] == 0:
        raise ValueError("mel tensors must contain at least one frame")

    mean = (left + right) / 2.0
    abs_difference = torch.abs(left - right)
    return torch.stack((mean, abs_difference), dim=1)


class TorchMelPairFeatureExtractor(nn.Module):
    """Build symmetric pair features from two waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.build_symmetric_pair_features(left, right)

    def build_symmetric_pair_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left_audio, right_audio = self._prepare_pair_audio(left, right)
        left_mel = self.to_db(self.mel(left_audio))
        right_mel = self.to_db(self.mel(right_audio))
        return build_symmetric_pair_features(left_mel, right_mel)

    @staticmethod
    def _prepare_audio(audio: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio)
        if not audio.is_floating_point():
            audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() != 2:
            raise ValueError("audio tensors must have shape [time] or [batch, time]")
        if audio.shape[-1] == 0:
            raise ValueError("audio tensors must contain at least one sample")
        if not torch.all(torch.isfinite(audio)):
            raise ValueError("audio tensors must contain only finite values")
        return audio

    @classmethod
    def _prepare_pair_audio(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_audio = cls._prepare_audio(left)
        right_audio = cls._prepare_audio(right)
        if left_audio.shape[0] != right_audio.shape[0]:
            raise ValueError("left and right audio must have the same batch size")

        frame_count = max(left_audio.shape[-1], right_audio.shape[-1])
        left_audio = _pad_time(left_audio, frame_count)
        right_audio = _pad_time(right_audio, frame_count)
        return left_audio, right_audio


def _align_feature_frames(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = max(left.shape[-1], right.shape[-1])
    return (_pad_time(left, frame_count), _pad_time(right, frame_count))


def _pad_time(values: torch.Tensor, frame_count: int) -> torch.Tensor:
    missing = frame_count - values.shape[-1]
    if missing <= 0:
        return values[..., :frame_count]
    return F.pad(values, (0, missing))
