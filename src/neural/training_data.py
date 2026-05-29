from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config as cfg
from src.neural.augmentations import QueryAugmentationConfig, apply_query_augmentations
from src.neural.audio_windows import crop_candidate_window, prepare_query_window
from src.neural.dataset import (
    PairExample,
    duration_bucket,
    make_hard_negative,
    make_jittered_positive,
    make_random_negative,
    make_same_time_positive,
)
from src.neural.features import TorchMelPairFeatureExtractor
from src.neural.pairs import (
    NEGATIVE_HARD,
    NEGATIVE_RANDOM,
    POSITIVE_JITTERED,
    POSITIVE_SAME_TIME,
    choose_query_valid_seconds,
    sample_pair_kind,
)
from src.neural.splits import SongSplitItem
from src.processing import preprocess as pp


@dataclass
class TrainingBatch:
    features: torch.Tensor
    label: torch.Tensor
    pair_type: str
    duration_bucket: str


class NeuralPairDataset(Dataset):
    def __init__(
        self,
        items: list[SongSplitItem],
        examples_per_epoch: int,
        sample_rate: int = cfg.SAMPLE_RATE,
        window_seconds: float = cfg.NEURAL_WINDOW_SECONDS,
        n_mels: int = cfg.NEURAL_N_MELS,
        n_fft: int = cfg.NEURAL_MEL_N_FFT,
        hop_length: int = cfg.NEURAL_MEL_HOP_LENGTH,
        positive_ratio: float = getattr(cfg, "NEURAL_POSITIVE_RATIO", 0.5),
        hard_negative_ratio: float = getattr(cfg, "NEURAL_HARD_NEGATIVE_RATIO", 0.2),
        positive_jitter_seconds: float = getattr(cfg, "NEURAL_POSITIVE_JITTER_SECONDS", 0.5),
        min_query_seconds: float = cfg.NEURAL_MIN_QUERY_SECONDS,
        query_augmentation_config: QueryAugmentationConfig | None = None,
        seed: int = 0,
    ) -> None:
        if examples_per_epoch < 0:
            raise ValueError("examples_per_epoch must be non-negative")
        if not items:
            raise ValueError("items must contain at least one song")

        self.items = list(items)
        self.examples_per_epoch = int(examples_per_epoch)
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.positive_ratio = float(positive_ratio)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.positive_jitter_seconds = float(positive_jitter_seconds)
        self.min_query_seconds = float(min_query_seconds)
        self.query_augmentation_config = (
            query_augmentation_config
            if query_augmentation_config is not None
            else _query_augmentation_config_from_settings()
        )
        self.seed = int(seed)
        self.audio_cache: dict[str, np.ndarray] = {}
        self.feature_extractor = TorchMelPairFeatureExtractor(
            sample_rate=self.sample_rate,
            n_mels=int(n_mels),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
        )
        self.feature_extractor.eval()

    def __len__(self) -> int:
        return self.examples_per_epoch

    def __getitem__(self, index: int) -> TrainingBatch:
        pair = self._sample_pair(index)
        query_audio = self._load_audio(pair.query_file_path)
        candidate_audio = self._load_audio(pair.candidate_file_path)

        query_start = int(round(pair.query_start_seconds * self.sample_rate))
        query_samples = int(round(pair.query_valid_seconds * self.sample_rate))
        query_segment = query_audio[query_start : query_start + query_samples]
        query_segment = apply_query_augmentations(
            query_segment,
            sample_rate=self.sample_rate,
            rng=np.random.default_rng(self.seed + int(index) + 1_000_000),
            config=self.query_augmentation_config,
        )

        query_window, _query_meta = prepare_query_window(
            query_segment,
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            min_query_seconds=self.min_query_seconds,
        )
        candidate_window, candidate_meta = crop_candidate_window(
            candidate_audio,
            sample_rate=self.sample_rate,
            start_seconds=pair.candidate_start_seconds,
            window_seconds=self.window_seconds,
        )
        if candidate_meta.skipped:
            raise ValueError("candidate window contains no audio")

        query_tensor = torch.as_tensor(query_window, dtype=torch.float32)
        candidate_tensor = torch.as_tensor(candidate_window, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Empty filters detected in mel frequency basis.*",
                category=UserWarning,
            )
            with torch.no_grad():
                features = self.feature_extractor(query_tensor, candidate_tensor).squeeze(0)

        return TrainingBatch(
            features=features,
            label=torch.tensor(float(pair.label), dtype=torch.float32),
            pair_type=pair.pair_type,
            duration_bucket=duration_bucket(pair.query_valid_seconds),
        )

    def _sample_pair(self, index: int) -> PairExample:
        rng = np.random.default_rng(self.seed + int(index))
        pair_kind = sample_pair_kind(
            rng,
            positive_ratio=self.positive_ratio,
            hard_negative_ratio=self.hard_negative_ratio,
        )
        query_song = self._song_dict(self.items[int(rng.integers(0, len(self.items)))])
        query_valid_seconds = choose_query_valid_seconds(rng, self.window_seconds)
        query_start_seconds = self._sample_start_seconds(
            rng,
            duration_seconds=float(query_song["duration"]),
            valid_seconds=query_valid_seconds,
        )

        if pair_kind == POSITIVE_SAME_TIME:
            return make_same_time_positive(
                query_song,
                start_seconds=query_start_seconds,
                query_valid_seconds=query_valid_seconds,
                window_seconds=self.window_seconds,
            )
        if pair_kind == POSITIVE_JITTERED:
            jitter = float(
                rng.uniform(-self.positive_jitter_seconds, self.positive_jitter_seconds)
            )
            return make_jittered_positive(
                query_song,
                query_start_seconds=query_start_seconds,
                jitter_seconds=jitter,
                query_valid_seconds=query_valid_seconds,
                window_seconds=self.window_seconds,
            )

        candidate_song = self._choose_different_song(rng, query_song)
        if pair_kind == NEGATIVE_HARD:
            return make_hard_negative(
                query_song,
                candidate_song,
                query_start_seconds=query_start_seconds,
                candidate_offset_seconds=self._clamp_start_seconds(
                    query_start_seconds,
                    duration_seconds=float(candidate_song["duration"]),
                ),
                query_valid_seconds=query_valid_seconds,
                window_seconds=self.window_seconds,
            )
        if pair_kind == NEGATIVE_RANDOM:
            return make_random_negative(
                query_song,
                candidate_song,
                query_start_seconds=query_start_seconds,
                candidate_start_seconds=self._sample_start_seconds(
                    rng,
                    duration_seconds=float(candidate_song["duration"]),
                    valid_seconds=self.window_seconds,
                ),
                query_valid_seconds=query_valid_seconds,
                window_seconds=self.window_seconds,
            )

        raise ValueError(f"unknown pair kind: {pair_kind}")

    def _choose_different_song(self, rng, query_song: dict) -> dict:
        if len(self.items) < 2:
            raise ValueError("negative pairs require at least two songs")

        query_song_id = int(query_song["song_id"])
        choices = [item for item in self.items if int(item.song_id) != query_song_id]
        return self._song_dict(choices[int(rng.integers(0, len(choices)))])

    def _load_audio(self, file_path: str) -> np.ndarray:
        resolved = self._resolve_path(file_path)
        cache_key = str(resolved)
        if cache_key not in self.audio_cache:
            self.audio_cache[cache_key] = pp.load_audio(resolved, target_sr=self.sample_rate)
        return self.audio_cache[cache_key]

    @staticmethod
    def _resolve_path(file_path: str) -> Path:
        path = Path(file_path)
        if path.is_absolute():
            return path
        return cfg.BASE_DIR / path

    @staticmethod
    def _song_dict(item: SongSplitItem) -> dict:
        return {
            "song_id": int(item.song_id),
            "file_path": str(item.file_path),
            "duration": float(item.duration),
        }

    def _sample_start_seconds(self, rng, duration_seconds: float, valid_seconds: float) -> float:
        max_start = max(0.0, float(duration_seconds) - float(valid_seconds))
        if max_start == 0.0:
            return 0.0
        return float(rng.uniform(0.0, max_start))

    def _clamp_start_seconds(self, start_seconds: float, duration_seconds: float) -> float:
        max_start = max(0.0, float(duration_seconds) - self.window_seconds)
        return min(max(0.0, float(start_seconds)), max_start)


def make_training_loader(
    dataset: NeuralPairDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_training_batches,
    )


def _query_augmentation_config_from_settings() -> QueryAugmentationConfig:
    return QueryAugmentationConfig(
        enabled=getattr(cfg, "NEURAL_QUERY_AUGMENTATION_ENABLED", True),
        noise_min=getattr(cfg, "NEURAL_QUERY_NOISE_MIN", 0.01),
        noise_max=getattr(cfg, "NEURAL_QUERY_NOISE_MAX", 0.03),
        volume_min=getattr(cfg, "NEURAL_QUERY_VOLUME_MIN", 0.25),
        volume_max=getattr(cfg, "NEURAL_QUERY_VOLUME_MAX", 3.0),
        time_stretch_min=getattr(cfg, "NEURAL_QUERY_TIME_STRETCH_MIN", 0.97),
        time_stretch_max=getattr(cfg, "NEURAL_QUERY_TIME_STRETCH_MAX", 1.03),
    )


def _collate_training_batches(items: list[TrainingBatch]) -> dict:
    return {
        "features": torch.stack([item.features for item in items]),
        "labels": torch.stack([item.label for item in items]),
        "pair_type": [item.pair_type for item in items],
        "duration_bucket": [item.duration_bucket for item in items],
    }
