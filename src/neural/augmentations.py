from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class QueryAugmentationConfig:
    enabled: bool = True
    noise_min: float = 0.01
    noise_max: float = 0.03
    volume_min: float = 0.25
    volume_max: float = 3.0
    time_stretch_min: float = 0.97
    time_stretch_max: float = 1.03


def apply_query_augmentations(
    audio: np.ndarray,
    sample_rate: int,
    rng,
    config: QueryAugmentationConfig,
) -> np.ndarray:
    _validate_config(config)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    processed = np.asarray(audio, dtype=np.float32)
    if not config.enabled or processed.size == 0:
        return processed

    original_samples = len(processed)
    stretch_rate = float(rng.uniform(config.time_stretch_min, config.time_stretch_max))
    if stretch_rate != 1.0:
        processed = librosa.effects.time_stretch(processed, rate=stretch_rate).astype(
            np.float32,
            copy=False,
        )
        processed = _fit_length(processed, original_samples)

    volume = float(rng.uniform(config.volume_min, config.volume_max))
    processed = processed * np.float32(volume)

    noise_level = float(rng.uniform(config.noise_min, config.noise_max))
    if noise_level > 0.0:
        processed = processed + rng.normal(
            loc=0.0,
            scale=noise_level,
            size=processed.shape,
        ).astype(np.float32)

    return np.clip(processed, -1.0, 1.0).astype(np.float32, copy=False)


def _fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    if len(audio) == samples:
        return audio.astype(np.float32, copy=False)
    if len(audio) > samples:
        return audio[:samples].astype(np.float32, copy=False)

    output = np.zeros(samples, dtype=np.float32)
    output[: len(audio)] = audio.astype(np.float32, copy=False)
    return output


def _validate_config(config: QueryAugmentationConfig) -> None:
    _validate_range("noise", config.noise_min, config.noise_max, minimum=0.0)
    _validate_range("volume", config.volume_min, config.volume_max, minimum=0.0)
    _validate_range("time_stretch", config.time_stretch_min, config.time_stretch_max, minimum=0.0)


def _validate_range(name: str, lower: float, upper: float, minimum: float) -> None:
    if lower < minimum or upper < minimum or upper < lower:
        raise ValueError(f"invalid {name} range: {lower}..{upper}")
