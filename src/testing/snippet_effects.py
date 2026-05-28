from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class SnippetEffects:
    noise: bool = False
    noise_level: float = 0.02
    volume: str = "none"
    volume_factor: float = 1.5
    time_stretch_rate: float = 1.0


def apply_snippet_effects(
    audio,
    sample_rate: int,
    rng,
    effects: SnippetEffects,
) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32)

    if effects.time_stretch_rate <= 0:
        raise ValueError(f"Invalid time_stretch_rate: {effects.time_stretch_rate}")

    if effects.volume_factor <= 0:
        raise ValueError(f"Invalid volume_factor: {effects.volume_factor}")

    if effects.noise_level < 0:
        raise ValueError(f"Invalid noise_level: {effects.noise_level}")

    if effects.time_stretch_rate != 1.0:
        processed = librosa.effects.time_stretch(
            y=processed,
            rate=effects.time_stretch_rate,
        ).astype(np.float32, copy=False)

    volume_multiplier = volume_factor_for_mode(
        mode=effects.volume,
        factor=effects.volume_factor,
        rng=rng,
    )
    if volume_multiplier != 1.0:
        processed = processed * np.float32(volume_multiplier)

    if effects.noise:
        noise = rng.normal(
            loc=0.0,
            scale=effects.noise_level,
            size=len(processed),
        ).astype(np.float32, copy=False)
        processed = processed + noise

    return processed.astype(np.float32, copy=False)


def volume_factor_for_mode(mode: str, factor: float, rng) -> float:
    if mode == "none":
        return 1.0
    if mode == "up":
        return factor
    if mode == "down":
        return 1.0 / factor
    if mode == "random":
        return float(rng.uniform(1.0 / factor, factor))

    raise ValueError(f"Unsupported volume mode: {mode}")
