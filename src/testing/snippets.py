from pathlib import Path

import librosa
import numpy as np

import config as cfg


def _get_snippet_from_file(
    file_path: Path,
    snippet_duration: float,
    rng,
) -> np.ndarray:
    sr = cfg.SAMPLE_RATE

    full_audio, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(full_audio) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    snippet_samples = int(snippet_duration * sr)

    if snippet_samples <= 0:
        raise ValueError(f"Invalid snippet_duration: {snippet_duration}")

    if len(full_audio) <= snippet_samples:
        return full_audio

    max_start_sample = len(full_audio) - snippet_samples
    start_sample = rng.integers(0, max_start_sample + 1)
    end_sample = start_sample + snippet_samples

    return full_audio[start_sample:end_sample]
