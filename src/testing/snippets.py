from pathlib import Path
from dataclasses import dataclass
import re

import librosa
import numpy as np
import soundfile as sf

import config as cfg


@dataclass
class AudioSnippet:
    audio: np.ndarray
    source_file: Path
    start_sample: int
    start_seconds: float
    duration_seconds: float
    sample_rate: int


def create_snippet_from_file(
    file_path: Path,
    snippet_duration: float,
    rng,
    align_to_hop: bool = False,
) -> AudioSnippet:
    sr = cfg.SAMPLE_RATE
    file_path = Path(file_path)

    full_audio, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(full_audio) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    snippet_samples = int(snippet_duration * sr)

    if snippet_samples <= 0:
        raise ValueError(f"Invalid snippet_duration: {snippet_duration}")

    if len(full_audio) <= snippet_samples:
        return AudioSnippet(
            audio=full_audio,
            source_file=file_path,
            start_sample=0,
            start_seconds=0.0,
            duration_seconds=len(full_audio) / sr,
            sample_rate=sr,
        )

    max_start_sample = len(full_audio) - snippet_samples
    if align_to_hop:
        max_start_frame = max_start_sample // cfg.HOP_LENGTH
        start_sample = int(rng.integers(0, max_start_frame + 1)) * cfg.HOP_LENGTH
    else:
        start_sample = int(rng.integers(0, max_start_sample + 1))
    end_sample = start_sample + snippet_samples

    return AudioSnippet(
        audio=full_audio[start_sample:end_sample],
        source_file=file_path,
        start_sample=start_sample,
        start_seconds=start_sample / sr,
        duration_seconds=snippet_samples / sr,
        sample_rate=sr,
    )


def _get_snippet_from_file(
    file_path: Path,
    snippet_duration: float,
    rng,
) -> np.ndarray:
    return create_snippet_from_file(file_path, snippet_duration, rng).audio


def save_snippet_to_file(snippet: AudioSnippet, output_file: Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, snippet.audio, snippet.sample_rate, subtype="FLOAT")


def sanitize_filename_part(value: str, max_length: int = 80) -> str:
    value = str(value)
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", value)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("._ ")
    if not sanitized:
        sanitized = "untitled"
    return sanitized[:max_length].rstrip("._ ")


def build_failed_snippet_filename(
    index: int,
    expected_id: int,
    expected_title: str,
    found_id: int,
    start_seconds: float,
) -> str:
    title = sanitize_filename_part(expected_title, max_length=60)
    found_part = "found_none" if found_id == -1 else f"found_{found_id}"
    return (
        f"failed_{index:03d}_expected_{expected_id}_{title}_"
        f"start_{start_seconds:.2f}s_{found_part}.wav"
    )


def build_test_snippet_filename(
    index: int,
    expected_id: int,
    expected_title: str,
    start_seconds: float,
) -> str:
    title = sanitize_filename_part(expected_title, max_length=60)
    return (
        f"test_{index:03d}_expected_{expected_id}_{title}_"
        f"start_{start_seconds:.2f}s.wav"
    )


def build_effect_snippet_filename(
    expected_id: int,
    expected_title: str,
    start_seconds: float,
    effect_name: str,
) -> str:
    title = sanitize_filename_part(expected_title, max_length=60)
    effect = sanitize_filename_part(effect_name, max_length=30)
    return (
        f"effect_expected_{expected_id}_{title}_"
        f"start_{start_seconds:.2f}s_{effect}.wav"
    )
