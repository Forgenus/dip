from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowMetadata:
    valid_seconds: float
    padding_seconds: float
    padding_ratio: float
    reliability: str
    skipped: bool


def reliability_for_query_duration(valid_seconds, min_query_seconds=2.0):
    if valid_seconds < min_query_seconds:
        return "skipped"
    if valid_seconds < 3.0:
        return "low"
    if valid_seconds < 5.0:
        return "medium"
    return "high"


def prepare_query_window(audio, sample_rate, window_seconds, min_query_seconds=2.0):
    _validate_sample_rate(sample_rate)
    window_samples = int(round(sample_rate * window_seconds))
    valid_samples = min(len(audio), window_samples)
    valid_seconds = valid_samples / sample_rate
    reliability = reliability_for_query_duration(valid_seconds, min_query_seconds)
    skipped = reliability == "skipped"

    window = np.zeros(window_samples, dtype=np.float32)
    if valid_samples:
        window[:valid_samples] = np.asarray(audio[:valid_samples], dtype=np.float32)

    return window, _metadata(
        target_samples=window_samples,
        valid_samples=valid_samples,
        sample_rate=sample_rate,
        valid_seconds=valid_seconds,
        reliability=reliability,
        skipped=skipped,
    )


def crop_candidate_window(audio, sample_rate, start_seconds, window_seconds):
    _validate_sample_rate(sample_rate)
    window_samples = int(round(sample_rate * window_seconds))
    start_sample = int(round(start_seconds * sample_rate))
    end_sample = start_sample + window_samples

    source_start = max(start_sample, 0)
    source_end = min(end_sample, len(audio))
    valid_samples = max(0, source_end - source_start)
    valid_seconds = valid_samples / sample_rate
    skipped = valid_samples == 0

    window = np.zeros(window_samples, dtype=np.float32)
    if valid_samples:
        target_start = source_start - start_sample
        target_end = target_start + valid_samples
        window[target_start:target_end] = np.asarray(
            audio[source_start:source_end],
            dtype=np.float32,
        )

    return window, _metadata(
        target_samples=window_samples,
        valid_samples=valid_samples,
        sample_rate=sample_rate,
        valid_seconds=valid_seconds,
        reliability="skipped" if skipped else "high",
        skipped=skipped,
    )


def _validate_sample_rate(sample_rate):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")


def _metadata(
    target_samples,
    valid_samples,
    sample_rate,
    valid_seconds,
    reliability,
    skipped,
):
    padding_samples = max(0, target_samples - valid_samples)
    padding_seconds = padding_samples / sample_rate
    padding_ratio = padding_samples / target_samples if target_samples else 0.0
    return WindowMetadata(
        valid_seconds=valid_seconds,
        padding_seconds=padding_seconds,
        padding_ratio=padding_ratio,
        reliability=reliability,
        skipped=skipped,
    )
