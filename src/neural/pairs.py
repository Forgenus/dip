"""Sampling utilities for neural pair training."""

POSITIVE_SAME_TIME = "positive_same_time"
POSITIVE_JITTERED = "positive_jittered"
NEGATIVE_RANDOM = "negative_random"
NEGATIVE_HARD = "negative_hard"

_FULL_QUERY_RATIO = 0.70
_MEDIUM_QUERY_RATIO = 0.20
_SAME_TIME_POSITIVE_RATIO = 0.70


def choose_query_valid_seconds(rng, window_seconds: float = 5.0) -> float:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    draw = float(rng.random())
    if draw < _FULL_QUERY_RATIO:
        return float(window_seconds)
    if draw < _FULL_QUERY_RATIO + _MEDIUM_QUERY_RATIO:
        return _uniform(rng, min(3.0, window_seconds), window_seconds)
    return _uniform(rng, min(2.0, window_seconds), min(3.0, window_seconds))


def sample_pair_kind(rng, positive_ratio: float, hard_negative_ratio: float) -> str:
    _validate_ratio("positive_ratio", positive_ratio)
    _validate_ratio("hard_negative_ratio", hard_negative_ratio)
    if positive_ratio + hard_negative_ratio > 1.0:
        raise ValueError("positive_ratio + hard_negative_ratio must be <= 1.0")

    draw = float(rng.random())
    if draw < positive_ratio:
        if float(rng.random()) < _SAME_TIME_POSITIVE_RATIO:
            return POSITIVE_SAME_TIME
        return POSITIVE_JITTERED
    if draw < positive_ratio + hard_negative_ratio:
        return NEGATIVE_HARD
    return NEGATIVE_RANDOM


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _uniform(rng, low: float, high: float) -> float:
    if high <= low:
        return float(high)
    return float(rng.uniform(low, high))
