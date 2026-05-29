"""Sampling helpers for neural pair training examples."""

PAIR_KIND_POSITIVE_SAME_TIME = "positive_same_time"
PAIR_KIND_POSITIVE_JITTERED = "positive_jittered"
PAIR_KIND_NEGATIVE_RANDOM = "negative_random"
PAIR_KIND_NEGATIVE_HARD = "negative_hard"

PAIR_KINDS = (
    PAIR_KIND_POSITIVE_SAME_TIME,
    PAIR_KIND_POSITIVE_JITTERED,
    PAIR_KIND_NEGATIVE_RANDOM,
    PAIR_KIND_NEGATIVE_HARD,
)

_FULL_DURATION_PROBABILITY = 0.70
_MEDIUM_DURATION_PROBABILITY = 0.20
_SAME_TIME_POSITIVE_PROBABILITY = 0.70


def choose_query_valid_seconds(rng, window_seconds: float = 5.0) -> float:
    """Choose a query duration, biased toward full-window snippets."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    draw = _random_float(rng)
    if draw < _FULL_DURATION_PROBABILITY:
        return float(window_seconds)

    if draw < _FULL_DURATION_PROBABILITY + _MEDIUM_DURATION_PROBABILITY:
        return _uniform_seconds(rng, min(3.0, window_seconds), window_seconds)

    return _uniform_seconds(rng, min(2.0, window_seconds), min(3.0, window_seconds))


def sample_pair_kind(rng, positive_ratio: float, hard_negative_ratio: float) -> str:
    """Sample a stable pair kind name for metrics and future logging."""
    _validate_ratio("positive_ratio", positive_ratio)
    _validate_ratio("hard_negative_ratio", hard_negative_ratio)
    if positive_ratio + hard_negative_ratio > 1.0:
        raise ValueError("positive_ratio + hard_negative_ratio must be <= 1.0")

    draw = _random_float(rng)
    if draw < positive_ratio:
        if _random_float(rng) < _SAME_TIME_POSITIVE_PROBABILITY:
            return PAIR_KIND_POSITIVE_SAME_TIME
        return PAIR_KIND_POSITIVE_JITTERED

    if draw < positive_ratio + hard_negative_ratio:
        return PAIR_KIND_NEGATIVE_HARD

    return PAIR_KIND_NEGATIVE_RANDOM


def _validate_ratio(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _uniform_seconds(rng, low: float, high: float) -> float:
    if high <= low:
        return float(high)
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(low + (high - low) * _random_float(rng))


def _random_float(rng) -> float:
    if hasattr(rng, "random"):
        return float(rng.random())
    raise TypeError("rng must provide random() and optionally uniform()")
