from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MatchSelection:
    song_id: int
    time_offset: int
    score: float
    max_count: int
    total_matches: int = 0
    expected_score: float | None = None
    expected_max_count: int | None = None
    reason: str = ""


def compute_candidate_score(max_count: int, total_matches: int) -> float:
    if total_matches <= 0:
        return 0.0

    return max_count / total_matches


def select_best_match(
    results: Dict[int, Tuple[int, int]],
    total_matches: int,
    expected_id: int = -1,
    min_offset_peak: int = 0,
    min_score: float = 0.0,
    min_margin: float = 0.0,
) -> MatchSelection:
    if not results or total_matches <= 0:
        return MatchSelection(
            song_id=-1,
            time_offset=0,
            score=0.0,
            max_count=0,
            total_matches=total_matches,
            expected_score=None,
            expected_max_count=None,
            reason="no candidates"
        )

    candidates: list[tuple[float, int, int, int]] = []

    for song_id, (max_count, time_offset) in results.items():
        if max_count < min_offset_peak:
            continue

        score = compute_candidate_score(max_count, total_matches)

        if score < min_score:
            continue

        candidates.append((score, song_id, max_count, time_offset))

    if not candidates:
        return MatchSelection(
            song_id=-1,
            time_offset=0,
            score=0.0,
            max_count=0,
            total_matches=total_matches,
            expected_score=None,
            expected_max_count=None,
            reason="no candidates passed thresholds"
        )

    candidates.sort(reverse=True)
    best_score, best_song_id, best_max_count, best_time_offset = candidates[0]

    if len(candidates) > 1:
        second_score = candidates[1][0]
        if best_score - second_score < min_margin:
            return MatchSelection(
                song_id=-1,
                time_offset=0,
                score=0.0,
                max_count=0,
                total_matches=total_matches,
                expected_score=None,
                expected_max_count=None,
                reason="margin threshold not met"
            )

    expected_score = None
    expected_max_count = None
    if expected_id != -1 and expected_id in results:
        expected_max_count, _ = results[expected_id]
        expected_score = compute_candidate_score(expected_max_count, total_matches)

    return MatchSelection(
        song_id=best_song_id,
        time_offset=best_time_offset,
        score=best_score,
        max_count=best_max_count,
        total_matches=total_matches,
        expected_score=expected_score,
        expected_max_count=expected_max_count,
        reason="selected"
    )
