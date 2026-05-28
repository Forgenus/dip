"""Query-side recognition pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

import config as cfg
from ..processing import fingerprint as fp
from .indexing import build_fingerprints_from_audio
from . import match_filter as mf
from .scoring import compute_candidate_score, select_best_match
from .search_trace import CandidateTrace, SearchTrace


invalidval = 2**32 - 2
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    song_id: int
    time_offset: float
    trace: SearchTrace


class QueryPipeline:
    def __init__(self, db) -> None:
        self.db = db

    def search(
        self,
        audio,
        expected_id: int = -1,
        file_path: Path | None = None,
    ) -> SearchResult:
        trace = SearchTrace(expected_id=expected_id)

        try:
            fingerprints_record = build_query_fingerprints(audio)
        except ValueError as error:
            mark_trace(trace, "stft", str(error))
            logger.info("STFT error: %s", error)
            logger.info("File path: %s", file_path)
            return SearchResult(-1, -1.0, trace)

        query_fp_count = len(fingerprints_record)
        trace.query_fp_count = query_fp_count

        if query_fp_count == 0:
            mark_trace(trace, "fingerprint_creation", "query produced 0 fingerprints")
            return SearchResult(-1, -1.0, trace)

        found_fp_list = lookup_matches(self.db, fingerprints_record)
        total_matches = len(found_fp_list)
        trace.db_match_count = total_matches
        trace.raw_matches_by_song = mf.count_raw_matches_by_song(found_fp_list)
        trace.correct_in_db_lookup = song_id_in_found_matches(found_fp_list, expected_id)
        if expected_id != -1:
            trace.expected_raw_match_count = trace.raw_matches_by_song.get(expected_id, 0)

        if not found_fp_list:
            mark_trace(trace, "db_lookup", "no matching addresses found in DB")
            return SearchResult(-1, -1.0, trace)

        if expected_id != -1 and not trace.correct_in_db_lookup:
            mark_trace(
                trace,
                "db_lookup",
                "correct song was not returned by fingerprint DB lookup",
            )
            return SearchResult(-1, -1.0, trace)

        id_fps = filter_matches(found_fp_list, query_fp_count)
        trace.candidates_after_filter = list(id_fps.keys())
        trace.correct_after_filter = expected_id in id_fps
        trace.unique_addresses_by_song = mf.count_unique_addresses_by_song(id_fps)
        if expected_id != -1:
            trace.expected_unique_address_count = trace.unique_addresses_by_song.get(expected_id, 0)

        if not id_fps:
            mark_trace(trace, "filter", "all candidates were removed by match_filter.filter")
            return SearchResult(-1, -1.0, trace)

        if expected_id != -1 and not trace.correct_after_filter:
            mark_trace(trace, "filter", "correct song was removed by match_filter.filter")
            return SearchResult(-1, -1.0, trace)

        results = mf.analyze_time_coherency(fingerprints_record, id_fps)
        if expected_id != -1 and expected_id in id_fps:
            trace.expected_offset_buckets = mf.time_coherency_buckets_for_song(
                fingerprints_record,
                id_fps,
                expected_id,
            )
        update_expected_trace(trace, results, expected_id, total_matches)
        trace.candidates_after_time = results.copy()
        trace.correct_after_time = expected_id in results
        trace.correct_time_result = results.get(expected_id)
        trace.top_candidates = build_top_candidates(results, total_matches)

        if not results:
            mark_trace(trace, "time_coherency", "no candidates after analyze_time_coherency")
            return SearchResult(-1, -1.0, trace)

        if expected_id != -1 and not trace.correct_after_time:
            mark_trace(
                trace,
                "time_coherency",
                "correct song was removed by analyze_time_coherency",
            )
            return SearchResult(-1, -1.0, trace)

        selection = select_best_match(results=results, total_matches=total_matches)
        trace.selected_id = selection.song_id
        trace.selected_score = selection.score
        trace.selected_max_count = selection.max_count
        trace.reason = selection.reason

        if selection.song_id == -1:
            mark_trace(
                trace,
                "selection",
                selection.reason or "no candidate passed final selection thresholds",
            )
            return SearchResult(-1, -1.0, trace)

        time_offset = -cfg.BIN_TIME * selection.time_offset
        if expected_id != -1 and selection.song_id != expected_id:
            mark_trace(
                trace,
                "selection",
                "correct song survived previous stages, but another candidate was selected",
            )
            return SearchResult(selection.song_id, time_offset, trace)

        mark_trace(trace, "matched", "song matched successfully")
        return SearchResult(selection.song_id, time_offset, trace)


def build_query_fingerprints(audio) -> list[tuple[int, int]]:
    return build_fingerprints_from_audio(audio, song_id=invalidval)


def lookup_matches(db, fingerprints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    addresses: List[int] = [address for address, _ in fingerprints]
    return db.fingerprints.lookup_flat_batch(addresses)


def filter_matches(
    found_fp_list: list[tuple[int, int]],
    query_fp_count: int,
) -> dict:
    return mf.filter(found_fp_list)


def build_top_candidates(
    results: dict[int, tuple[int, int]],
    total_matches: int,
) -> list[CandidateTrace]:
    if not results or total_matches <= 0:
        return []

    ranked: list[tuple[float, int, int, int]] = []
    for song_id, (max_count, time_offset_bins) in results.items():
        score = compute_candidate_score(max_count, total_matches)
        ranked.append((score, max_count, song_id, time_offset_bins))

    ranked.sort(reverse=True)

    candidates: list[CandidateTrace] = []
    for rank, (score, max_count, song_id, time_offset_bins) in enumerate(
        ranked[:cfg.NEURAL_SHADOW_TOP_N],
        start=1,
    ):
        candidates.append(
            CandidateTrace(
                song_id=song_id,
                rank=rank,
                score=score,
                max_count=max_count,
                time_offset_bins=time_offset_bins,
                time_offset_seconds=-cfg.BIN_TIME * time_offset_bins,
            )
        )
    return candidates


def mark_trace(trace: SearchTrace, stage: str, reason: str) -> None:
    trace.dropped_stage = stage
    trace.reason = reason


def update_expected_trace(
    trace: SearchTrace,
    results: dict[int, tuple[int, int]],
    expected_id: int,
    total_matches: int,
) -> None:
    if expected_id in results:
        expected_max_count, expected_offset = results[expected_id]
        trace.expected_max_count = expected_max_count
        trace.expected_time_offset = expected_offset
        trace.expected_score = compute_candidate_score(expected_max_count, total_matches)


def song_id_in_found_matches(
    found_fp_list: list[tuple[int, int]],
    expected_id: int,
) -> bool:
    if expected_id == -1:
        return False

    for _, hash_value in found_fp_list:
        _, song_id = fp.decode_hash(hash_value)
        if song_id == expected_id:
            return True
    return False
