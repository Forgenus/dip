from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import numpy as np

import config as cfg
from src.processing import fft_filter as ff
from src.processing import fingerprint as fp
from src.processing import preprocess as pp


Fingerprint = Tuple[int, int]
Point = Tuple[int, int]


@dataclass
class PipelineSnapshot:
    points: np.ndarray
    fingerprints: List[Fingerprint]


@dataclass
class OffsetSpreadSummary:
    count: int
    top_buckets: Dict[int, int]
    minimum: int | None
    maximum: int | None
    mean: float | None
    std: float | None


@dataclass
class FailureAnalysisReport:
    error: str | None
    snippet_points_count: int
    original_segment_points_count: int
    common_points_count: int
    snippet_only_points_count: int
    original_only_points_count: int
    snippet_fp_count: int
    original_segment_fp_count: int
    common_fingerprint_count: int
    common_address_count: int
    common_address_ratio: float
    expected_offset_top_buckets: Dict[int, int]
    offset_spread: OffsetSpreadSummary
    snippet_time_distribution: Dict[int, int]
    original_time_distribution: Dict[int, int]
    snippet_freq_distribution: Dict[int, int]
    original_freq_distribution: Dict[int, int]
    start_sample_mod_hop: int | None
    full_song_start_frame: int | None
    full_song_end_frame: int | None
    full_song_window_fp_count: int
    full_song_common_address_count: int
    full_song_common_address_ratio: float
    full_song_offset_top_buckets: Dict[int, int]
    full_song_offset_spread: OffsetSpreadSummary


def analyze_failed_snippet_against_file(
    snippet_audio,
    original_file: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = cfg.SAMPLE_RATE,
) -> FailureAnalysisReport:
    original_audio = pp.load_audio(Path(original_file), target_sr=sample_rate)
    start_sample = max(0, int(round(start_seconds * sample_rate)))
    segment_samples = max(0, int(round(duration_seconds * sample_rate)))
    original_segment = original_audio[start_sample:start_sample + segment_samples]

    report = analyze_failed_snippet_audio(
        snippet_audio=snippet_audio,
        original_segment_audio=original_segment,
    )
    add_full_song_indexing_comparison(
        report=report,
        snippet_audio=snippet_audio,
        original_audio=original_audio,
        start_sample=start_sample,
        segment_samples=segment_samples,
    )
    return report


def analyze_failed_snippet_audio(
    snippet_audio,
    original_segment_audio,
) -> FailureAnalysisReport:
    try:
        snippet = build_snapshot(snippet_audio)
        original = build_snapshot(original_segment_audio)
    except ValueError as error:
        return empty_report(str(error))

    snippet_points = points_set(snippet.points)
    original_points = points_set(original.points)
    common_points = snippet_points & original_points

    snippet_addresses = {address for address, _ in snippet.fingerprints}
    original_addresses = {address for address, _ in original.fingerprints}
    common_addresses = snippet_addresses & original_addresses
    address_union_count = len(snippet_addresses | original_addresses)

    snippet_fp_set = set(snippet.fingerprints)
    original_fp_set = set(original.fingerprints)
    common_fingerprints = snippet_fp_set & original_fp_set

    offset_spread = summarize_common_offsets(
        fingerprints_by_address(snippet.fingerprints),
        fingerprints_by_address(original.fingerprints),
        bucket_size=cfg.DELTA_BUCKET_SIZE,
    )

    return FailureAnalysisReport(
        error=None,
        snippet_points_count=len(snippet_points),
        original_segment_points_count=len(original_points),
        common_points_count=len(common_points),
        snippet_only_points_count=len(snippet_points - original_points),
        original_only_points_count=len(original_points - snippet_points),
        snippet_fp_count=len(snippet.fingerprints),
        original_segment_fp_count=len(original.fingerprints),
        common_fingerprint_count=len(common_fingerprints),
        common_address_count=len(common_addresses),
        common_address_ratio=(
            len(common_addresses) / address_union_count
            if address_union_count
            else 0.0
        ),
        expected_offset_top_buckets=offset_spread.top_buckets,
        offset_spread=offset_spread,
        snippet_time_distribution=point_distribution(snippet.points, axis=0, bucket_size=10),
        original_time_distribution=point_distribution(original.points, axis=0, bucket_size=10),
        snippet_freq_distribution=point_distribution(snippet.points, axis=1, bucket_size=32),
        original_freq_distribution=point_distribution(original.points, axis=1, bucket_size=32),
        start_sample_mod_hop=None,
        full_song_start_frame=None,
        full_song_end_frame=None,
        full_song_window_fp_count=0,
        full_song_common_address_count=0,
        full_song_common_address_ratio=0.0,
        full_song_offset_top_buckets={},
        full_song_offset_spread=empty_offset_spread(),
    )


def empty_report(error: str) -> FailureAnalysisReport:
    empty_spread = empty_offset_spread()
    return FailureAnalysisReport(
        error=error,
        snippet_points_count=0,
        original_segment_points_count=0,
        common_points_count=0,
        snippet_only_points_count=0,
        original_only_points_count=0,
        snippet_fp_count=0,
        original_segment_fp_count=0,
        common_fingerprint_count=0,
        common_address_count=0,
        common_address_ratio=0.0,
        expected_offset_top_buckets={},
        offset_spread=empty_spread,
        snippet_time_distribution={},
        original_time_distribution={},
        snippet_freq_distribution={},
        original_freq_distribution={},
        start_sample_mod_hop=None,
        full_song_start_frame=None,
        full_song_end_frame=None,
        full_song_window_fp_count=0,
        full_song_common_address_count=0,
        full_song_common_address_ratio=0.0,
        full_song_offset_top_buckets={},
        full_song_offset_spread=empty_spread,
    )


def empty_offset_spread() -> OffsetSpreadSummary:
    return OffsetSpreadSummary(
        count=0,
        top_buckets={},
        minimum=None,
        maximum=None,
        mean=None,
        std=None,
    )


def add_full_song_indexing_comparison(
    report: FailureAnalysisReport,
    snippet_audio,
    original_audio,
    start_sample: int,
    segment_samples: int,
) -> None:
    if report.error is not None:
        return

    try:
        snippet = build_snapshot(snippet_audio)
        full_song = build_snapshot(original_audio)
    except ValueError as error:
        report.error = f"full_song_indexing_comparison: {error}"
        return

    start_frame = start_sample // cfg.HOP_LENGTH
    end_frame = (start_sample + segment_samples + cfg.HOP_LENGTH - 1) // cfg.HOP_LENGTH
    indexed_window = select_anchor_time_window(
        full_song.fingerprints,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    snippet_addresses = {address for address, _ in snippet.fingerprints}
    indexed_addresses = {address for address, _ in indexed_window}
    common_addresses = snippet_addresses & indexed_addresses
    address_union_count = len(snippet_addresses | indexed_addresses)

    offset_spread = summarize_common_offsets(
        fingerprints_by_address(snippet.fingerprints),
        fingerprints_by_address(indexed_window),
        bucket_size=cfg.DELTA_BUCKET_SIZE,
    )

    report.start_sample_mod_hop = start_sample % cfg.HOP_LENGTH
    report.full_song_start_frame = start_frame
    report.full_song_end_frame = end_frame
    report.full_song_window_fp_count = len(indexed_window)
    report.full_song_common_address_count = len(common_addresses)
    report.full_song_common_address_ratio = (
        len(common_addresses) / address_union_count
        if address_union_count
        else 0.0
    )
    report.full_song_offset_top_buckets = offset_spread.top_buckets
    report.full_song_offset_spread = offset_spread


def build_snapshot(audio) -> PipelineSnapshot:
    spectrogram = ff.stft(audio)
    points = ff.filter_spectrogram(spectrogram.T)
    fingerprints = fp.create_fingerprints(points, song_id=0)
    return PipelineSnapshot(points=points, fingerprints=fingerprints)


def points_set(points: np.ndarray) -> set[Point]:
    return {
        (int(time), int(freq))
        for time, freq in points
    }


def fingerprints_by_address(
    fingerprints: Iterable[Fingerprint],
) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)

    for address, hash_value in fingerprints:
        grouped[int(address)].append(int(hash_value))

    return dict(grouped)


def select_anchor_time_window(
    fingerprints: Iterable[Fingerprint],
    start_frame: int,
    end_frame: int,
) -> List[Fingerprint]:
    selected: List[Fingerprint] = []

    for address, hash_value in fingerprints:
        anchor_time, _ = fp.decode_hash(hash_value)

        if start_frame <= anchor_time <= end_frame:
            selected.append((address, hash_value))

    return selected


def summarize_common_offsets(
    snippet_by_address: Dict[int, List[int]],
    original_by_address: Dict[int, List[int]],
    bucket_size: int,
    top_n: int = 10,
) -> OffsetSpreadSummary:
    offsets: list[int] = []
    bucket_counts: Counter[int] = Counter()

    for address in snippet_by_address.keys() & original_by_address.keys():
        for snippet_hash in snippet_by_address[address]:
            snippet_time, _ = fp.decode_hash(snippet_hash)

            for original_hash in original_by_address[address]:
                original_time, _ = fp.decode_hash(original_hash)
                offset = snippet_time - original_time
                bucket = round(offset / bucket_size) * bucket_size
                offsets.append(offset)
                bucket_counts[bucket] += 1

    top_buckets = dict(bucket_counts.most_common(top_n))

    if not offsets:
        return OffsetSpreadSummary(
            count=0,
            top_buckets={},
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
        )

    return OffsetSpreadSummary(
        count=len(offsets),
        top_buckets=top_buckets,
        minimum=min(offsets),
        maximum=max(offsets),
        mean=mean(offsets),
        std=pstdev(offsets) if len(offsets) > 1 else 0.0,
    )


def point_distribution(
    points: np.ndarray,
    axis: int,
    bucket_size: int,
    top_n: int = 8,
) -> Dict[int, int]:
    counts: Counter[int] = Counter()

    for point in points:
        bucket = (int(point[axis]) // bucket_size) * bucket_size
        counts[bucket] += 1

    return dict(counts.most_common(top_n))
