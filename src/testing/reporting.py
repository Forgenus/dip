from dataclasses import dataclass
from pathlib import Path

from src.recognition.search_trace import SearchTrace
from src.testing.failure_analysis import FailureAnalysisReport


@dataclass
class FailedSnippetRecord:
    output_file: Path
    expected_id: int
    expected_title: str
    found_id: int
    found_title: str
    source_file: Path
    start_seconds: float
    duration_seconds: float


def format_search_trace(trace: SearchTrace) -> str:
    lines = [
        "=" * 72,
        "SEARCH TRACE",
        "-" * 72,
        f"dropped_stage={trace.dropped_stage}",
        f"reason={trace.reason}",
        "-" * 72,
        "lookup/filter",
        f"query_fp_count={trace.query_fp_count}",
        f"total_matches={trace.total_matches}",
        f"expected_raw_matches={trace.expected_raw_match_count}",
        f"expected_unique_addresses={trace.expected_unique_address_count}",
        "-" * 72,
        "selection/time coherency",
        f"selected_id={trace.selected_id}",
        f"selected_score={trace.selected_score:.4f}",
        f"expected_score={trace.expected_score if trace.expected_score is not None else 'None'}",
        f"selected_offset={trace.candidates_after_time.get(trace.selected_id)}",
        f"expected_offset={trace.candidates_after_time.get(trace.expected_id)}",
        f"expected_offset_buckets_top_10={dict(list(trace.expected_offset_buckets.items())[:10])}",
        f"candidates_after_filter={trace.candidates_after_filter[:10]}",
        f"candidates_after_time_first_10={dict(list(trace.candidates_after_time.items())[:10])}",
    ]
    if trace.failure_analysis is not None:
        lines.extend(format_failure_analysis_report(trace.failure_analysis).splitlines())
    lines.append("=" * 72)
    return "\n".join(lines)


def format_failure_analysis_report(report: FailureAnalysisReport) -> str:
    return "\n".join(
        [
            "-" * 72,
            "FAILED SNIPPET VS ORIGINAL SEGMENT",
            "-" * 72,
            f"analysis_error={report.error if report.error else 'None'}",
            "-" * 72,
            "peaks",
            f"snippet_points_count={report.snippet_points_count}",
            f"original_segment_points_count={report.original_segment_points_count}",
            f"common_points_count={report.common_points_count}",
            f"snippet_only_points_count={report.snippet_only_points_count}",
            f"original_only_points_count={report.original_only_points_count}",
            "-" * 72,
            "fingerprints",
            f"snippet_fp_count={report.snippet_fp_count}",
            f"original_segment_fp_count={report.original_segment_fp_count}",
            f"common_fingerprint_count={report.common_fingerprint_count}",
            f"common_address_count={report.common_address_count}",
            f"common_address_ratio={report.common_address_ratio:.4f}",
            "-" * 72,
            "coherent offsets",
            f"expected_offset_top_buckets={report.expected_offset_top_buckets}",
            "-" * 72,
            "SNIPPET VS FULL SONG INDEXING WINDOW",
            f"start_sample_mod_hop={report.start_sample_mod_hop}",
            f"full_song_start_frame={report.full_song_start_frame}",
            f"full_song_end_frame={report.full_song_end_frame}",
            f"full_song_window_fp_count={report.full_song_window_fp_count}",
            f"full_song_common_address_count={report.full_song_common_address_count}",
            f"full_song_common_address_ratio={report.full_song_common_address_ratio:.4f}",
            f"full_song_offset_top_buckets={report.full_song_offset_top_buckets}",
        ]
    )

