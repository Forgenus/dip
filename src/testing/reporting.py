from src.recognition.search_trace import SearchTrace


def format_search_trace(trace: SearchTrace) -> str:
    lines = [
        f"dropped_stage={trace.dropped_stage}",
        f"reason={trace.reason}",
        f"query_fp_count={trace.query_fp_count}",
        f"db_match_count={trace.db_match_count}",
        f"correct_in_db_lookup={trace.correct_in_db_lookup}",
        f"correct_after_filter={trace.correct_after_filter}",
        f"correct_after_time={trace.correct_after_time}",
        f"correct_time_result={trace.correct_time_result}",
        f"selected_id={trace.selected_id}",
        f"selected_score={trace.selected_score:.4f}",
        f"expected_score={trace.expected_score if trace.expected_score is not None else 'None'}",
        f"selected_max_count={trace.selected_max_count}",
        f"expected_max_count={trace.expected_max_count if trace.expected_max_count is not None else 'None'}",
        f"selected_offset={trace.candidates_after_time.get(trace.selected_id)}",
        f"expected_offset={trace.candidates_after_time.get(trace.expected_id)}",
        f"candidates_after_filter={trace.candidates_after_filter[:10]}",
        f"candidates_after_time_first_10={dict(list(trace.candidates_after_time.items())[:10])}",
    ]
    return "\n".join(lines)
