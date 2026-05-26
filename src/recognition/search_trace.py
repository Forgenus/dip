from dataclasses import dataclass
from typing import Dict

@dataclass
class SearchTrace:
    expected_id: int = -1
    query_fp_count: int = 0

    db_match_count: int = 0
    correct_in_db_lookup: bool = False

    candidates_after_filter: list[int] = None
    correct_after_filter: bool = False

    candidates_after_time: Dict[int, tuple[int, int]] = None
    correct_after_time: bool = False
    correct_time_result: tuple[int, int] | None = None

    selected_id: int = -1
    selected_score: float = 0.0
    expected_score: float | None = None
    expected_time_offset: int | None = None
    selected_max_count: int = 0
    expected_max_count: int | None = None

    dropped_stage: str = "unknown"
    reason: str = ""

    def __post_init__(self):
        if self.candidates_after_filter is None:
            self.candidates_after_filter = []
        if self.candidates_after_time is None:
            self.candidates_after_time = {}
