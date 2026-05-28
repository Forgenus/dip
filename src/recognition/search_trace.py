from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CandidateTrace:
    song_id: int
    rank: int
    score: float
    max_count: int
    time_offset_bins: int
    time_offset_seconds: float


@dataclass
class NeuralCandidateTrace:
    song_id: int
    rank: int
    fingerprint_score: float
    fingerprint_max_count: int
    fingerprint_time_offset_seconds: float
    same_probability: float
    decision: str
    threshold: float
    reliability: str
    query_valid_seconds: float
    candidate_valid_seconds: float
    padding_ratio: float


@dataclass
class SearchTrace:
    expected_id: int = -1
    query_fp_count: int = 0

    db_match_count: int = 0
    correct_in_db_lookup: bool = False
    raw_matches_by_song: Dict[int, int] = None
    expected_raw_match_count: int = 0

    candidates_after_filter: list[int] = None
    correct_after_filter: bool = False
    unique_addresses_by_song: Dict[int, int] = None
    expected_unique_address_count: int = 0

    candidates_after_time: Dict[int, tuple[int, int]] = None
    correct_after_time: bool = False
    correct_time_result: tuple[int, int] | None = None
    expected_offset_buckets: Dict[int, int] = None

    selected_id: int = -1
    selected_score: float = 0.0
    expected_score: float | None = None
    expected_time_offset: int | None = None
    selected_max_count: int = 0
    expected_max_count: int | None = None
    failure_analysis: Any = None
    attempt_sample_offset: int = 0
    offset_fallback_selected: bool = False
    offset_fallback_attempts: list[dict[str, Any]] = None

    dropped_stage: str = "unknown"
    reason: str = ""
    top_candidates: list[CandidateTrace] = None
    neural_enabled: bool = False
    neural_checked: bool = False
    neural_reason: str = ""
    neural_results: list[NeuralCandidateTrace] = None
    neural_error: str | None = None

    def __post_init__(self):
        if self.top_candidates is None:
            self.top_candidates = []
        if self.candidates_after_filter is None:
            self.candidates_after_filter = []
        if self.candidates_after_time is None:
            self.candidates_after_time = {}
        if self.raw_matches_by_song is None:
            self.raw_matches_by_song = {}
        if self.unique_addresses_by_song is None:
            self.unique_addresses_by_song = {}
        if self.expected_offset_buckets is None:
            self.expected_offset_buckets = {}
        if self.offset_fallback_attempts is None:
            self.offset_fallback_attempts = []
        if self.neural_results is None:
            self.neural_results = []

    @property
    def total_matches(self) -> int:
        return self.db_match_count
