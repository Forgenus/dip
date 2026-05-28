from dataclasses import dataclass


@dataclass
class PairExample:
    query_song_id: int
    candidate_song_id: int
    query_file_path: str
    candidate_file_path: str
    query_start_seconds: float
    candidate_start_seconds: float
    query_valid_seconds: float
    padding_ratio: float
    pair_type: str
    label: int


def make_same_time_positive(
    song: dict,
    start_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    return PairExample(
        query_song_id=int(song["song_id"]),
        candidate_song_id=int(song["song_id"]),
        query_file_path=str(song["file_path"]),
        candidate_file_path=str(song["file_path"]),
        query_start_seconds=start_seconds,
        candidate_start_seconds=start_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="positive_same_time",
        label=1,
    )


def make_jittered_positive(
    song: dict,
    query_start_seconds: float,
    jitter_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    return PairExample(
        query_song_id=int(song["song_id"]),
        candidate_song_id=int(song["song_id"]),
        query_file_path=str(song["file_path"]),
        candidate_file_path=str(song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=max(0.0, query_start_seconds + jitter_seconds),
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="positive_jittered",
        label=1,
    )


def make_random_negative(
    query_song: dict,
    candidate_song: dict,
    query_start_seconds: float,
    candidate_start_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    if int(query_song["song_id"]) == int(candidate_song["song_id"]):
        raise ValueError("random negative requires different songs")
    return PairExample(
        query_song_id=int(query_song["song_id"]),
        candidate_song_id=int(candidate_song["song_id"]),
        query_file_path=str(query_song["file_path"]),
        candidate_file_path=str(candidate_song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=candidate_start_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="negative_random",
        label=0,
    )


def make_hard_negative(
    query_song: dict,
    candidate_song: dict,
    query_start_seconds: float,
    candidate_offset_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    if int(query_song["song_id"]) == int(candidate_song["song_id"]):
        raise ValueError("hard negative requires different songs")
    return PairExample(
        query_song_id=int(query_song["song_id"]),
        candidate_song_id=int(candidate_song["song_id"]),
        query_file_path=str(query_song["file_path"]),
        candidate_file_path=str(candidate_song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=candidate_offset_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="negative_hard",
        label=0,
    )


def padding_ratio(valid_seconds: float, window_seconds: float) -> float:
    if window_seconds <= 0:
        return 0.0
    return max(0.0, window_seconds - valid_seconds) / window_seconds


def duration_bucket(valid_seconds: float) -> str:
    if valid_seconds >= 5.0:
        return "5.0s"
    if valid_seconds >= 4.0:
        return "4-5s"
    if valid_seconds >= 3.0:
        return "3-4s"
    return "2-3s"
