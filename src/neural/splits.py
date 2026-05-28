"""Song-level split helpers for neural model training."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import config as cfg


@dataclass
class SongSplitItem:
    song_id: int
    file_path: str
    title: str
    artist: str
    duration: float


@dataclass
class SongSplit:
    version: int
    seed: int
    ratios: dict[str, float]
    counts: dict[str, int]
    train: list[SongSplitItem]
    validation_heldout: list[SongSplitItem]
    test_heldout: list[SongSplitItem]


def collect_song_items(service_or_db: Any) -> list[SongSplitItem]:
    """Collect songs from the service/database shape and sort them by song_id."""
    songs = _extract_songs(service_or_db)
    items = [_song_to_item(song) for song in songs]
    return sorted(items, key=lambda item: item.song_id)


def split_song_items(
    items: Iterable[SongSplitItem],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SongSplit:
    """Create a deterministic song-level split from the provided items."""
    _validate_ratios(train_ratio, validation_ratio, test_ratio)

    sorted_items = sorted(list(items), key=lambda item: item.song_id)
    shuffled = list(sorted_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(np.floor(total * train_ratio))
    validation_count = int(np.floor(total * validation_ratio))
    test_count = total - train_count - validation_count

    train_end = train_count
    validation_end = train_end + validation_count

    return SongSplit(
        version=1,
        seed=int(seed),
        ratios={
            "train": float(train_ratio),
            "validation_heldout": float(validation_ratio),
            "test_heldout": float(test_ratio),
        },
        counts={
            "total": total,
            "train": train_count,
            "validation_heldout": validation_count,
            "test_heldout": test_count,
        },
        train=shuffled[:train_end],
        validation_heldout=shuffled[train_end:validation_end],
        test_heldout=shuffled[validation_end:],
    )


def save_song_split(split: SongSplit, path: Path | str) -> None:
    """Save a split as stable, human-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(split), file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_song_split(path: Path | str) -> SongSplit:
    """Load a split JSON file into dataclasses."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return SongSplit(
        version=int(data["version"]),
        seed=int(data["seed"]),
        ratios={
            "train": float(data["ratios"]["train"]),
            "validation_heldout": float(data["ratios"]["validation_heldout"]),
            "test_heldout": float(data["ratios"]["test_heldout"]),
        },
        counts={
            "total": int(data["counts"]["total"]),
            "train": int(data["counts"]["train"]),
            "validation_heldout": int(data["counts"]["validation_heldout"]),
            "test_heldout": int(data["counts"]["test_heldout"]),
        },
        train=[_dict_to_item(song) for song in data["train"]],
        validation_heldout=[_dict_to_item(song) for song in data["validation_heldout"]],
        test_heldout=[_dict_to_item(song) for song in data["test_heldout"]],
    )


def _extract_songs(service_or_db: Any) -> Iterable[Any]:
    if isinstance(service_or_db, dict):
        return service_or_db.values()

    songs_db = getattr(service_or_db, "songs", service_or_db)

    if hasattr(songs_db, "get_all_songs"):
        return songs_db.get_all_songs()

    raw_db = getattr(songs_db, "db", None)
    if isinstance(raw_db, dict):
        return raw_db.values()

    if hasattr(service_or_db, "db"):
        return _extract_songs(service_or_db.db)

    raise TypeError("Expected a service, MusicDatabase, SongInfoDB, or compatible song database")


def _song_to_item(song: Any) -> SongSplitItem:
    return SongSplitItem(
        song_id=int(_get_song_value(song, "song_id")),
        file_path=_portable_path(_get_song_value(song, "file_path")),
        title=str(_get_song_value(song, "title", "")),
        artist=str(_get_song_value(song, "artist", "")),
        duration=float(_get_song_value(song, "duration", 0.0)),
    )


def _dict_to_item(song: dict[str, Any]) -> SongSplitItem:
    return SongSplitItem(
        song_id=int(song["song_id"]),
        file_path=str(song["file_path"]),
        title=str(song.get("title", "")),
        artist=str(song.get("artist", "")),
        duration=float(song.get("duration", 0.0)),
    )


def _get_song_value(song: Any, key: str, default: Any = None) -> Any:
    if isinstance(song, dict):
        return song.get(key, default)
    return getattr(song, key, default)


def _portable_path(file_path: Any) -> str:
    path = Path(file_path)
    try:
        return path.resolve().relative_to(cfg.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")

    total = sum(ratios)
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
