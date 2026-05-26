import json
from pathlib import Path

import config as cfg


def normalize_relative_path(path: Path) -> str:
    return path.as_posix()


def read_metadata(src_path: Path) -> dict:
    tags = {
        "title": src_path.stem,
        "artist": "Unknown Artist",
        "album": "Unknown Album",
        "year": None,
        "duration": None,
    }

    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(src_path, easy=True)

        if audio is None:
            return tags

        if "title" in audio:
            tags["title"] = audio["title"][0]

        if "artist" in audio:
            tags["artist"] = audio["artist"][0]

        if "album" in audio:
            tags["album"] = audio["album"][0]

        if "date" in audio:
            tags["year"] = audio["date"][0]

        if audio.info and hasattr(audio.info, "length"):
            tags["duration"] = round(float(audio.info.length), 1)

    except Exception:
        pass

    return tags


def build_destination_path(src_path: Path, src_root: Path, dst_root: Path) -> Path:
    relative_path = src_path.relative_to(src_root).with_suffix(".wav")
    return dst_root / relative_path


def load_existing_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def save_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
