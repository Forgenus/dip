import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from mutagen import File as MutagenFile

import config as cfg


TARGET_SAMPLE_RATE = cfg.SAMPLE_RATE
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = "s16"
TARGET_FORMAT = "wav"

DEFAULT_NUM_THREADS = 8
AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg"}

LOG_FILE = "conversion.log"
METADATA_FILE = cfg.METADATA_JSON_PATH

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    source_path: Path
    destination_path: Path
    relative_path: str
    metadata: dict
    converted: bool
    error: str | None = None


def setup_logging() -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def normalize_relative_path(path: Path) -> str:
    """
    Возвращает путь в unix-like формате для JSON:
    Artist/Album/song.wav вместо Artist\\Album\\song.wav.
    """
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

    except Exception as error:
        logger.warning("Failed to read metadata from %s: %s", src_path, error)

    return tags


def build_destination_path(src_path: Path, src_root: Path, dst_root: Path) -> Path:
    relative_path = src_path.relative_to(src_root).with_suffix(f".{TARGET_FORMAT}")
    return dst_root / relative_path


def convert_file(src_path: Path, src_root: Path, dst_root: Path) -> ConversionResult:
    dst_path = build_destination_path(src_path, src_root, dst_root)

    # Ключ metadata.json — путь относительно папки с песнями.
    # Например: Artist/Album/song.wav
    relative_path = normalize_relative_path(dst_path.relative_to(dst_root))

    metadata = read_metadata(src_path)

    if dst_path.exists():
        message = f"[SKIP] {dst_path} already exists"
        print(message)
        logger.info(message)

        return ConversionResult(
            source_path=src_path,
            destination_path=dst_path,
            relative_path=relative_path,
            metadata=metadata,
            converted=False,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        TARGET_BIT_DEPTH,
        str(dst_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        message = f"[OK] {src_path} -> {dst_path}"
        print(message)
        logger.info(message)

        return ConversionResult(
            source_path=src_path,
            destination_path=dst_path,
            relative_path=relative_path,
            metadata=metadata,
            converted=True,
        )

    except subprocess.CalledProcessError as error:
        error_message = error.stderr.decode(errors="ignore")
        message = f"[ERROR] {src_path}: {error_message}"
        print(message)
        logger.error(message)

        return ConversionResult(
            source_path=src_path,
            destination_path=dst_path,
            relative_path=relative_path,
            metadata=metadata,
            converted=False,
            error=error_message,
        )


def find_audio_files(src_root: Path) -> list[Path]:
    return [
        path
        for path in src_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def load_existing_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.warning("%s is invalid, creating new metadata file", metadata_path)
        return {}


def save_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def process_directory(
    src_root: Path,
    dst_root: Path,
    num_threads: int = DEFAULT_NUM_THREADS,
) -> None:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    files_to_convert = find_audio_files(src_root)

    print(f"Found {len(files_to_convert)} files to process.")
    logger.info("Conversion started: found %d files", len(files_to_convert))

    metadata_path = cfg.METADATA_JSON_PATH
    metadata = load_existing_metadata(metadata_path)

    converted_count = 0
    skipped_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(convert_file, file_path, src_root, dst_root)
            for file_path in files_to_convert
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                error_count += 1
                logger.exception("[THREAD ERROR] %s", error)
                print(f"[THREAD ERROR] {error}")
                continue

            if result.error:
                error_count += 1
                continue

            metadata[result.relative_path] = result.metadata

            if result.converted:
                converted_count += 1
            else:
                skipped_count += 1

    save_metadata(metadata_path, metadata)

    print()
    print("Done.")
    print(f"Converted: {converted_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Metadata saved to: {metadata_path}")

    logger.info(
        "Conversion completed. Converted=%d, skipped=%d, errors=%d",
        converted_count,
        skipped_count,
        error_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch audio converter to WAV mono 16-bit with metadata JSON"
    )

    parser.add_argument(
        "--src",
        type=Path,
        default=cfg.UNPROCESSED_DIR,
        help="Source directory with original music files",
    )

    parser.add_argument(
        "--dst",
        type=Path,
        default=cfg.SONGS_DIR,
        help="Destination directory for converted files",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of parallel conversion threads",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    process_directory(
        src_root=args.src,
        dst_root=args.dst,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()