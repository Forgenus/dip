import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_command import build_ffmpeg_command
from .metadata import normalize_relative_path, read_metadata, build_destination_path, load_existing_metadata, save_metadata
import config as cfg

TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"
DEFAULT_NUM_THREADS = 8
AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg"}
LOG_FILE = "conversion.log"

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


def convert_file(src_path: Path, src_root: Path, dst_root: Path) -> ConversionResult:
    dst_path = build_destination_path(src_path, src_root, dst_root)
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
    cmd = build_ffmpeg_command(src_path, dst_path)

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


def process_directory(src_root: Path, dst_root: Path, num_threads: int = DEFAULT_NUM_THREADS) -> None:
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
