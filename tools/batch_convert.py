from importlib.metadata import files
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import sys
from mutagen import File as MutagenFile  # для чтения тегов
import config as cfg
log = print
# Настройки конвертации
TARGET_SAMPLE_RATE = 11025
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = 's16'
TARGET_FORMAT = 'wav'
NUM_THREADS = 16
LOG_FILE = 'conversion.log'
METADATA_FILE = 'metadata.json'


logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

metadata_dict = {}

def read_metadata(src_path: Path) -> dict:
    tags = {
        "title": src_path.stem,
        "artist": "Unknown Artist",
        "album": "Unknown Album",
        "year": None,
        "duration": None
    }

    try:
        audio = MutagenFile(src_path, easy=True)
        if audio:
            if 'title' in audio:
                tags['title'] = audio['title'][0]
            if 'artist' in audio:
                tags['artist'] = audio['artist'][0]
            if 'album' in audio:
                tags['album'] = audio['album'][0]
            if 'date' in audio:
                tags['year'] = audio['date'][0]
            tags['duration'] = f"{audio.info.length:.1f}"
    except Exception:
        pass

    return tags

def convert_file(src_path: Path, src_root: Path, dst_root: Path):
    relative_path = src_path.relative_to(src_root).with_suffix(f'.{TARGET_FORMAT}')
    dst_path = dst_root / relative_path
    if dst_path.exists():
        msg = f"[SKIP] {dst_path} уже существует"
        log(msg)
        logging.info(msg)
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg команда
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(src_path),
        '-vn',
        '-ac', str(TARGET_CHANNELS),
        '-ar', str(TARGET_SAMPLE_RATE),
        '-sample_fmt', TARGET_BIT_DEPTH,
        str(dst_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        msg = f"[OK] {src_path} -> {dst_path}"
        log(msg)
        logging.info(msg)

        # сохраняем метаданные
        tags = read_metadata(src_path)
        metadata_dict[str(dst_path.relative_to(dst_root))] = tags

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode(errors='ignore')
        msg = f"[ERROR] {src_path}: {error_msg}"
        log(msg)
        logging.error(msg)

def process_directory(src_root: str, dst_root: str):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    audio_exts = ['.mp3', '.flac', '.m4a', '.aac', '.wav']
    files_to_convert = [p for p in src_root.rglob('*') if p.suffix.lower() in audio_exts]

    log(f"Найдено {len(files_to_convert)} файлов для конвертации.")
    logging.info(f"Convert start: found {len(files_to_convert)} files.")

    # Если файл метаданных уже существует, читаем его
    metadata_path = dst_root / METADATA_FILE
    existing_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"{METADATA_FILE} is faulty, creating new")

    # Запускаем многопоточную конвертацию
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(convert_file, f, src_root, dst_root) for f in files_to_convert]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                msg = f"[THREAD ERROR] {e}"
                log(msg)
                logging.exception(msg)

    # Объединяем существующие и новые метаданные
    merged_metadata = existing_metadata.copy()
    merged_metadata.update(metadata_dict)

    # Сохраняем в JSON корректно
    dst_root.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(merged_metadata, f, indent=2, ensure_ascii=False)

    logging.info("Conversion and metadata saving completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch audio converter to WAV mono 16-bit 11025 Hz with metadata saved in JSON")
    parser.add_argument('--src', type=str,default=str(cfg.UNPROCESSED_DIR), help='Source directory with music')
    parser.add_argument('--dst', type=str,default=str(cfg.SONGS_DIR),help='Destination directory for converted files')
    args = parser.parse_args()

    process_directory(args.src, args.dst)