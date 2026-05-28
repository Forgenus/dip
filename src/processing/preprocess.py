import numpy as np
import soundfile as sf
from scipy import signal
from typing import Dict, Tuple
from pathlib import Path
import logging
import math
import time

import config as cfg

logger = logging.getLogger(__name__)

def metadata_key_for_song(file_path: Path) -> str:
    """
    Возвращает путь файла относительно cfg.SONGS_DIR
    в формате, пригодном для metadata.json.

    Пример:
    D:/dip/data/processed/Queen/song.wav
    ->
    Queen/song.wav
    """
    return file_path.resolve().relative_to(cfg.SONGS_DIR.resolve()).as_posix()

def load_audio(file_path:Path, target_sr:int=cfg.SAMPLE_RATE):
    """
    Загружает аудио файл и преобразует его в моно с заданной частотой дискретизации
    """
    audio, source_sr = sf.read(str(file_path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    if target_sr and source_sr != target_sr:
        gcd = math.gcd(source_sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, source_sr // gcd).astype(np.float32, copy=False)
    else:
        audio = np.asarray(audio, dtype=np.float32)

    return audio

DEFAULT_METADATA = {
    "title": None,
    "artist": "Unknown Artist",
    "album": "Unknown Album",
    "year": None,
    "duration": None,
}

def extract_metadata(file_path: Path, metadata: dict) -> dict:
    default_metadata = {
        **DEFAULT_METADATA,
        "title": file_path.stem,
    }

    try:
        key = file_path.resolve().relative_to(cfg.SONGS_DIR.resolve()).as_posix()
    except ValueError:
        return default_metadata

    return metadata.get(key, default_metadata)


def load_audio_with_metadata(file_path: Path,  metadata: dict, target_sr:int=cfg.SAMPLE_RATE) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Загружает аудио и извлекает метаданные одной функцией
    
    Returns:
        Tuple[audio, metadata]
    """
    audio = load_audio(file_path, target_sr)
    metadata = extract_metadata(file_path, metadata)
    # Если title не найден, используем имя файла
    if not metadata['title']:
        import os
        metadata['title'] = file_path.stem
    
    return audio, metadata


