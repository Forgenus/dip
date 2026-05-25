from random import random
import subprocess
import librosa
import numpy as np
import sys
import soundfile as sf
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from typing import Dict, Tuple, Any
from pathlib import Path
import json
import random
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import config as cfg

log = print

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
    audio = librosa.load(file_path, sr=target_sr, mono=True)[0]
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


