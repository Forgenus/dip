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
def load_audio(file_path:Path, target_sr:int=11025):
    """
    Загружает аудио файл и преобразует его в моно с заданной частотой дискретизации
    """
    audio = librosa.load(file_path, sr=target_sr, mono=True)[0]
    return audio


def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Извлекает метаданные для файла из JSON-базы.
    Если информации нет — возвращает пустые значения.

    Args:
        file_path: путь к аудиофайлу

    Returns:
        Dict с метаданными (title, artist, album, year, genre, duration)
    """
    metadata: Dict[str, Any] = {
        'title': '',
        'artist': '',
        'album': '',
        'year': '',
        'genre': '',
        'duration': 0.0
    }

    json_path = Path(cfg.METADATA_JSON_PATH)  
    if not json_path.exists():
        return metadata

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)  # ожидаем, что JSON — словарь {filename_or_path: {tags...}}

        # Используем file_path.stem или полный путь в качестве ключа
        key = str(file_path.resolve().relative_to(cfg.SONGS_DIR))  # абсолютный путь
        if key not in all_metadata:
            key = file_path.name  # fallback на имя файла

        if key in all_metadata:
            file_tags = all_metadata[key]
            for k in metadata.keys():
                if k in file_tags:
                    metadata[k] = file_tags[k]

    except Exception as e:
        print(f"Exception while extracting metadata for {file_path}: {e}")

    return metadata


def load_audio_with_metadata(file_path: Path, target_sr:int=11025) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Загружает аудио и извлекает метаданные одной функцией
    
    Returns:
        Tuple[audio, metadata]
    """
    audio = load_audio(file_path, target_sr)
    metadata = extract_metadata(file_path)
    # Если title не найден, используем имя файла
    if not metadata['title']:
        import os
        metadata['title'] = file_path.stem
    
    return audio, metadata


