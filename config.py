import os
from dotenv import load_dotenv
from pathlib import Path

# Загружаем .env
load_dotenv()
BASE_DIR = Path(__file__).parent.absolute()


def get_path(env_var: str, default: str) -> Path:
    """
    Получает путь из переменной окружения или использует значение по умолчанию.
    Возвращает абсолютный путь относительно корня проекта.
    """
    path_str = os.getenv(env_var, default)
    if not os.path.isabs(path_str):
        return BASE_DIR / path_str
    return Path(path_str)

METADATA_JSON_PATH = get_path('METADATA_JSON_PATH','data/processed/metadata.json')
UNPROCESSED_DIR = get_path('UNPROCESSED_DIR','data/raw')
SONGS_DIR = get_path('SONGS_DIR', 'data/processed')
VALIDATE = os.getenv('VALIDATE', 'True').lower() in ('true', '1', 'yes')
DATABASE_DIR = get_path('DATABASE_DIR', 'data/databases')
FINGERPRINT_DB_NAME = get_path('FINGERPRINT_DB_NAME', 'fingerprints')
SONG_INFO_DB_NAME = get_path('SONG_INFO_DB_NAME', 'songs')

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '11025'))
N_FFT = int(os.getenv('N_FFT', '1024'))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', '512'))
BIN_TIME = HOP_LENGTH/SAMPLE_RATE

RAW_AUDIO_DIR = get_path('RAW_AUDIO_DIR', 'data/raw')
PROCESSED_AUDIO_DIR = get_path('PROCESSED_AUDIO_DIR', 'data/processed')

WIDTH = int(os.getenv('WIDTH', '5'))
RNG_SEED = int(os.getenv('RNG_SEED','10'))
WINDOW = os.getenv("WINDOW",'hann')
