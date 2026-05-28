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
FAILED_SNIPPETS_DIR = get_path('FAILED_SNIPPETS_DIR', 'data/debug/failed_snippets')
TEST_SNIPPETS_DIR = get_path('TEST_SNIPPETS_DIR', 'data/debug/test_snippets')
EFFECT_SNIPPETS_DIR = get_path('EFFECT_SNIPPETS_DIR', 'data/debug/effect_snippets')
FINGERPRINT_DB_NAME = os.getenv("FINGERPRINT_DB_NAME", "fingerprints")
SONG_INFO_DB_NAME = os.getenv("SONG_INFO_DB_NAME", "songs")

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '11025'))
N_FFT = int(os.getenv('N_FFT', '1024'))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', '768'))
# Время за один таймфрейм (в секундах) - вычисляется динамически
TIME_PER_FRAME = HOP_LENGTH / SAMPLE_RATE
BIN_TIME = TIME_PER_FRAME  # Для обратной совместимости
SEARCH_OFFSET_FALLBACK_MIN_SCORE = float(os.getenv('SEARCH_OFFSET_FALLBACK_MIN_SCORE', '0.03'))
SEARCH_OFFSET_FALLBACK_SAMPLES = [
    HOP_LENGTH // 4,
    HOP_LENGTH // 2,
    (3 * HOP_LENGTH) // 4,
]

DELTA_BUCKET_SIZE = int(os.getenv('BUCKET_SIZE', '7'))

RAW_AUDIO_DIR = get_path('RAW_AUDIO_DIR', 'data/raw')
PROCESSED_AUDIO_DIR = get_path('PROCESSED_AUDIO_DIR', 'data/processed')
LOW_PASS_CUTOFF_HZ = int(os.getenv('LOW_PASS_CUTOFF_HZ', '5000'))
MIN_TARGET_DELTA = int(os.getenv('MIN_TARGET_DELTA', '2'))
MAX_TARGET_DELTA = int(os.getenv('MAX_TARGET_DELTA', '30'))
TARGETS_PER_ANCHOR = int(os.getenv('TARGETS_PER_ANCHOR', '5'))
RNG_SEED = int(os.getenv('RNG_SEED','10'))
WINDOW = os.getenv("WINDOW",'hann')
