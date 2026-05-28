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
NEURAL_SHADOW_ENABLED = os.getenv("NEURAL_SHADOW_ENABLED", "False").lower() in ("true", "1", "yes")
NEURAL_SHADOW_TOP_N = int(os.getenv("NEURAL_SHADOW_TOP_N", "3"))
NEURAL_WINDOW_SECONDS = float(os.getenv("NEURAL_WINDOW_SECONDS", "5.0"))
NEURAL_MIN_QUERY_SECONDS = float(os.getenv("NEURAL_MIN_QUERY_SECONDS", "2.0"))
NEURAL_DECISION_THRESHOLD = float(os.getenv("NEURAL_DECISION_THRESHOLD", "0.70"))
NEURAL_MODEL_PATH = get_path("NEURAL_MODEL_PATH", "data/models/neural_pair_classifier.pt")
NEURAL_N_MELS = int(os.getenv("NEURAL_N_MELS", "80"))
NEURAL_MEL_HOP_LENGTH = int(os.getenv("NEURAL_MEL_HOP_LENGTH", "384"))
NEURAL_MEL_N_FFT = int(os.getenv("NEURAL_MEL_N_FFT", str(N_FFT)))
NEURAL_INPUT_MODE = os.getenv("NEURAL_INPUT_MODE", "symmetric_mean_absdiff")
NEURAL_TRAIN_BATCH_SIZE = int(os.getenv("NEURAL_TRAIN_BATCH_SIZE", "128"))
NEURAL_TRAIN_EPOCHS = int(os.getenv("NEURAL_TRAIN_EPOCHS", "30"))
NEURAL_TRAIN_LR = float(os.getenv("NEURAL_TRAIN_LR", "1e-3"))
NEURAL_TRAIN_WEIGHT_DECAY = float(os.getenv("NEURAL_TRAIN_WEIGHT_DECAY", "1e-4"))
NEURAL_TRAIN_MIXED_PRECISION = os.getenv("NEURAL_TRAIN_MIXED_PRECISION", "True").lower() in ("true", "1", "yes")
NEURAL_TRAIN_NUM_WORKERS = int(os.getenv("NEURAL_TRAIN_NUM_WORKERS", "4"))
NEURAL_SPLIT_PATH = get_path("NEURAL_SPLIT_PATH", "data/neural/splits/song_split.json")
NEURAL_SPLIT_TRAIN_RATIO = float(os.getenv("NEURAL_SPLIT_TRAIN_RATIO", "0.80"))
NEURAL_SPLIT_VALIDATION_RATIO = float(os.getenv("NEURAL_SPLIT_VALIDATION_RATIO", "0.10"))
NEURAL_SPLIT_TEST_RATIO = float(os.getenv("NEURAL_SPLIT_TEST_RATIO", "0.10"))
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
