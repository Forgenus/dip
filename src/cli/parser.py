import argparse
from pathlib import Path

import config as cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Music Recognition Service",
    )

    subparsers = parser.add_subparsers(
        dest="action",
        required=True,
        help="Доступные команды",
    )

    parser_process = subparsers.add_parser(
        "process",
        help="Добавить в базу аудиофайлы из папки",
        description=(
            "Обрабатывает аудиофайлы из указанной папки: загружает аудио, "
            "строит fingerprints и добавляет песни в базу."
        ),
    )
    parser_process.add_argument(
        "--src",
        type=Path,
        default=cfg.SONGS_DIR,
        help=f"Папка с аудиофайлами для добавления в базу. По умолчанию: {cfg.SONGS_DIR}",
    )
    parser_process.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Максимальное количество worker-процессов для индексации при обработке. "
            "Если не указано, используется значение по умолчанию."
        ),
    )

    parser_add = subparsers.add_parser(
        "add",
        help="Добавить один аудиофайл в базу",
        description="Добавляет один указанный аудиофайл в базу распознавания.",
    )
    parser_add.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Путь к аудиофайлу, который нужно добавить в базу.",
    )

    parser_find = subparsers.add_parser(
        "find",
        help="Найти песню по аудиофайлу",
        description=(
            "Строит fingerprints для указанного аудиофайла и ищет совпадение "
            "в базе распознавания."
        ),
    )
    parser_find.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Путь к аудиофайлу, по которому нужно найти песню.",
    )

    parser_find.add_argument(
        "--no-offset-fallback",
        dest="offset_fallback",
        action="store_false",
        default=True,
        help="Disable extra non-hop-aligned search attempts for weak results.",
    )

    parser_recreate = subparsers.add_parser(
        "recreate",
        help="Очистить базу и заново добавить песни из SONGS_DIR",
        description=(
            "Полностью очищает базу, после чего заново добавляет аудиофайлы "
            f"из папки {cfg.SONGS_DIR}."
        ),
    )
    parser_recreate.add_argument(
        "--max",
        type=int,
        default=0,
        help=(
            "Максимальное количество файлов для добавления после очистки базы. "
            "0 означает без ограничения."
        ),
    )
    parser_recreate.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Максимальное количество worker-процессов для индексации. "
            "Если не указано, используется значение по умолчанию."
        ),
    )

    parser_test = subparsers.add_parser(
        "test",
        help="Запустить тестирование распознавания",
        description=(
            "Запускает тест распознавания: выбирает случайные песни из базы, "
            "вырезает сниппеты заданной длительности и проверяет, найдёт ли "
            "система правильную песню."
        ),
    )
    parser_test.add_argument(
        "--reset-db",
        action="store_true",
        help=(
            "Перед тестом очистить базу и заново добавить песни из SONGS_DIR. "
            "Использует --max-files как ограничение количества файлов."
        ),
    )
    parser_test.add_argument(
        "--max-files",
        type=int,
        default=0,
        help=(
            "Максимальное количество файлов, добавляемых в базу при --reset-db. "
            "0 означает без ограничения."
        ),
    )
    parser_test.add_argument(
        "--test-count",
        type=int,
        default=10,
        help="Количество тестовых попыток распознавания.",
    )
    parser_test.add_argument(
        "--keep-rng",
        action="store_true",
        help=(
            "Не сбрасывать генератор случайных чисел перед тестом. "
            "Без этого флага генератор сбрасывается к cfg.RNG_SEED."
        ),
    )
    parser_test.add_argument(
        "--snippet-duration",
        type=float,
        default=8,
        help="Длительность тестового аудиофрагмента в секундах.",
    )
    parser_test.add_argument(
        "--failed-snippets-dir",
        type=Path,
        default=cfg.FAILED_SNIPPETS_DIR,
        help="Directory for WAV snippets saved from failed recognition tests.",
    )
    parser_test.add_argument(
        "--align-snippet-start",
        action="store_true",
        help="Align test snippet start sample to HOP_LENGTH for STFT grid diagnostics.",
    )
    parser_test.add_argument(
        "--time-stretch-rate",
        type=float,
        default=1.0,
        help="Apply librosa time-stretch to each test snippet. 1.0 keeps original speed.",
    )
    parser_test.add_argument(
        "--save-test-snippets",
        action="store_true",
        help="Save every generated test snippet after alignment and effects are applied.",
    )
    parser_test.add_argument(
        "--test-snippets-dir",
        type=Path,
        default=cfg.TEST_SNIPPETS_DIR,
        help="Directory for WAV snippets saved with --save-test-snippets.",
    )
    parser_test.add_argument(
        "--noise",
        action="store_true",
        help="Apply white noise to each test snippet.",
    )
    parser_test.add_argument(
        "--noise-level",
        type=float,
        default=0.02,
        help="Standard deviation for white noise added with --noise.",
    )
    parser_test.add_argument(
        "--volume",
        choices=["none", "up", "down", "random"],
        default="none",
        help="Volume manipulation mode for each test snippet.",
    )
    parser_test.add_argument(
        "--volume-factor",
        type=float,
        default=1.5,
        help="Multiplier for --volume up/down/random.",
    )

    parser_test.add_argument(
        "--no-offset-fallback",
        dest="offset_fallback",
        action="store_false",
        default=True,
        help="Disable extra non-hop-aligned search attempts for weak results.",
    )
    parser_test.add_argument(
        "--failure-analysis",
        action="store_true",
        help="Print detailed failed-snippet analysis for failed recognition tests.",
    )
    parser_test.add_argument(
        "--neural-shadow",
        dest="neural_shadow",
        action="store_true",
        default=True,
        help="Run neural validator in shadow mode and print its test impact summary.",
    )
    parser_test.add_argument(
        "--no-neural-shadow",
        dest="neural_shadow",
        action="store_false",
        help="Disable neural shadow validation for this test run.",
    )

    parser_debug_effects = subparsers.add_parser(
        "debug-effects",
        help="Export one random snippet with each audio effect applied separately.",
        description=(
            "Creates one random snippet from a random song in the database, saves the original WAV, "
            "then saves separate WAV files for noise, volume up, volume down, random volume, and time stretch."
        ),
    )
    parser_debug_effects.add_argument(
        "--snippet-duration",
        type=float,
        default=8,
        help="Duration of the exported snippet in seconds.",
    )
    parser_debug_effects.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.EFFECT_SNIPPETS_DIR,
        help=f"Directory for exported effect snippets. Default: {cfg.EFFECT_SNIPPETS_DIR}",
    )
    parser_debug_effects.add_argument(
        "--align-snippet-start",
        action="store_true",
        help="Align snippet start sample to HOP_LENGTH for STFT grid diagnostics.",
    )
    parser_debug_effects.add_argument(
        "--noise-level",
        type=float,
        default=0.02,
        help="Standard deviation for the exported noise variant.",
    )
    parser_debug_effects.add_argument(
        "--volume-factor",
        type=float,
        default=1.5,
        help="Multiplier for exported volume up/down/random variants.",
    )
    parser_debug_effects.add_argument(
        "--time-stretch-rate",
        type=float,
        default=1.1,
        help="Librosa time-stretch rate for the exported time_stretch variant.",
    )

    parser_neural_split = subparsers.add_parser(
        "neural-split",
        help="Create a song-level train/validation/test split for neural training.",
        description="Creates a deterministic song-level split JSON for neural model training.",
    )
    parser_neural_split.add_argument(
        "--output",
        type=Path,
        default=cfg.NEURAL_SPLIT_PATH,
        help=f"Output split JSON path. Default: {cfg.NEURAL_SPLIT_PATH}",
    )
    parser_neural_split.add_argument(
        "--seed",
        type=int,
        default=cfg.RNG_SEED,
        help=f"Random seed for deterministic splitting. Default: {cfg.RNG_SEED}",
    )
    parser_neural_split.add_argument(
        "--train-ratio",
        type=float,
        default=cfg.NEURAL_SPLIT_TRAIN_RATIO,
        help=f"Train split ratio. Default: {cfg.NEURAL_SPLIT_TRAIN_RATIO}",
    )
    parser_neural_split.add_argument(
        "--validation-ratio",
        type=float,
        default=cfg.NEURAL_SPLIT_VALIDATION_RATIO,
        help=f"Validation heldout split ratio. Default: {cfg.NEURAL_SPLIT_VALIDATION_RATIO}",
    )
    parser_neural_split.add_argument(
        "--test-ratio",
        type=float,
        default=cfg.NEURAL_SPLIT_TEST_RATIO,
        help=f"Test heldout split ratio. Default: {cfg.NEURAL_SPLIT_TEST_RATIO}",
    )
    parser_neural_split.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing split file.",
    )

    parser_neural_train = subparsers.add_parser(
        "neural-train",
        help="Train the neural pair classifier model.",
        description="Validates neural training configuration and starts the training entrypoint.",
    )
    parser_neural_train.add_argument(
        "--split",
        type=Path,
        default=cfg.NEURAL_SPLIT_PATH,
        help=f"Path to the song split JSON. Default: {cfg.NEURAL_SPLIT_PATH}",
    )
    parser_neural_train.add_argument(
        "--epochs",
        type=int,
        default=cfg.NEURAL_TRAIN_EPOCHS,
        help=f"Training epoch count. Default: {cfg.NEURAL_TRAIN_EPOCHS}",
    )
    parser_neural_train.add_argument(
        "--batch-size",
        type=int,
        default=cfg.NEURAL_TRAIN_BATCH_SIZE,
        help=f"Training batch size. Default: {cfg.NEURAL_TRAIN_BATCH_SIZE}",
    )
    parser_neural_train.add_argument(
        "--examples-per-epoch",
        type=int,
        default=cfg.NEURAL_TRAIN_EXAMPLES_PER_EPOCH,
        help=f"Training examples sampled per epoch. Default: {cfg.NEURAL_TRAIN_EXAMPLES_PER_EPOCH}",
    )
    parser_neural_train.add_argument(
        "--validation-examples",
        type=int,
        default=cfg.NEURAL_VALIDATION_EXAMPLES,
        help=f"Validation examples sampled per evaluation split. Default: {cfg.NEURAL_VALIDATION_EXAMPLES}",
    )
    parser_neural_train.add_argument(
        "--num-workers",
        type=int,
        default=cfg.NEURAL_TRAIN_NUM_WORKERS,
        help=f"DataLoader worker processes. Default: {cfg.NEURAL_TRAIN_NUM_WORKERS}",
    )
    parser_neural_train.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Default: auto.",
    )
    parser_neural_train.add_argument(
        "--fresh",
        action="store_true",
        help="Start neural training from a new model even if an existing checkpoint is present.",
    )

    subparsers.add_parser(
        "print",
        help="Вывести статистику базы",
        description="Печатает статистику базы распознавания.",
    )

    parser_print_songs = subparsers.add_parser(
        "print-songs",
        help="Вывести информацию о песнях в базе",
        description=(
            "Печатает информацию обо всех песнях в базе или об одной песне, "
            "если указан --id."
        ),
    )
    parser_print_songs.add_argument(
        "--id",
        type=int,
        default=-1,
        help=(
            "ID песни для вывода информации. "
            "-1 означает вывести все песни."
        ),
    )

    subparsers.add_parser(
        "clear",
        help="Очистить базу",
        description="Удаляет все данные из базы распознавания.",
    )

    subparsers.add_parser(
        "debugsearch",
        help="Запустить отладочный поиск по файлам из SONGS_DIR",
        description=(
            "Проходит по аудиофайлам из SONGS_DIR, запускает поиск для каждого "
            "файла и печатает найденный результат."
        ),
    )

    return parser
