import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.conversion import DEFAULT_NUM_THREADS, process_directory, setup_logging
import config as cfg


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
