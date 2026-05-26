from pathlib import Path

import config as cfg


def build_ffmpeg_command(src_path: Path, dst_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-af",
        f"lowpass=f={cfg.LOW_PASS_CUTOFF_HZ},volume=0.6",
        "-ac",
        "1",
        "-ar",
        str(cfg.SAMPLE_RATE),
        str(dst_path),
    ]
