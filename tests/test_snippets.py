import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config as cfg
from src.testing.snippets import create_snippet_from_file


class FixedRng:
    def __init__(self, value: int):
        self.value = value

    def integers(self, low: int, high: int) -> int:
        self.low = low
        self.high = high
        return self.value


class CreateSnippetFromFileTests(unittest.TestCase):
    def test_random_start_is_not_aligned_by_default(self):
        audio = np.arange(cfg.SAMPLE_RATE * 20, dtype=np.float32)
        rng = FixedRng(cfg.HOP_LENGTH + 123)

        with patch("src.testing.snippets.librosa.load", return_value=(audio, cfg.SAMPLE_RATE)):
            snippet = create_snippet_from_file(
                Path("song.wav"),
                snippet_duration=8.0,
                rng=rng,
            )

        self.assertEqual(cfg.HOP_LENGTH + 123, snippet.start_sample)
        self.assertNotEqual(0, snippet.start_sample % cfg.HOP_LENGTH)

    def test_align_to_hop_chooses_hop_aligned_start(self):
        audio = np.arange(cfg.SAMPLE_RATE * 20, dtype=np.float32)
        rng = FixedRng(3)

        with patch("src.testing.snippets.librosa.load", return_value=(audio, cfg.SAMPLE_RATE)):
            snippet = create_snippet_from_file(
                Path("song.wav"),
                snippet_duration=8.0,
                rng=rng,
                align_to_hop=True,
            )

        self.assertEqual(3 * cfg.HOP_LENGTH, snippet.start_sample)
        self.assertEqual(0, snippet.start_sample % cfg.HOP_LENGTH)


if __name__ == "__main__":
    unittest.main()
