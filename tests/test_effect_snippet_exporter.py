import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.testing.snippets import AudioSnippet
from src.testing.effect_snippet_exporter import EffectSnippetExporter


class FakeService:
    def get_random_song(self, rng):
        return {
            "song_id": 12,
            "title": "Example Song",
            "file_path": Path("songs/example.wav"),
        }


class EffectSnippetExporterTests(unittest.TestCase):
    def test_exports_original_and_each_effect_variant(self):
        snippet = AudioSnippet(
            audio=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            source_file=Path("songs/example.wav"),
            start_sample=11025,
            start_seconds=1.0,
            duration_seconds=3 / 11025,
            sample_rate=11025,
        )
        args = SimpleNamespace(
            snippet_duration=8.0,
            output_dir=Path("out"),
            align_snippet_start=True,
            noise_level=0.04,
            volume_factor=1.7,
            time_stretch_rate=0.9,
        )

        with patch(
            "src.testing.effect_snippet_exporter.create_snippet_from_file",
            return_value=snippet,
        ) as create_snippet, patch(
            "src.testing.effect_snippet_exporter.save_snippet_to_file",
        ) as save_snippet, patch(
            "src.testing.effect_snippet_exporter.apply_snippet_effects",
            side_effect=lambda audio, sample_rate, rng, effects: audio + 1,
        ) as apply_effects:
            saved_files = EffectSnippetExporter(FakeService()).run(args)

        create_snippet.assert_called_once()
        self.assertEqual(Path("songs/example.wav"), create_snippet.call_args.kwargs["file_path"])
        self.assertEqual(8.0, create_snippet.call_args.kwargs["snippet_duration"])
        self.assertTrue(create_snippet.call_args.kwargs["align_to_hop"])

        self.assertEqual(6, len(saved_files))
        self.assertEqual(6, save_snippet.call_count)
        self.assertEqual(5, apply_effects.call_count)

        saved_names = [path.name for path in saved_files]
        self.assertTrue(any(name.endswith("_original.wav") for name in saved_names))
        self.assertTrue(any(name.endswith("_noise.wav") for name in saved_names))
        self.assertTrue(any(name.endswith("_volume_up.wav") for name in saved_names))
        self.assertTrue(any(name.endswith("_volume_down.wav") for name in saved_names))
        self.assertTrue(any(name.endswith("_volume_random.wav") for name in saved_names))
        self.assertTrue(any(name.endswith("_time_stretch.wav") for name in saved_names))


if __name__ == "__main__":
    unittest.main()
