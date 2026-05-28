import unittest
from unittest.mock import patch

import numpy as np

from src.testing.snippet_effects import SnippetEffects, apply_snippet_effects


class FixedRng:
    def normal(self, loc, scale, size):
        return np.full(size, scale, dtype=np.float32)

    def uniform(self, low, high):
        return high


class SnippetEffectsTests(unittest.TestCase):
    def test_volume_up_multiplies_audio(self):
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        effects = SnippetEffects(volume="up", volume_factor=2.0)

        processed = apply_snippet_effects(audio, sample_rate=11025, rng=FixedRng(), effects=effects)

        np.testing.assert_allclose(processed, np.array([0.2, -0.4, 0.6], dtype=np.float32))

    def test_noise_adds_configured_noise_level(self):
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        effects = SnippetEffects(noise=True, noise_level=0.05)

        processed = apply_snippet_effects(audio, sample_rate=11025, rng=FixedRng(), effects=effects)

        np.testing.assert_allclose(processed, np.array([0.15, -0.15, 0.35], dtype=np.float32))

    def test_time_stretch_uses_configured_rate(self):
        audio = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        effects = SnippetEffects(time_stretch_rate=1.25)

        with patch(
            "src.testing.snippet_effects.librosa.effects.time_stretch",
            return_value=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        ) as time_stretch:
            processed = apply_snippet_effects(audio, sample_rate=11025, rng=FixedRng(), effects=effects)

        time_stretch.assert_called_once()
        self.assertEqual(1.25, time_stretch.call_args.kwargs["rate"])
        np.testing.assert_allclose(processed, np.array([0.1, -0.2, 0.3], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
