import unittest

import numpy as np

from src.processing.fft_filter import (
    filter_spectrogram,
    filter_spectrogram_log_band_strongest,
)


class FftFilterTests(unittest.TestCase):
    def test_log_band_strongest_keeps_band_maxima_above_scaled_mean(self):
        data = np.array(
            [
                [0.0, 2.0, 0.5, 4.0, 0.2, 0.1, 1.0, 0.3],
            ]
        )

        points = filter_spectrogram_log_band_strongest(
            data,
            band_edges=((0, 2), (2, 4), (4, 8)),
            mean_coefficient=1.0,
        )

        self.assertEqual(points.tolist(), [[0, 3]])

    def test_filter_spectrogram_dispatches_log_band_strongest(self):
        data = np.array(
            [
                [0.0, 2.0, 0.5, 4.0, 0.2, 0.1, 1.0, 0.3],
            ]
        )

        points = filter_spectrogram(
            data,
            method="log_band_strongest",
        )

        self.assertEqual(points.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
