import unittest

from src.neural.dataset import (
    PairExample,
    duration_bucket,
    make_hard_negative,
    make_random_negative,
    make_same_time_positive,
    padding_ratio,
)


SONGS = [
    {"song_id": 1, "file_path": "a.wav", "duration": 20.0},
    {"song_id": 2, "file_path": "b.wav", "duration": 30.0},
]


class NeuralDatasetTests(unittest.TestCase):
    def test_make_same_time_positive_uses_same_song_and_label_one(self):
        example = make_same_time_positive(SONGS[0], start_seconds=4.0, query_valid_seconds=5.0)

        self.assertEqual(1, example.label)
        self.assertEqual(1, example.query_song_id)
        self.assertEqual(1, example.candidate_song_id)
        self.assertEqual("positive_same_time", example.pair_type)
        self.assertEqual(0.0, example.padding_ratio)

    def test_make_random_negative_uses_different_songs_and_label_zero(self):
        example = make_random_negative(
            query_song=SONGS[0],
            candidate_song=SONGS[1],
            query_start_seconds=2.0,
            candidate_start_seconds=3.0,
            query_valid_seconds=3.0,
        )

        self.assertEqual(0, example.label)
        self.assertEqual(1, example.query_song_id)
        self.assertEqual(2, example.candidate_song_id)
        self.assertEqual("negative_random", example.pair_type)
        self.assertAlmostEqual(0.4, example.padding_ratio)

    def test_duration_bucket_names_query_duration_ranges(self):
        self.assertEqual("5.0s", duration_bucket(5.0))
        self.assertEqual("4-5s", duration_bucket(4.2))
        self.assertEqual("3-4s", duration_bucket(3.1))
        self.assertEqual("2-3s", duration_bucket(2.5))

    def test_padding_ratio_handles_zero_window_and_overfull_query(self):
        self.assertEqual(0.0, padding_ratio(0, 0))
        self.assertEqual(0.0, padding_ratio(6, 5))

    def test_duration_bucket_uses_exact_boundaries(self):
        self.assertEqual("4-5s", duration_bucket(4.999))
        self.assertEqual("4-5s", duration_bucket(4.0))
        self.assertEqual("3-4s", duration_bucket(3.999))
        self.assertEqual("3-4s", duration_bucket(3.0))
        self.assertEqual("2-3s", duration_bucket(2.0))

    def test_make_random_negative_rejects_same_song(self):
        with self.assertRaises(ValueError):
            make_random_negative(
                query_song=SONGS[0],
                candidate_song=SONGS[0],
                query_start_seconds=2.0,
                candidate_start_seconds=3.0,
                query_valid_seconds=3.0,
            )

    def test_make_hard_negative_rejects_same_song(self):
        with self.assertRaises(ValueError):
            make_hard_negative(
                query_song=SONGS[0],
                candidate_song=SONGS[0],
                query_start_seconds=2.0,
                candidate_offset_seconds=3.0,
                query_valid_seconds=3.0,
            )


if __name__ == "__main__":
    unittest.main()
