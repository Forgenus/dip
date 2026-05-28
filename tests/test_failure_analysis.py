import unittest

from src.testing.failure_analysis import (
    fingerprints_by_address,
    select_anchor_time_window,
    summarize_common_offsets,
)
from src.processing.fingerprint import encode_address, encode_hash


class FailureAnalysisTests(unittest.TestCase):
    def test_summarize_common_offsets_groups_common_address_anchor_deltas(self):
        address_a = encode_address(10, 20, 30, 40, 3)
        address_b = encode_address(11, 21, 31, 41, 4)
        snippet = [
            (address_a, encode_hash(10, 0)),
            (address_b, encode_hash(20, 0)),
        ]
        original = [
            (address_a, encode_hash(3, 0)),
            (address_b, encode_hash(12, 0)),
        ]

        summary = summarize_common_offsets(
            fingerprints_by_address(snippet),
            fingerprints_by_address(original),
            bucket_size=7,
        )

        self.assertEqual(summary.top_buckets, {7: 2})
        self.assertEqual(summary.count, 2)
        self.assertEqual(summary.minimum, 7)
        self.assertEqual(summary.maximum, 8)

    def test_select_anchor_time_window_keeps_hashes_by_anchor_time(self):
        fingerprints = [
            (1, encode_hash(9, 0)),
            (2, encode_hash(10, 0)),
            (3, encode_hash(12, 0)),
            (4, encode_hash(15, 0)),
        ]

        selected = select_anchor_time_window(fingerprints, start_frame=10, end_frame=12)

        self.assertEqual(selected, fingerprints[1:3])


if __name__ == "__main__":
    unittest.main()
