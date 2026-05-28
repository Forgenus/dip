import unittest

from src.neural.evaluation import confusion_at_threshold, grouped_confusion


class NeuralEvaluationTests(unittest.TestCase):
    def test_confusion_at_threshold_counts_binary_outcomes(self):
        rows = [
            {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
            {"label": 1, "probability": 0.2, "pair_type": "positive_jittered"},
            {"label": 0, "probability": 0.8, "pair_type": "negative_hard"},
            {"label": 0, "probability": 0.1, "pair_type": "negative_random"},
        ]

        report = confusion_at_threshold(rows, threshold=0.7)

        self.assertEqual(1, report["TP"])
        self.assertEqual(1, report["FN"])
        self.assertEqual(1, report["FP"])
        self.assertEqual(1, report["TN"])
        self.assertEqual(0.5, report["precision"])
        self.assertEqual(0.5, report["recall"])

    def test_confusion_at_threshold_empty_rows_returns_zero_counts_and_rates(self):
        report = confusion_at_threshold([], threshold=0.7)

        self.assertEqual(0, report["TP"])
        self.assertEqual(0, report["FN"])
        self.assertEqual(0, report["FP"])
        self.assertEqual(0, report["TN"])
        self.assertEqual(0.0, report["precision"])
        self.assertEqual(0.0, report["recall"])
        self.assertEqual(0.0, report["false_positive_rate"])
        self.assertEqual(0.0, report["false_negative_rate"])

    def test_confusion_at_threshold_treats_threshold_equality_as_predicted_same(self):
        rows = [
            {"label": 1, "probability": 0.7, "pair_type": "positive_same_time"},
            {"label": 0, "probability": 0.7, "pair_type": "negative_hard"},
        ]

        report = confusion_at_threshold(rows, threshold=0.7)

        self.assertEqual(1, report["TP"])
        self.assertEqual(0, report["FN"])
        self.assertEqual(1, report["FP"])
        self.assertEqual(0, report["TN"])

    def test_confusion_at_threshold_handles_all_positive_denominators(self):
        rows = [
            {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
            {"label": 1, "probability": 0.2, "pair_type": "positive_jittered"},
        ]

        report = confusion_at_threshold(rows, threshold=0.7)

        self.assertEqual(1, report["TP"])
        self.assertEqual(1, report["FN"])
        self.assertEqual(0, report["FP"])
        self.assertEqual(0, report["TN"])
        self.assertEqual(1.0, report["precision"])
        self.assertEqual(0.5, report["recall"])
        self.assertEqual(0.0, report["false_positive_rate"])
        self.assertEqual(0.5, report["false_negative_rate"])

    def test_confusion_at_threshold_handles_all_negative_denominators(self):
        rows = [
            {"label": 0, "probability": 0.9, "pair_type": "negative_hard"},
            {"label": 0, "probability": 0.2, "pair_type": "negative_random"},
        ]

        report = confusion_at_threshold(rows, threshold=0.7)

        self.assertEqual(0, report["TP"])
        self.assertEqual(0, report["FN"])
        self.assertEqual(1, report["FP"])
        self.assertEqual(1, report["TN"])
        self.assertEqual(0.0, report["precision"])
        self.assertEqual(0.0, report["recall"])
        self.assertEqual(0.5, report["false_positive_rate"])
        self.assertEqual(0.0, report["false_negative_rate"])

    def test_grouped_confusion_reports_each_group(self):
        rows = [
            {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
            {"label": 0, "probability": 0.1, "pair_type": "negative_random"},
        ]

        grouped = grouped_confusion(rows, threshold=0.7, group_key="pair_type")

        self.assertEqual({"negative_random", "positive_same_time"}, set(grouped.keys()))
        self.assertEqual(1, grouped["positive_same_time"]["TP"])
        self.assertEqual(1, grouped["negative_random"]["TN"])

    def test_grouped_confusion_accepts_generator_input_and_groups_in_one_pass(self):
        rows = (
            row
            for row in [
                {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
                {"label": 0, "probability": 0.8, "pair_type": "negative_hard"},
                {"label": 0, "probability": 0.1, "pair_type": "negative_hard"},
            ]
        )

        grouped = grouped_confusion(rows, threshold=0.7, group_key="pair_type")

        self.assertEqual({"negative_hard", "positive_same_time"}, set(grouped.keys()))
        self.assertEqual(1, grouped["positive_same_time"]["TP"])
        self.assertEqual(1, grouped["negative_hard"]["FP"])
        self.assertEqual(1, grouped["negative_hard"]["TN"])
        self.assertEqual([], list(rows))

    def test_grouped_confusion_missing_group_key_raises_key_error(self):
        rows = [
            {"label": 1, "probability": 0.9},
        ]

        with self.assertRaises(KeyError):
            grouped_confusion(rows, threshold=0.7, group_key="pair_type")


if __name__ == "__main__":
    unittest.main()
