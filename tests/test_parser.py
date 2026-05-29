import unittest

from src.cli.parser import build_parser


class ParserTests(unittest.TestCase):
    def test_test_command_accepts_snippet_effect_flags(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "test",
                "--noise",
                "--noise-level",
                "0.03",
                "--volume",
                "up",
                "--volume-factor",
                "1.8",
                "--time-stretch-rate",
                "1.05",
                "--save-test-snippets",
            ]
        )

        self.assertTrue(args.noise)
        self.assertEqual(0.03, args.noise_level)
        self.assertEqual("up", args.volume)
        self.assertEqual(1.8, args.volume_factor)
        self.assertEqual(1.05, args.time_stretch_rate)
        self.assertTrue(args.save_test_snippets)

    def test_debug_effects_command_accepts_export_flags(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "debug-effects",
                "--snippet-duration",
                "6",
                "--output-dir",
                "data/debug/custom_effects",
                "--align-snippet-start",
                "--noise-level",
                "0.04",
                "--volume-factor",
                "1.7",
                "--time-stretch-rate",
                "0.9",
            ]
        )

        self.assertEqual("debug-effects", args.action)
        self.assertEqual(6, args.snippet_duration)
        self.assertEqual("data\\debug\\custom_effects", str(args.output_dir))
        self.assertTrue(args.align_snippet_start)
        self.assertEqual(0.04, args.noise_level)
        self.assertEqual(1.7, args.volume_factor)
        self.assertEqual(0.9, args.time_stretch_rate)

    def test_test_command_accepts_offset_fallback_and_failure_analysis_flags(self):
        parser = build_parser()

        default_args = parser.parse_args(["test"])
        disabled_args = parser.parse_args(
            [
                "test",
                "--no-offset-fallback",
                "--failure-analysis",
            ]
        )

        self.assertTrue(default_args.offset_fallback)
        self.assertFalse(default_args.failure_analysis)
        self.assertFalse(disabled_args.offset_fallback)
        self.assertTrue(disabled_args.failure_analysis)

    def test_test_command_enables_neural_shadow_by_default(self):
        parser = build_parser()

        default_args = parser.parse_args(["test"])
        disabled_args = parser.parse_args(["test", "--no-neural-shadow"])

        self.assertTrue(default_args.neural_shadow)
        self.assertFalse(disabled_args.neural_shadow)

    def test_neural_train_command_accepts_training_flags(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "neural-train",
                "--split",
                "data/neural/custom_split.json",
                "--epochs",
                "2",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--examples-per-epoch",
                "16",
                "--validation-examples",
                "6",
                "--num-workers",
                "0",
                "--fresh",
            ]
        )

        self.assertEqual("neural-train", args.action)
        self.assertEqual("data\\neural\\custom_split.json", str(args.split))
        self.assertEqual(2, args.epochs)
        self.assertEqual(8, args.batch_size)
        self.assertEqual("cpu", args.device)
        self.assertEqual(16, args.examples_per_epoch)
        self.assertEqual(6, args.validation_examples)
        self.assertEqual(0, args.num_workers)
        self.assertTrue(args.fresh)

    def test_neural_train_reuses_checkpoint_by_default(self):
        parser = build_parser()

        args = parser.parse_args(["neural-train"])

        self.assertFalse(args.fresh)


if __name__ == "__main__":
    unittest.main()
