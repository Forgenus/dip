import unittest

import config as cfg
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


class NeuralConfigTests(unittest.TestCase):
    def test_neural_training_config_defaults_exist(self):
        self.assertEqual(80, cfg.NEURAL_N_MELS)
        self.assertEqual(384, cfg.NEURAL_MEL_HOP_LENGTH)
        self.assertEqual(1024, cfg.NEURAL_MEL_N_FFT)
        self.assertEqual("symmetric_mean_absdiff", cfg.NEURAL_INPUT_MODE)
        self.assertEqual(128, cfg.NEURAL_TRAIN_BATCH_SIZE)
        self.assertEqual(30, cfg.NEURAL_TRAIN_EPOCHS)
        self.assertEqual(1e-3, cfg.NEURAL_TRAIN_LR)
        self.assertEqual(1e-4, cfg.NEURAL_TRAIN_WEIGHT_DECAY)
        self.assertTrue(cfg.NEURAL_TRAIN_MIXED_PRECISION)
        self.assertEqual(4, cfg.NEURAL_TRAIN_NUM_WORKERS)
        self.assertEqual(0.80, cfg.NEURAL_SPLIT_TRAIN_RATIO)
        self.assertEqual(0.10, cfg.NEURAL_SPLIT_VALIDATION_RATIO)
        self.assertEqual(0.10, cfg.NEURAL_SPLIT_TEST_RATIO)


if __name__ == "__main__":
    unittest.main()
