import io
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.cli.commands import CommandHandler
from src.neural.training import run_training


class NeuralTrainCliTests(unittest.TestCase):
    def test_command_handler_dispatches_neural_train(self):
        handler = CommandHandler(service=object())

        with patch("src.neural.training.run_training") as run:
            handler.neural_train(Namespace(action="neural-train"))

        run.assert_called_once()

    def test_run_training_reports_missing_split(self):
        args = Namespace(
            split=Path("missing-split.json"),
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run_training(args)

        self.assertEqual(1, result)
        self.assertIn("Split file not found", stdout.getvalue())

    def test_run_training_reports_invalid_split_file(self):
        split_path = Path("invalid-song-split.json")
        split_path.write_text("{}", encoding="utf-8")
        args = Namespace(
            split=split_path,
            epochs=3,
            batch_size=4,
            device="cpu",
        )

        try:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                result = run_training(args)
        finally:
            split_path.unlink()

        output = stdout.getvalue()
        self.assertEqual(1, result)
        self.assertIn("Invalid split file", output)


if __name__ == "__main__":
    unittest.main()
