import argparse
import shlex
import time
from pathlib import Path

import config as cfg
from src.neural.splits import collect_song_items, save_song_split, split_song_items
from src.testing.effect_snippet_exporter import EffectSnippetExporter
from src.testing.test_runner import TestRunner


class CommandHandler:
    def __init__(self, service):
        self.service = service
        self.actions = {
            "process": self.process,
            "add": self.add,
            "find": self.find,
            "debugsearch": self.debug_search,
            "clear": self.clear,
            "recreate": self.recreate,
            "print": self.print_stats,
            "print-songs": self.print_songs,
            "test": self.test,
            "debug-effects": self.debug_effects,
            "neural-split": self.neural_split,
                    "neural-train": self.neural_train,
        }

    def handle(self, command: str, parser: argparse.ArgumentParser) -> None:
        try:
            args_list = shlex.split(command)
           
        except ValueError as error:
            print(f"Ошибка разбора аргументов: {error}")
            return

        try:
            args = parser.parse_args(args_list)
            print(f"Parsed arguments: {args}")
        except SystemExit:
            return

        action = self.actions.get(args.action)
        if action is None:
            print(f"Unknown action: {args.action}")
            return

        action(args)

    def test(self, args) -> None:
        TestRunner(self.service).run(args)

    def debug_effects(self, args) -> None:
        EffectSnippetExporter(self.service).run(args)

    def neural_split(self, args) -> None:
        output_path = Path(args.output)
        if output_path.exists() and not args.force:
            print(f"Split file already exists: {output_path}")
            print("Use --force to overwrite it.")
            return

        try:
            items = collect_song_items(self.service)
        except (TypeError, ValueError) as error:
            print(f"Cannot create neural split: {error}")
            return

        if not items:
            print("Cannot create neural split: song database is empty.")
            return

        try:
            split = split_song_items(
                items,
                seed=args.seed,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
            )
            save_song_split(split, output_path)
        except ValueError as error:
            print(f"Cannot create neural split: {error}")
            return

        counts = split.counts
        print(f"Neural split saved: {output_path}")
        print(
            "Counts: "
            f"train={counts['train']}, "
            f"validation_heldout={counts['validation_heldout']}, "
            f"test_heldout={counts['test_heldout']}, "
            f"total={counts['total']}"
        )

    def neural_train(self, args) -> None:
        """Train the neural pair classifier model (skeleton)."""
        from src.neural.training import run_training
        run_training(args)

    def process(self, args) -> None:
        start = time.perf_counter()
        src = args.src or cfg.SONGS_DIR
        added = self.service.add_songs_from_folder(src)
        elapsed = time.perf_counter() - start

        print(f"Finished processing folder {src}, added {added} songs")
        print(f"Processing completed in {elapsed:.2f} seconds")

    def add(self, args) -> None:
        self.service.add_song_from_file(args.src)

    def find(self, args) -> None:
        match_id, time_offset = self.service.search_song_from_file(
            args.src,
            offset_fallback=getattr(args, "offset_fallback", True),
        )

        if match_id == -1:
            print("No match found")
            return

        print(f"Found match: {match_id}")
        self._print_song_info(match_id)
        print(f"time offset = {time_offset}")

    def debug_search(self, args) -> None:
        self.service.debug_search()

    def clear(self, args) -> None:
        self.service.clear_all()

    def recreate(self, args) -> None:
        self.service.clear_all()
        self.service.add_songs_from_folder(cfg.SONGS_DIR, max_amount=args.max)

    def print_stats(self, args) -> None:
        self.service.print_stats()

    def print_songs(self, args) -> None:
        if args.id == -1:
            for song_id in self.service.db.songs.db.keys():
                self._print_song_info(song_id)
                print("-" * 20)
        else:
            self._print_song_info(args.id)

    def _print_song_info(self, song_id: int) -> None:
        song_info = self.service.get_song_by_id(song_id)
        if not song_info:
            print(f"Song with ID {song_id} not found")
            return

        print(f"ID: {song_info['song_id']}")
        print(f"Title: {song_info['title']}")
        print(f"Artist: {song_info['artist']}")
        print(f"Album: {song_info['album']}")
        print(f"Year: {song_info['year']}")
        print(f"Genre: {song_info['genre']}")
        print(f"Duration: {song_info['duration']} seconds")
        print(f"File Path: {song_info['file_path']}")
        print(f"FPs: {song_info['fingerprint_count']}")
