import argparse
import shlex
import time

import config as cfg
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
        match_id, time_offset = self.service.search_song_from_file(args.src)

        if match_id == -1:
            print("No match found")
            return

        print(f"Found match: {match_id}")
        self.service.print_song_info(match_id)
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
            self.service.print_all_songs()
        else:
            self.service.print_song_info(args.id)