import logging

from src.cli.parser import build_parser
from src.cli.commands import CommandHandler
from src.recognition.service import MusicRecognitionService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()


    service = MusicRecognitionService()

    handler = CommandHandler(service)

    try:
        while True:
            try:
                command = input(">>> ").strip()
            except KeyboardInterrupt:
                print("\nCtrl+C pressed")
                break

            if command.lower() in {"exit", "quit"}:
                break

            if not command:
                continue

            handler.handle(command, parser)
    finally:
        service.close()


if __name__ == "__main__":
    main()
