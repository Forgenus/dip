import sys
from dotenv import load_dotenv
import argparse
import config as cfg
from pathlib import Path
from src.recognition.service import MusicRecognitionService # type: ignore
from src.testing.test_runner import TestRunner
import time
log = print

#в stft нормализуется и потом еще _to_db происходит, оставить одно и потестить
#убрать вызов напрямую к lookup_flat_batch к db.fingerprints проксировать через db
def main():
    load_dotenv()
    service = MusicRecognitionService() 
    try:
        while True:
            # Получаем строку команды от пользователя
            cmd_input = input(">>> ").strip()
            if cmd_input.lower() == "exit":
                break  # Выход из программы

            # Разбиваем строку на аргументы, как если бы они были из командной строки
            import shlex
            try:
                args_list = shlex.split(cmd_input)
            except ValueError as e:
                print(f"Ошибка разбора аргументов: {e}")
                continue

            # Настройка парсера
            parser = argparse.ArgumentParser(description="Music Recognition Service")
            subparsers = parser.add_subparsers(dest="action", required=True)

            parser_process = subparsers.add_parser("process")
            parser_process.add_argument("--src", type=Path, default=cfg.SONGS_DIR)

            parser_add = subparsers.add_parser("add")
            parser_add.add_argument("--src", type=Path, required=True)

            parser_find = subparsers.add_parser("find")
            parser_find.add_argument("--src", type=Path, required=True)
        
           

            parser_recreate = subparsers.add_parser("recreate")
            parser_recreate.add_argument("--max", type=int,default=0)


            parser_test = subparsers.add_parser(
                "test",
                help="Запуск тестирования системы распознавания"
            )

            # Очистка базы перед тестом
            parser_test.add_argument(
                "--reset-db",
                action="store_true",
                help="Очистить базу данных перед выполнением теста"
            )

            # Максимальное количество файлов для заполнения БД
            parser_test.add_argument(
                "--max-files",
                type=int,
                default=0,
                help="Максимальное количество файлов для заполнения БД перед тестом"
            )

            # Сколько песен протестировать
            parser_test.add_argument(
                "--test-count",
                type=int,
                default=10,
                help="Количество песен для тестирования"
            )
            parser_test.add_argument(
                "--keep-rng",
                action="store_true",
                help="Не переинициализировать PRNG при каждом вызове"
            )
            # Длительность тестовых фрагментов
            parser_test.add_argument(
                "--snippet-duration",
                type=float,
                default=10.0,
                help="Длительность тестового аудиофрагмента в секундах"
            )

            # Добавление шума
            parser_test.add_argument(
                "--noise",
                action="store_true",
                help="Добавлять шум к тестовым фрагментам"
            )

            parser_test.add_argument(
                "--noise-level",
                type=float,
                default=0.02,
                help="Интенсивность шума (0.0–1.0), используется если указан --noise"
            )

            # Изменение громкости
            parser_test.add_argument(
                "--volume",
                choices=["none", "up", "down", "random"],
                default="none",
                help="Изменение громкости тестовых фрагментов"
            )

            parser_test.add_argument(
                "--volume-factor",
                type=float,
                default=1.5,
                help="Коэффициент изменения громкости (используется при volume != none)"
            )
            subparsers.add_parser("print")
            parser_print_songs = subparsers.add_parser("print-songs")
            parser_print_songs.add_argument(
                "--id",
                type=int,
                default=-1
            )
            subparsers.add_parser("clear")
            subparsers.add_parser("debugsearch")


            try:
                args = parser.parse_args(args_list)
            except SystemExit:
                # argparse вызывает sys.exit() при ошибке
                continue

           
            def do_test(args):
                tr = TestRunner(service)
                tr.run(args)
            def do_process(args): 
                start = time.perf_counter()
                added = service.add_songs_from_folder(args.src or cfg.SONGS_DIR)
                elapsed = time.perf_counter() - start
                log(f"Finished processing folder {args.src or cfg.SONGS_DIR}, added {added} songs")
                log(f"Processing completed in {elapsed:.2f} seconds")
                
                
            def do_add(args):
                service.add_song_from_file(args.src)
            def do_find(args):
                match_id, time_offset = service.search_song_from_file(args.src)
                if match_id == -1:
                    log("No match found")
                else:
                    log(f"Found match: {match_id}")
                    service.print_song_info(match_id)
                    log(f"time offset = {time_offset}")
            def do_debugsearch(args): service.debug_search()
            def do_clear(args): service.db.clear_all()
            def do_recreate(args):
                service.db.clear_all()
                service.add_songs_from_folder(cfg.SONGS_DIR,max_amount = args.max)


            def do_print(args): service.db.print_stats()
            def do_print_songs(args):
                if args.id == -1:service.print_all_songs()
                else: service.print_song_info(args.id)

            action_map = {
                "process": do_process,
                "add": do_add,
                "find": do_find,
                "debugsearch": do_debugsearch,
                "clear": do_clear,
                "recreate": do_recreate,
                "print": do_print,
                "print-songs": do_print_songs,
                "test": do_test
            }

            # Выполняем действие
            action_map[args.action](args)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed")
    finally:
        service.close()





if __name__ == "__main__":
    main()