"""Batch indexing orchestration for music files."""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import logging
import time

import config as cfg
from ..processing import preprocess as pp
from .indexing import compute_payload


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingSong:
    file_path: Path
    song_id: int


class BatchIndexer:
    def __init__(self, db, metadata: dict) -> None:
        self.db = db
        self.metadata = metadata

    def add_songs_from_folder(
        self,
        folder_path: Path,
        files: list[Path],
        max_amount: int = 0,
        max_workers: int | None = None,
    ) -> int:
        if max_amount and max_amount > 0:
            files = files[:max_amount]
        logger.info("[main] scan folder done: files=%s max_amount=%s", len(files), max_amount)

        if max_workers is None:
            max_workers = 12
        elif max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        max_pending = max_workers
        wait_log_seconds = getattr(cfg, "PROCESS_WAIT_LOG_SECONDS", 5)
        stall_timeout_seconds = getattr(cfg, "PROCESS_STALL_TIMEOUT_SECONDS", 60)

        added = 0
        submitted = 0
        completed = 0
        pending: dict[Future, PendingSong] = {}
        file_iter = iter(files)

        logger.info("[main] ProcessPoolExecutor create start: max_workers=%s max_pending=%s", max_workers, max_pending)
        ex = ProcessPoolExecutor(max_workers=max_workers)
        logger.info("[main] ProcessPoolExecutor create done")
        stalled = False
        try:
            logger.info("[main] initial submit start")
            for _ in range(max_pending):
                submitted += self._submit_next(ex, file_iter, pending)
            logger.info("[main] initial submit done: pending=%s submitted=%s", len(pending), submitted)

            last_progress_at = time.monotonic()
            while pending:
                logger.info("[main] wait start: pending=%s completed=%s added=%s", len(pending), completed, added)
                done, _ = wait(
                    set(pending),
                    timeout=wait_log_seconds,
                    return_when=FIRST_COMPLETED,
                )
                logger.info("[main] wait returned: done=%s pending=%s", len(done), len(pending) - len(done))

                if not done:
                    idle_seconds = time.monotonic() - last_progress_at
                    logger.warning(
                        "Waiting for %s pending worker tasks for %.0f seconds: %s",
                        len(pending),
                        idle_seconds,
                        self._pending_descriptions(pending),
                    )
                    if stall_timeout_seconds > 0 and idle_seconds >= stall_timeout_seconds:
                        logger.error(
                            "Worker pool stalled for %.0f seconds; cancelling %s pending tasks: %s",
                            idle_seconds,
                            len(pending),
                            self._pending_descriptions(pending),
                        )
                        stalled = True
                        break
                    continue

                last_progress_at = time.monotonic()

                for fut in done:
                    pending_song = pending.pop(fut)
                    completed += 1
                    if self._handle_completed_future(fut, pending_song, completed):
                        added += 1

              
                while len(pending) < max_pending:
                    submitted_now = self._submit_next(ex, file_iter, pending)
                    submitted += submitted_now
                    if not submitted_now:
                        break

        finally:
            self._cleanup_executor(ex, pending, stalled)

        logger.info(
            "Worker pool finished: folder=%s submitted=%s completed=%s added=%s stalled=%s",
            folder_path,
            submitted,
            completed,
            added,
            stalled,
        )

        logger.info("[main] save_all start")
        self.db.save_all()
        logger.info("[main] save_all done")
        return added

    def _submit_next(
        self,
        executor: ProcessPoolExecutor,
        file_iter,
        pending: dict[Future, PendingSong],
    ) -> int:
        for file_path in file_iter:
            file_path = Path(file_path)
            if self.db.is_path_exists(file_path):
                logger.info("File already processed, skipping: %s", file_path)
                continue

            song_id = self.db.reserve_song_id()
            future = executor.submit(compute_payload, file_path, song_id)
            pending[future] = PendingSong(file_path=file_path, song_id=song_id)
            return 1
        return 0

    def _handle_completed_future(self, fut: Future, pending_song: PendingSong, completed: int) -> bool:
        try:
            payload = fut.result()
        except Exception:
            logger.exception(
                "Worker error: song_id=%s file=%s",
                pending_song.song_id,
                pending_song.file_path,
            )
            return False

        metadata = self._extract_metadata(payload.file_path)
        return self._add_payload(payload, metadata)

    def _extract_metadata(self, file_path: Path) -> dict:
        metadata = pp.extract_metadata(file_path, self.metadata)
        return metadata

    def _add_payload(self, payload, metadata: dict) -> bool:
        try:

            self.db.add_song(
                song_id=payload.song_id,
                title=metadata.get("title", ""),
                artist=metadata.get("artist", ""),
                genre=metadata.get("genre", ""),
                year=metadata.get("year", ""),
                album=metadata.get("album", ""),
                file_path=payload.file_path,
                fingerprints=payload.fingerprints,
                duration=metadata.get("duration", "0.0"),
                save_after=False,
            )
            return True
        except TypeError:
            logger.exception(
                "Error while adding file: song_id=%s title=%s file=%s duration=%s",
                payload.song_id,
                metadata.get("title"),
                payload.file_path,
                metadata.get("duration"),
            )
            return False

    def _cleanup_executor(
        self,
        executor: ProcessPoolExecutor,
        pending: dict[Future, PendingSong],
        stalled: bool,
    ) -> None:
        logger.info("[main] executor cleanup start: pending=%s stalled=%s", len(pending), stalled)
        for fut in pending:
            fut.cancel()

        if stalled and hasattr(executor, "terminate_workers"):
            executor.terminate_workers()
        else:
            executor.shutdown(wait=not stalled, cancel_futures=True)
        logger.info("[main] executor cleanup done")

    @staticmethod
    def _pending_descriptions(pending: dict[Future, PendingSong]) -> list[str]:
        return [f"song_id={item.song_id} file={item.file_path}" for item in pending.values()]
