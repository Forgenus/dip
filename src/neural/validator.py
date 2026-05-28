from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import torch

import config as cfg
from src.neural import audio_windows
from src.neural.features import TorchMelPairFeatureExtractor
from src.neural.model import PairClassifier
from src.processing import preprocess as pp
from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace


@dataclass
class NeuralValidationResult:
    checked: bool
    reason: str
    results: list[NeuralCandidateTrace]
    error: str | None = None


class NeuralValidator:
    def __init__(
        self,
        db,
        model=None,
        enabled: bool = cfg.NEURAL_SHADOW_ENABLED,
        threshold: float = cfg.NEURAL_DECISION_THRESHOLD,
        top_n: int = cfg.NEURAL_SHADOW_TOP_N,
        sample_rate: int = cfg.SAMPLE_RATE,
        window_seconds: float = cfg.NEURAL_WINDOW_SECONDS,
        min_query_seconds: float = cfg.NEURAL_MIN_QUERY_SECONDS,
        model_path: Path = cfg.NEURAL_MODEL_PATH,
        n_mels: int = cfg.NEURAL_N_MELS,
        n_fft: int = cfg.NEURAL_MEL_N_FFT,
        hop_length: int = cfg.NEURAL_MEL_HOP_LENGTH,
    ) -> None:
        self.db = db
        self.model = model
        self.enabled = enabled
        self.threshold = threshold
        self.top_n = top_n
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.min_query_seconds = min_query_seconds
        self.model_path = model_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_extractor = TorchMelPairFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.feature_extractor.eval()

    def evaluate_top_candidates(
        self,
        query_audio,
        candidates: list[CandidateTrace],
    ) -> NeuralValidationResult:
        if not self.enabled:
            return NeuralValidationResult(False, "disabled", [])
        if not candidates:
            return NeuralValidationResult(False, "no_candidates", [])

        try:
            query_window, query_meta = audio_windows.prepare_query_window(
                query_audio,
                sample_rate=self.sample_rate,
                window_seconds=self.window_seconds,
                min_query_seconds=self.min_query_seconds,
            )
            if query_meta.skipped:
                return NeuralValidationResult(False, "query_too_short", [])

            model = None
            results: list[NeuralCandidateTrace] = []

            for candidate in candidates[: self.top_n]:
                song = self.db.get_song_by_id(candidate.song_id)
                if not song:
                    continue

                file_path = self._song_file_path(song)
                if file_path is None:
                    continue

                candidate_audio = pp.load_audio(file_path, target_sr=self.sample_rate)
                candidate_window, candidate_meta = audio_windows.crop_candidate_window(
                    candidate_audio,
                    sample_rate=self.sample_rate,
                    start_seconds=candidate.time_offset_seconds,
                    window_seconds=self.window_seconds,
                )
                if candidate_meta.skipped:
                    continue

                if model is None:
                    model = self._load_model()
                probability = self._predict_probability(
                    model,
                    query_window,
                    candidate_window,
                )
                decision = "same" if probability >= self.threshold else "not_same"
                results.append(
                    NeuralCandidateTrace(
                        song_id=candidate.song_id,
                        rank=candidate.rank,
                        fingerprint_score=candidate.score,
                        fingerprint_max_count=candidate.max_count,
                        fingerprint_time_offset_seconds=candidate.time_offset_seconds,
                        same_probability=probability,
                        decision=decision,
                        threshold=self.threshold,
                        reliability=query_meta.reliability,
                        query_valid_seconds=query_meta.valid_seconds,
                        candidate_valid_seconds=candidate_meta.valid_seconds,
                        padding_ratio=query_meta.padding_ratio,
                    )
                )

            return NeuralValidationResult(True, "shadow_wide", results)
        except Exception as error:
            return NeuralValidationResult(True, "shadow_wide", [], error=str(error))

    def _load_model(self):
        if self.model is not None:
            self.model.eval()
            return self.model

        model = PairClassifier(input_channels=2)
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.model = model
        return model

    def _predict_probability(self, model, query_window, candidate_window) -> float:
        query_tensor = torch.as_tensor(query_window, dtype=torch.float32)
        candidate_tensor = torch.as_tensor(candidate_window, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Empty filters detected in mel frequency basis.*",
                category=UserWarning,
            )
            with torch.no_grad():
                batch = self.feature_extractor(query_tensor, candidate_tensor)
                logits = model(batch)
                probabilities = torch.sigmoid(logits)
        probability = probabilities.reshape(-1)[0].item()
        return round(float(probability), 6)

    def _song_file_path(self, song: Any) -> Path | None:
        if isinstance(song, dict):
            file_path = song.get("file_path")
        else:
            file_path = getattr(song, "file_path", None)
        return Path(file_path) if file_path is not None else None
