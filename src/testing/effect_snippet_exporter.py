from dataclasses import replace
from pathlib import Path

import numpy as np

import config as cfg
from src.testing.snippet_effects import SnippetEffects, apply_snippet_effects
from src.testing.snippets import (
    build_effect_snippet_filename,
    create_snippet_from_file,
    save_snippet_to_file,
)


class EffectSnippetExporter:
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, service):
        self.service = service

    def run(self, args) -> list[Path]:
        song = self.service.get_random_song(rng=EffectSnippetExporter.rng)
        if song is None:
            print("No songs in database")
            return []

        output_dir = Path(getattr(args, "output_dir", cfg.EFFECT_SNIPPETS_DIR))
        snippet = create_snippet_from_file(
            file_path=song["file_path"],
            snippet_duration=getattr(args, "snippet_duration", 8.0),
            rng=EffectSnippetExporter.rng,
            align_to_hop=getattr(args, "align_snippet_start", False),
        )

        print(
            f"Exporting effect snippets for {song['title']} "
            f"({song['song_id']}) from {snippet.start_seconds:.2f}s"
        )

        saved_files = [
            self._save_variant(
                snippet=snippet,
                output_dir=output_dir,
                song=song,
                effect_name="original",
            )
        ]

        for effect_name, effects in self._effect_variants(args):
            audio = apply_snippet_effects(
                audio=snippet.audio,
                sample_rate=snippet.sample_rate,
                rng=EffectSnippetExporter.rng,
                effects=effects,
            )
            effected_snippet = replace(
                snippet,
                audio=audio,
                duration_seconds=len(audio) / snippet.sample_rate,
            )
            saved_files.append(
                self._save_variant(
                    snippet=effected_snippet,
                    output_dir=output_dir,
                    song=song,
                    effect_name=effect_name,
                )
            )

        return saved_files

    def _effect_variants(self, args) -> list[tuple[str, SnippetEffects]]:
        noise_level = getattr(args, "noise_level", 0.02)
        volume_factor = getattr(args, "volume_factor", 1.5)
        time_stretch_rate = getattr(args, "time_stretch_rate", 1.0)

        return [
            ("noise", SnippetEffects(noise=True, noise_level=noise_level)),
            ("volume_up", SnippetEffects(volume="up", volume_factor=volume_factor)),
            ("volume_down", SnippetEffects(volume="down", volume_factor=volume_factor)),
            ("volume_random", SnippetEffects(volume="random", volume_factor=volume_factor)),
            ("time_stretch", SnippetEffects(time_stretch_rate=time_stretch_rate)),
        ]

    def _save_variant(self, snippet, output_dir: Path, song, effect_name: str) -> Path:
        filename = build_effect_snippet_filename(
            expected_id=song["song_id"],
            expected_title=song["title"],
            start_seconds=snippet.start_seconds,
            effect_name=effect_name,
        )
        output_file = output_dir / filename
        save_snippet_to_file(snippet, output_file)
        print(f"Saved {effect_name}: {output_file}")
        return output_file
