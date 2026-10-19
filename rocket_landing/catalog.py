"""Discover models and previously measured results in a project folder."""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SavedModel:
    label: str
    path: Path
    preset: str
    seed: int
    checkpoint: str
    summary: dict | None


def discover_models(root):
    root = Path(root).resolve()
    models = []
    for config in sorted((root / "runs").glob("*/seed_*/config.json")):
        try:
            settings = json.loads(config.read_text(encoding="utf-8"))
            preset, seed = settings["preset"], settings["seed"]
            if not isinstance(preset, str) or not isinstance(seed, int):
                continue
        except (OSError, ValueError, TypeError, KeyError):
            continue
        for checkpoint in ("final", "best"):
            path = config.parent / f"{checkpoint}_model.zip"
            if not path.is_file():
                continue
            summary = None
            try:
                measured = json.loads((config.parent / "evaluation.json").read_text())
                # A final model's test score must never be attached to its best
                # checkpoint merely because the two live in the same folder.
                if (Path(measured["model"]).name == path.name and
                        measured.get("scenario", "calm") == "calm" and
                        measured.get("model_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()):
                    summary = measured
            except (OSError, ValueError, KeyError):
                pass
            models.append(SavedModel(f"{preset.title()} / seed {seed} / {checkpoint}",
                                     path, preset, seed, checkpoint, summary))
    return models
