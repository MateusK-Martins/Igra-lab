"""Export final inference weights with their tokenizer and run identity."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from src.config.snapshot import normalize_settings


def export_model(directory: Path, *, model, tokenizer_path, manifest, state):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".model.pt.tmp"
    try:
        torch.save(
            {
                "format_version": 1,
                "model": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "settings": normalize_settings(manifest),
            },
            temporary,
        )
        os.replace(temporary, directory / "model.pt")
    finally:
        temporary.unlink(missing_ok=True)
    (directory / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    (directory / "manifest.json").write_text(
        json.dumps(normalize_settings(manifest), indent=2) + "\n"
    )
    counters = {
        key: value
        for key, value in asdict(state).items()
        if not isinstance(value, torch.Tensor)
    }
    (directory / "training-state.json").write_text(
        json.dumps(counters, indent=2) + "\n"
    )
