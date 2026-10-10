"""Identify prepared artifacts, resolved settings, and code for a training run."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from src.config.snapshot import normalize_settings
from src.data.serialization import TokenStore


def file_sha256(path: Path) -> str:
    """Hash in chunks so token stores do not need to fit in RAM."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_manifest(
    *,
    project_directory: Path,
    data_directory: Path,
    settings: dict[str, object],
) -> dict:
    """Build stable checkpoint settings; no timestamps or resume-path fields.

    Supply resolved optimizer, scheduler, precision, training, chunking, loader
    and device settings. Model configuration and runtime are identified by
    source hashes, including uncommitted edits. Files must remain unchanged
    while training. Git and the prepared train/validation artifacts are required.
    """
    resolved_settings = normalize_settings(settings)
    tokenizer_hash = file_sha256(data_directory / "tokenizer.json")
    artifacts = {"tokenizer.json": tokenizer_hash}
    for split in ("train", "validation"):
        store = TokenStore(data_directory / split)
        store.read_summary()
        metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError(f"{split} data does not match the loaded tokenizer")
        # Validate document offsets as well as the token count/file size.
        for _ in store.iter_document_offsets():
            pass
        for path in (store.tokens_path, store.documents_path, store.metadata_path):
            artifacts[path.relative_to(data_directory).as_posix()] = file_sha256(path)

    return build_artifact_manifest(
        project_directory=project_directory,
        settings=resolved_settings,
        artifacts=artifacts,
    )


def build_artifact_manifest(
    *, project_directory: Path, settings: dict[str, object], artifacts: dict[str, str]
) -> dict:
    """Record explicit artifact hashes, settings, code and environment for a run."""
    resolved_settings = normalize_settings(settings)
    commit = subprocess.run(
        ["git", "-C", str(project_directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = sorted(
        path
        for directory in ("src", "configs/model", "scripts")
        for path in (project_directory / directory).rglob("*.py")
    )
    return {
        "format_version": 1,
        "settings": resolved_settings,
        "artifacts": artifacts,
        "code": {
            "git_commit": commit,
            "files": {
                path.relative_to(project_directory).as_posix(): file_sha256(path)
                for path in sources
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "machine": platform.machine(),
            "cuda_devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
    }


def save_manifest(manifest: dict, path: Path) -> None:
    """Keep an existing matching manifest; never overwrite a different run."""
    manifest = normalize_settings(manifest)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing manifest belongs to a different run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
