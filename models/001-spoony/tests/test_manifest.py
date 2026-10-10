import json
import subprocess
from pathlib import Path

import pytest

from src.config.snapshot import normalize_settings
from src.config.types import ChunkingConfig
from src.data.serialization import TokenStore
from src.training.manifest import build_manifest, file_sha256, save_manifest


@pytest.fixture
def prepared_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for args in (
        ["init", "-q"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(project), *args], check=True, capture_output=True
        )
    for name in ("src/model.py", "configs/model/model.py", "scripts/__init__.py"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")
    data = project / "data"
    data.mkdir()
    tokenizer = data / "tokenizer.json"
    tokenizer.write_text('{"fixture": 1}', encoding="utf-8")
    for split in ("train", "validation"):
        TokenStore(data / split).write_documents(
            [("doc", [1, 2, 3, 4, 5])], tokenizer_sha256=file_sha256(tokenizer)
        )
    return {
        "project_directory": project,
        "data_directory": data,
        "settings": {"chunking": ChunkingConfig(2, 2), "device": "cpu"},
    }


def test_manifest_is_stable_json_and_records_settings_artifacts_and_uncommitted_code(
    prepared_project,
):
    manifest = build_manifest(**prepared_project)
    assert manifest == build_manifest(**prepared_project)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["settings"]["chunking"] == {"context_length": 2, "stride": 2}
    assert len(manifest["artifacts"]) == 7
    assert len(manifest["code"]["git_commit"]) == 40
    assert sorted(manifest["code"]["files"]) == [
        "configs/model/model.py",
        "scripts/__init__.py",
        "src/model.py",
    ]
    assert manifest["environment"]["torch"]
    for name, digest in manifest["artifacts"].items():
        assert digest == file_sha256(prepared_project["data_directory"] / name)


@pytest.mark.parametrize(
    "name", ["src/model.py", "configs/model/model.py", "scripts/__init__.py"]
)
def test_source_edits_change_manifest_without_a_new_commit(prepared_project, name):
    original = build_manifest(**prepared_project)
    (prepared_project["project_directory"] / name).write_text("# edited\n")
    changed = build_manifest(**prepared_project)
    assert changed["code"]["git_commit"] == original["code"]["git_commit"]
    assert changed != original


def test_same_size_token_edit_changes_manifest(prepared_project):
    original = build_manifest(**prepared_project)
    path = prepared_project["data_directory"] / "train/tokens.bin"
    data = bytearray(path.read_bytes())
    data[0] = 2
    path.write_bytes(data)
    assert build_manifest(**prepared_project)["artifacts"] != original["artifacts"]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_tokenizer_mismatch_rejected(prepared_project, split):
    path = prepared_project["data_directory"] / split / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["tokenizer_sha256"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"{split} data does not match"):
        build_manifest(**prepared_project)


def test_corrupt_document_offsets_rejected(prepared_project):
    path = prepared_project["data_directory"] / "train/documents.jsonl"
    path.write_text('{"document_id": "doc", "start": 1, "end": 5}\n')
    with pytest.raises(ValueError, match="offsets"):
        build_manifest(**prepared_project)


def test_missing_artifact_fails(prepared_project):
    (prepared_project["data_directory"] / "validation/tokens.bin").unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(**prepared_project)


def test_save_roundtrip_and_matching_existing_manifest(prepared_project, tmp_path):
    manifest = build_manifest(**prepared_project)
    path = tmp_path / "run/manifest.json"
    save_manifest(manifest, path)
    original = path.read_bytes()
    save_manifest(manifest, path)
    assert path.read_bytes() == original
    assert json.loads(original) == manifest


def test_save_rejects_different_run_without_overwriting(tmp_path):
    path = tmp_path / "manifest.json"
    save_manifest({"settings": {"lr": 0.1}}, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="different run"):
        save_manifest({"settings": {"lr": 0.2}}, path)
    assert path.read_bytes() == original


def test_normalize_supports_dataclasses_paths_and_tuples():
    assert normalize_settings(
        {"chunking": ChunkingConfig(2, 1), "path": Path("data"), "betas": (0.9, 0.95)}
    ) == {
        "chunking": {"context_length": 2, "stride": 1},
        "path": "data",
        "betas": [0.9, 0.95],
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_normalize_rejects_values_unsuitable_for_checkpoint_comparison(value):
    with pytest.raises((ValueError, TypeError)):
        normalize_settings({"value": value})


def test_checkpoint_rejects_changed_data_identified_by_manifest(
    prepared_project, tmp_path
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.training.checkpoint import load_checkpoint, save_checkpoint
    from src.training.state import TrainingState

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 2)), generator=torch.Generator().manual_seed(42)
    )
    components = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": torch.amp.GradScaler("cpu", enabled=False),
        "train_loader": loader,
        "validation_loader": loader,
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        **components,
        state=TrainingState(),
        settings=build_manifest(**prepared_project),
    )
    tokens = prepared_project["data_directory"] / "train/tokens.bin"
    data = bytearray(tokens.read_bytes())
    data[0] = 2
    tokens.write_bytes(data)
    with pytest.raises(ValueError, match="settings do not match"):
        load_checkpoint(path, **components, settings=build_manifest(**prepared_project))
