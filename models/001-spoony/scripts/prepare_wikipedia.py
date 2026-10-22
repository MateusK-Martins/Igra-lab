"""Prepare and pin the corpus with the frozen 32K tokenizer; never train a model."""

import json

from configs.tokenizers.preparation import preparation_config
from src.config.paths import project_root
from src.data.wikipedia import main as prepare
from src.training.manifest import file_sha256


def main():
    prepare()
    root = preparation_config.output_directory
    artifacts = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "preparation-report.json"
    }
    (project_root() / "configs/data/wikipedia-vocab32k-artifacts.json").write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
