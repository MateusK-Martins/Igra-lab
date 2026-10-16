"""Expand the already materialized, pinned SQuAD corpus; keep the pretrained BPE."""

import hashlib
import json
import random

import pyarrow.parquet as pq

from configs.sft.data import (
    OUTPUT_DIRECTORY,
    RAW_DIRECTORY,
    REVISION,
    SELECTION_SEED,
    VALIDATION_LIMIT,
    data_config,
)
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.manifest import file_sha256


def main():
    tokenizer = ByteLevelBPETokenizer.load(data_config.tokenizer_path)
    train_rows = pq.read_table(RAW_DIRECTORY / "train.parquet").to_pylist()
    contexts = {hashlib.sha256(row["context"].encode()).digest() for row in train_rows}
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "rajpurkar/squad",
        "revision": REVISION,
        "seed": SELECTION_SEED,
        "selection": "all eligible training records; seeded shuffle; full passages; original 1000 validation questions",
        "splits": {},
    }
    for split in ("train", "validation"):
        rows = (
            train_rows
            if split == "train"
            else pq.read_table(RAW_DIRECTORY / "validation.parquet").to_pylist()
        )
        random.Random(SELECTION_SEED).shuffle(rows)
        seen = set()
        kept = skipped = 0
        path = OUTPUT_DIRECTORY / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                answers = row["answers"]["text"]
                identity = (row["context"], row["question"])
                instruction = row["context"] + "\nQuestion: " + row["question"]
                prompt = tokenizer.encode(
                    data_config.instruction_prefix
                    + instruction
                    + data_config.response_prefix
                )
                overlap = (
                    split == "validation"
                    and hashlib.sha256(row["context"].encode()).digest() in contexts
                )
                if (
                    not answers
                    or not answers[0].strip()
                    or identity in seen
                    or overlap
                    or len(prompt) + len(tokenizer.encode(answers[0]))
                    > data_config.context_length
                ):
                    skipped += 1
                    continue
                seen.add(identity)
                record = {
                    "id": row["id"],
                    "instruction": instruction,
                    "question": row["question"],
                    "response": answers[0],
                    "answers": answers,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                if split == "validation" and kept == VALIDATION_LIMIT:
                    break
        report["splits"][split] = {
            "raw_examples": len(rows),
            "examples": kept,
            "skipped": skipped,
            "raw_sha256": file_sha256(RAW_DIRECTORY / f"{split}.parquet"),
            "prepared_sha256": file_sha256(path),
        }
    (OUTPUT_DIRECTORY / "preparation-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
