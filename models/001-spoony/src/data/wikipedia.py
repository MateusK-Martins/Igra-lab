"""Materialize a pinned, deduplicated Wikipedia token slice. Never train a model."""

import hashlib
import json
import shutil
import sys
from array import array
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as HFTokenizer

from configs.data.wikipedia import (
    DATASET_ID,
    EXPECTED_TOKENIZER_SHA256,
    RAW_DIRECTORY,
    REVISION,
    SHARD_COUNT,
    SPLIT_SEED,
    SUBSET,
    TOKENIZER_ARTIFACT,
    TOKENIZER_DOCUMENTS,
    TRAIN_TOKENS,
    VALIDATION_TOKENS,
)
from configs.tokenizers.preparation import preparation_config
from src.data.tokenizer import ByteLevelBPETokenizer


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_for(digest: str) -> str:
    key = hashlib.sha256(f"{SPLIT_SEED}:{digest}".encode()).digest()
    return "validation" if int.from_bytes(key[:8], "big") % 100 == 0 else "train"


def articles():
    for shard in range(SHARD_COUNT):
        filename = f"train-{shard:05d}-of-{SHARD_COUNT:05d}.parquet"
        local = RAW_DIRECTORY / filename
        if not local.exists():
            print(f"Downloading {filename}", flush=True)
            cached = hf_hub_download(
                DATASET_ID,
                f"{SUBSET}/{filename}",
                repo_type="dataset",
                revision=REVISION,
                local_dir=RAW_DIRECTORY / "download",
            )
            Path(cached).replace(local)
        print(f"Reading {filename}", flush=True)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=256):
            yield from batch.to_pylist()


class SliceWriter:
    """Write the existing TokenStore format with an exact token cap."""

    def __init__(self, directory: Path, raw_path: Path, limit: int):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.limit = directory, limit
        self.tokens = (directory / "tokens.bin").open("xb")
        self.documents = (directory / "documents.jsonl").open("x")
        self.raw = raw_path.open("x", encoding="utf-8")
        self.count = self.documents_count = 0
        self.truncated_final_document = False

    def append(self, row, ids, digest):
        if self.count == self.limit:
            return
        remaining = self.limit - self.count
        selected = ids[:remaining]
        if not selected:
            return
        self.truncated_final_document = len(selected) < len(ids)
        values = array("I", selected)
        if sys.byteorder != "little":
            values.byteswap()
        values.tofile(self.tokens)
        end = self.count + len(selected)
        self.documents.write(
            json.dumps(
                {
                    "document_id": f"wikipedia/{row['id']}/{digest}",
                    "start": self.count,
                    "end": end,
                }
            )
            + "\n"
        )
        self.raw.write(
            json.dumps({**row, "content_sha256": digest}, ensure_ascii=False) + "\n"
        )
        self.count = end
        self.documents_count += 1

    def close(self, tokenizer_hash):
        self.tokens.close()
        self.documents.close()
        self.raw.close()
        (self.directory / "metadata.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "dtype": "uint32-le",
                    "document_count": self.documents_count,
                    "token_count": self.count,
                    "tokenizer_sha256": tokenizer_hash,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    output = preparation_config.output_directory
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        if (RAW_DIRECTORY / f"{split}.jsonl").exists():
            raise FileExistsError(f"Existing raw {split} split")
    destination = output
    output = destination.with_name(destination.name + ".preparing")
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "validation"):
        (RAW_DIRECTORY / f"{split}.jsonl.preparing").unlink(missing_ok=True)
    output.mkdir(parents=True)
    tokenizer_path = TOKENIZER_ARTIFACT
    if (
        hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        != EXPECTED_TOKENIZER_SHA256
    ):
        raise ValueError("Bundled Wikipedia tokenizer hash mismatch")
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    # Copy bytes rather than reserializing: identity must survive a fresh clone.
    (output / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    end_id = tokenizer.token_to_id(preparation_config.end_of_document_token)
    tokenizer_hash = hashlib.sha256(
        (output / "tokenizer.json").read_bytes()
    ).hexdigest()
    encoder = HFTokenizer.from_file(str(output / "tokenizer.json"))
    writers = {
        split: SliceWriter(
            output / split, RAW_DIRECTORY / f"{split}.jsonl.preparing", limit
        )
        for split, limit in (("train", TRAIN_TOKENS), ("validation", VALIDATION_TOKENS))
    }
    seen = set()
    iterator = iter(articles())
    duplicate_count = 0
    last_progress = 0
    try:
        while not all(w.count == w.limit for w in writers.values()):
            rows = list(islice(iterator, 256))
            if not rows:
                raise ValueError("Wikipedia exhausted before configured token budgets")
            selected = []
            for row in rows:
                text = row["text"].strip()
                digest = content_hash(text)
                if not text:
                    continue
                if digest in seen:
                    duplicate_count += 1
                    continue
                seen.add(digest)
                split = split_for(digest)
                if writers[split].count < writers[split].limit:
                    selected.append((row, digest, split))
            encoded = encoder.encode_batch(
                [r[0]["text"] for r in selected], add_special_tokens=False
            )
            for (row, digest, split), encoding in zip(selected, encoded, strict=True):
                writers[split].append(row, encoding.ids + [end_id], digest)
            if writers["train"].count - last_progress >= 10_000_000:
                last_progress = writers["train"].count
                print({key: w.count for key, w in writers.items()}, flush=True)
    finally:
        for writer in writers.values():
            writer.close(tokenizer_hash)
    report = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "subset": SUBSET,
        "selection": "source shard/row order, strip-text SHA256 deduplication; seed42 content-hash 1% validation; no repetition",
        "tokenizer_train_documents": TOKENIZER_DOCUMENTS,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer_hash,
        "duplicates_skipped": duplicate_count,
        "splits": {
            key: {
                "tokens": w.count,
                "articles": w.documents_count,
                "final_article_truncated": w.truncated_final_document,
            }
            for key, w in writers.items()
        },
        "token_budget_includes_end_of_document_tokens": True,
        "raw_final_article": "raw JSONL retains full article; token store truncates only final article to exact budget",
        "model_training_started": False,
    }
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for split in ("train", "validation"):
        (RAW_DIRECTORY / f"{split}.jsonl.preparing").replace(
            RAW_DIRECTORY / f"{split}.jsonl"
        )
    output.replace(destination)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
