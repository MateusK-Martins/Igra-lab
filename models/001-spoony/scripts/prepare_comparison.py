"""Prepare train-only shared BPE, full stories and length-filtered SQuAD."""

import hashlib
import json
import random
from itertools import chain
from urllib.request import urlopen

import pyarrow.parquet as pq

from configs.data.corpus import RAW_DIRECTORY as STORY_RAW_DIRECTORY
from configs.data.datasets import dataset_pipe
from configs.data.qa_corpus import (
    DATASET_ID,
    QA_LIMITS,
    RAW_DIRECTORY,
    REVISION,
    SELECTION_SEED,
    SOURCE_FILES,
    qa_config,
)
from configs.tokenizers.preparation import preparation_config
from src.data.prepare import DataProcessor
from src.data.tokenizer import ByteLevelBPETokenizer


def rows(split):
    for batch in pq.ParquetFile(RAW_DIRECTORY / f"{split}.parquet").iter_batches(
        batch_size=2048
    ):
        yield from batch.to_pylist()


def instruction(row):
    return row["context"] + "\nQuestion: " + row["question"]


def eligible(row, tokenizer, config):
    answers = row["answers"]["text"]
    if not answers or not answers[0].strip():
        return False
    prompt = tokenizer.encode(
        config.instruction_prefix + instruction(row) + config.response_prefix
    )
    response = tokenizer.encode(answers[0]) + [tokenizer.token_to_id(config.end_token)]
    return len(prompt) + len(response) - 1 <= config.context_length


def main():
    output = preparation_config.output_directory
    if output.exists():
        raise FileExistsError(output)
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for split, file in SOURCE_FILES.items():
        path = RAW_DIRECTORY / f"{split}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{file}"
            with urlopen(url, timeout=60) as source, path.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            print("downloaded SQuAD", split, flush=True)
    texts = chain(
        (doc.text for doc in dataset_pipe.documents("train")),
        (
            qa_config.instruction_prefix
            + instruction(row)
            + qa_config.response_prefix
            + row["answers"]["text"][0]
            for row in rows("train")
            if row["answers"]["text"]
        ),
    )
    settings = preparation_config.tokenizer
    tokenizer = ByteLevelBPETokenizer.train(
        texts,
        target_vocab_size=settings.target_vocab_size,
        min_frequency=settings.min_frequency,
        special_tokens=settings.special_tokens,
    )
    processor = DataProcessor(
        tokenizer,
        dataset_pipe,
        output,
        end_of_document_token=preparation_config.end_of_document_token,
    )
    story_summaries = {
        split: processor.process_split(split).__dict__
        for split in ("train", "validation")
    }
    train_contexts = {
        hashlib.sha256(row["context"].encode()).hexdigest() for row in rows("train")
    }
    report = {
        "story_source": json.loads((STORY_RAW_DIRECTORY / "subset.json").read_text()),
        "qa_dataset": DATASET_ID,
        "qa_revision": REVISION,
        "qa_license": "CC-BY-SA-4.0",
        "tokenizer_training": "all selected training stories plus official SQuAD training texts only; no validation",
        "selection": "seed-42 shuffle of each official QA split followed by length/dedup filtering; full contexts retained",
        "selection_seed": SELECTION_SEED,
        "stories": story_summaries,
        "qa": {},
    }
    for split, limit in QA_LIMITS.items():
        seen = set()
        count = skipped = 0
        path = output / f"qa-{split}.jsonl"
        with path.open("x", encoding="utf-8") as destination:
            split_rows = list(rows(split))
            random.Random(SELECTION_SEED).shuffle(split_rows)
            for row in split_rows:
                identity = (row["context"], row["question"])
                overlap = (
                    split == "validation"
                    and hashlib.sha256(row["context"].encode()).hexdigest()
                    in train_contexts
                )
                if (
                    identity in seen
                    or overlap
                    or not eligible(row, tokenizer, qa_config)
                ):
                    skipped += 1
                    continue
                seen.add(identity)
                record = {
                    "id": row["id"],
                    "instruction": instruction(row),
                    "question": row["question"],
                    "response": row["answers"]["text"][0],
                    "answers": row["answers"]["text"],
                }
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count == limit:
                    break
        if count != limit:
            raise ValueError(f"Only {count}/{limit} eligible {split} QA records")
        report["qa"][split] = {
            "examples": count,
            "skipped_before_limit": skipped,
            "raw_sha256": hashlib.sha256(
                (RAW_DIRECTORY / f"{split}.parquet").read_bytes()
            ).hexdigest(),
            "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print("prepared QA", split, count, flush=True)
    (output / "preparation-report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
