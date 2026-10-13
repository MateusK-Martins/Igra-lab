"""Download bounded, revision-pinned story prefixes and reject exact duplicates."""

import hashlib
import json
from urllib.request import urlopen

from configs.data.corpus import (
    DATASET_ID,
    DOCUMENT_LIMITS,
    RAW_DIRECTORY,
    REVISION,
    SOURCE_FILES,
)


def main():
    if RAW_DIRECTORY.exists():
        raise FileExistsError(f"Refusing to overwrite {RAW_DIRECTORY}")
    RAW_DIRECTORY.mkdir(parents=True)
    seen = set()
    report = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "selection": "first unique nonempty documents in source order",
        "splits": {},
    }
    for split, limit in DOCUMENT_LIMITS.items():
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{SOURCE_FILES[split]}"
        path = RAW_DIRECTORY / f"{split}.jsonl"
        count = skipped = 0
        lines = []
        with (
            urlopen(url, timeout=60) as response,
            path.open("w", encoding="utf-8") as output,
        ):
            for raw in response:
                line = raw.decode("utf-8")
                if line.strip() != "<|endoftext|>":
                    lines.append(line)
                    continue
                text = "".join(lines).strip()
                lines = []
                digest = hashlib.sha256(text.encode()).hexdigest()
                if not text or digest in seen:
                    skipped += 1
                    continue
                seen.add(digest)
                output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
                if count == limit:
                    break
        if count != limit:
            raise ValueError(f"Only obtained {count}/{limit} {split} documents")
        report["splits"][split] = {
            "documents": count,
            "duplicates_or_empty_skipped": skipped,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_url": url,
        }
        print(split, count, flush=True)
    (RAW_DIRECTORY / "subset.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
