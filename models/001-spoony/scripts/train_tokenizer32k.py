"""Train the 32K byte-level BPE tokenizer from a local training-only slice."""

import json

from configs.tokenizers.corpus import (
    corpus_path,
    document_limit,
    output_path,
    report_path,
)
from configs.tokenizers.preparation import preparation_config
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.manifest import file_sha256


def main():
    if output_path.exists():
        raise FileExistsError(output_path)

    def texts():
        with corpus_path.open() as file:
            for index, line in enumerate(file):
                if index == document_limit:
                    break
                yield json.loads(line)["text"]

    config = preparation_config.tokenizer
    tokenizer = ByteLevelBPETokenizer.train(
        texts(),
        target_vocab_size=config.target_vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
    )
    assert tokenizer.vocab_size == 32768
    tokenizer.save(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "wikimedia/wikipedia",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "subset": "20231101.en",
        "selection": "first 20000 documents from the deduplicated training split of the 600M corpus; no validation text",
        "documents": document_limit,
        "corpus_sha256": file_sha256(corpus_path),
        "vocabulary": tokenizer.vocab_size,
        "tokenizer_sha256": file_sha256(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
