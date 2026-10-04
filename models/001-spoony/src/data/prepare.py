"""Coordinate document encoding and processed-data storage."""

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.types import TokenizerConfig
from src.data.pipeline import DataSetPipe
from src.data.schemas import Split
from src.data.serialization import TokenStore, TokenStoreSummary
from src.data.tokenizer import ByteLevelBPETokenizer, Tokenizer


def train_tokenizer(
    dataset_pipe: DataSetPipe, config: TokenizerConfig
) -> ByteLevelBPETokenizer:
    """Train BPE from a fresh traversal of train documents only."""
    return ByteLevelBPETokenizer.train(
        (document.text for document in dataset_pipe.documents("train")),
        target_vocab_size=config.target_vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
    )


class DataProcessor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_pipe: DataSetPipe,
        output_directory: Path,
        *,
        end_of_document_token: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_pipe = dataset_pipe
        self.output_directory = output_directory
        self.end_token_id = tokenizer.token_to_id(end_of_document_token)

    def _encoded_documents(self, split: Split) -> Iterator[tuple[str, Sequence[int]]]:
        """Encode each document, append the end token, yield its ID and tokens."""

        for document in self.dataset_pipe.documents(split):
            token_ids = self.tokenizer.encode(document.text)
            token_ids.append(self.end_token_id)

            yield document.document_id, token_ids

    def process_split(self, split: Split) -> TokenStoreSummary:
        """Save tokenizer.json, then stream encoded documents to a TokenStore.

        Store each split under output_directory / split. Save the tokenizer
        once and require subsequent splits to use the identical saved artifact.
        Record its SHA-256 in split metadata to associate tokens with vocabulary.
        """
        if split not in ("train", "validation"):
            raise ValueError("Split must be 'train' or 'validation'")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        tokenizer_path = self.output_directory / "tokenizer.json"

        if tokenizer_path.exists():
            with TemporaryDirectory() as temporary_directory:
                candidate_path = Path(temporary_directory) / "tokenizer.json"
                self.tokenizer.save(candidate_path)

                if candidate_path.read_bytes() != tokenizer_path.read_bytes():
                    raise ValueError("Output directory contains a different tokenizer")
        else:
            self.tokenizer.save(tokenizer_path)

        tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

        store = TokenStore(self.output_directory / split)

        return store.write_documents(
            self._encoded_documents(split), tokenizer_sha256=tokenizer_sha256
        )
