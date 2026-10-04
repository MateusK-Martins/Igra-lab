"""Disk storage contracts for a single processed split.

tokens.bin uses little-endian unsigned 32-bit IDs. documents.jsonl records
document_id, start, and end; offsets count tokens, and end is exclusive.
metadata.json records format version, dtype, document count, and token count.
Writers must reject existing output files rather than silently overwrite them.
"""

import json
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentOffset:
    document_id: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenStoreSummary:
    document_count: int
    token_count: int


class TokenStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.tokens_path = self.directory / "tokens.bin"
        self.documents_path = self.directory / "documents.jsonl"
        self.metadata_path = self.directory / "metadata.json"

    def write_documents(
        self,
        documents: Iterable[tuple[str, Sequence[int]]],
        *,
        tokenizer_sha256: str,
    ) -> TokenStoreSummary:

        self.directory.mkdir(parents=True, exist_ok=True)

        if (
            self.tokens_path.exists()
            or self.documents_path.exists()
            or self.metadata_path.exists()
        ):
            raise FileExistsError(
                "There are training setup data currently active, unable to save current data"
            )

        document_count = 0
        token_count = 0

        with (
            self.tokens_path.open("xb") as tokens_file,
            self.documents_path.open("x", encoding="utf-8") as documents_file,
        ):
            for document_id, token_ids in documents:
                start = token_count
                end = start + len(token_ids)

                for token_id in token_ids:
                    if type(token_id) is not int:
                        raise TypeError("Token ID must be an integer")
                    if not 0 <= token_id <= 2**32 - 1:
                        raise ValueError(
                            "Token ID must fit in an unsigned 32-bit integer"
                        )

                    tokens_file.write(struct.pack("<I", token_id))

                    token_count += 1

                record = {
                    "document_id": document_id,
                    "start": start,
                    "end": end,
                }

                documents_file.write(json.dumps(record) + "\n")

                document_count += 1

        with self.metadata_path.open("x", encoding="utf-8") as metadata_file:
            metadata = {
                "format_version": 1,
                "dtype": "uint32-le",
                "document_count": document_count,
                "token_count": token_count,
                "tokenizer_sha256": tokenizer_sha256,
            }

            metadata_file.write(json.dumps(metadata, indent=2))

        return TokenStoreSummary(document_count=document_count, token_count=token_count)

    def read_summary(self) -> TokenStoreSummary:
        """Load metadata and check format version and token-file size."""

        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

            if metadata["format_version"] != 1:
                raise ValueError("Unsupported token storage format")

            if metadata["dtype"] != "uint32-le":
                raise ValueError("Unsupported token dtype")

            document_count = metadata["document_count"]
            token_count = metadata["token_count"]
            for count in (document_count, token_count):
                if type(count) is not int or count < 0:
                    raise ValueError("Stored counts must be non-negative integers")

            expected_bytes = token_count * 4
            actual_bytes = self.tokens_path.stat().st_size

            if actual_bytes != expected_bytes:
                raise ValueError("Token file size does not match metadata")

        return TokenStoreSummary(
            document_count=document_count,
            token_count=token_count,
        )

    def read_tokens(self, start: int, end: int) -> list[int]:
        """Read [start, end) without loading the entire file.

        Require 0 <= start <= end <= token_count; byte offset is start * 4.
        """
        summary = self.read_summary()
        if type(start) is not int or type(end) is not int:
            raise TypeError("Token offsets must be integers")
        if not 0 <= start <= end <= summary.token_count:
            raise ValueError("Token range is out of bounds")

        expected_bytes = (end - start) * 4
        with self.tokens_path.open("rb") as file:
            file.seek(start * 4)
            data = file.read(expected_bytes)
        if len(data) != expected_bytes:
            raise ValueError("Token file ended before the requested range")
        return [token_id for (token_id,) in struct.iter_unpack("<I", data)]

    def iter_document_offsets(self) -> Iterator[DocumentOffset]:
        """Read documents.jsonl one line at a time, in stored order."""
        summary = self.read_summary()
        previous_end = 0
        document_count = 0
        with self.documents_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                    document_id = record["document_id"]
                    start = record["start"]
                    end = record["end"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"Invalid document offset record on line {line_number}"
                    ) from error
                if not isinstance(document_id, str) or not document_id.strip():
                    raise ValueError(f"Invalid document ID on line {line_number}")
                if type(start) is not int or type(end) is not int:
                    raise ValueError(f"Invalid offsets on line {line_number}")
                if start != previous_end or not start <= end <= summary.token_count:
                    raise ValueError(f"Inconsistent offsets on line {line_number}")
                document_count += 1
                if document_count > summary.document_count:
                    raise ValueError("Document count does not match metadata")
                previous_end = end
                yield DocumentOffset(document_id, start, end)

        if document_count != summary.document_count:
            raise ValueError("Document count does not match metadata")
        if previous_end != summary.token_count:
            raise ValueError("Document offsets do not cover all stored tokens")
