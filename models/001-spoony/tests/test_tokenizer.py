"""Tests for Spoony's initial tokenizer."""

import pytest

from src.data.tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["hello", "Olá, mundo!", "🤖"])
def test_encode_decode_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_token_ids_are_utf8_bytes() -> None:
    tokenizer = ByteTokenizer()

    assert tokenizer.encode("Olá") == [79, 108, 195, 161]
    assert tokenizer.vocab_size == 256
