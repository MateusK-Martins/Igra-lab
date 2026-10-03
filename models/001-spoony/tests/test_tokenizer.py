"""Tests for the byte-level BPE tokenizer."""

import pytest

from src.data.tokenizer import ByteLevelBPETokenizer


@pytest.fixture
def tokenizer() -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer.train(
        [
            "A model learns language from repeated patterns.",
            "Olá, mundo! Um modelo aprende padrões de linguagem.",
            "🤖 model language model language model",
        ],
        target_vocab_size=320,
        min_frequency=1,
        special_tokens=["<|unk|>", "<|eot|>"],
    )


@pytest.mark.parametrize("text", ["hello", "Olá, mundo!", "🤖"])
def test_encode_decode_round_trip(
    tokenizer: ByteLevelBPETokenizer,
    text: str,
) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_token_has_a_stable_id(tokenizer: ByteLevelBPETokenizer) -> None:
    assert tokenizer.token_to_id("<|eot|>") == 1


def test_save_and_load_preserves_encoding(
    tokenizer: ByteLevelBPETokenizer,
    tmp_path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    restored = ByteLevelBPETokenizer.load(tokenizer_path)

    assert restored.encode("Spoony aprende") == tokenizer.encode("Spoony aprende")
    assert restored.vocab_size == tokenizer.vocab_size
