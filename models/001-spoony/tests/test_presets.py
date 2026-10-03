"""Tests for named experiment configurations."""

from src.config.presets import tinystories_v2


def test_tinystories_preset_has_expected_data_settings() -> None:
    config = tinystories_v2()

    assert config.tokenizer.target_vocab_size == 4096
    assert config.tokenizer.special_tokens == ("<|unk|>", "<|eot|>")
    assert config.chunking.context_length == 128
    assert config.chunking.stride == 128
