"""Tests for layered model-configuration merging."""

import pytest

from src.config.errors import ConfigError
from src.config.merge import merge_configs


def test_rejects_merge_without_configs() -> None:
    with pytest.raises(ConfigError, match="Not enough arguments"):
        merge_configs()


def test_returns_deep_copy_for_one_config() -> None:
    config = {"tokenizer": {"special_tokens": ["<|unk|>", "<|eot|>"]}}

    merged = merge_configs(config)
    merged["tokenizer"]["special_tokens"].append("<|pad|>")

    assert config == {"tokenizer": {"special_tokens": ["<|unk|>", "<|eot|>"]}}


def test_later_scalar_value_overrides_earlier_value() -> None:
    merged = merge_configs(
        {"training": {"learning_rate": 0.0003}},
        {"training": {"learning_rate": 0.0001}},
    )

    assert merged["training"]["learning_rate"] == 0.0001


def test_merges_nested_dictionaries() -> None:
    merged = merge_configs(
        {
            "training": {
                "learning_rate": 0.0003,
                "optimizer": {"weight_decay": 0.1},
            }
        },
        {"training": {"optimizer": {"beta_1": 0.9}}},
    )

    assert merged == {
        "training": {
            "learning_rate": 0.0003,
            "optimizer": {"weight_decay": 0.1, "beta_1": 0.9},
        }
    }


def test_later_list_replaces_and_does_not_share_override_list() -> None:
    override = {"tokenizer": {"special_tokens": ["<|unk|>", "<|eot|>"]}}

    merged = merge_configs(
        {"tokenizer": {"special_tokens": ["<|unk|>"]}},
        override,
    )
    merged["tokenizer"]["special_tokens"].append("<|pad|>")

    assert override == {"tokenizer": {"special_tokens": ["<|unk|>", "<|eot|>"]}}


def test_new_key_does_not_share_override_data() -> None:
    override = {"logging": {"tags": ["pretraining"]}}

    merged = merge_configs({"training": {"learning_rate": 0.0003}}, override)
    merged["logging"]["tags"].append("local")

    assert override == {"logging": {"tags": ["pretraining"]}}


def test_later_configs_win_in_order() -> None:
    merged = merge_configs(
        {"training": {"micro_batch_size": 4}},
        {"training": {"micro_batch_size": 8}},
        {"training": {"micro_batch_size": 16}},
    )

    assert merged["training"]["micro_batch_size"] == 16


@pytest.mark.parametrize(
    ("base", "override"),
    [
        ({"training": {"learning_rate": 0.0003}}, {"training": "disabled"}),
        ({"training": "disabled"}, {"training": {"learning_rate": 0.0003}}),
    ],
)
def test_rejects_table_and_scalar_conflicts(
    base: dict[str, object],
    override: dict[str, object],
) -> None:
    with pytest.raises(ConfigError, match="Type mismatch"):
        merge_configs(base, override)
