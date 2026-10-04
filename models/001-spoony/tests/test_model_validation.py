from types import SimpleNamespace

import pytest

from src.model.assembler import Model
from src.model.definitions import Embedding, Linear, LMHead, Repeat, Sequential


def make_model(**overrides):
    settings = {
        "tokenizer": SimpleNamespace(vocab_size=10),
        "input": Embedding(4),
        "blocks": [Linear(4, 8), Linear(8, 4)],
        "lm_head": LMHead(4, True),
    }
    settings.update(overrides)
    return Model(**settings)


def test_valid_width_changes_and_empty_body() -> None:
    make_model().validate()
    make_model(blocks=[]).validate()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
@pytest.mark.parametrize(
    "component,field",
    [
        ("tokenizer", "vocab_size"),
        ("input", "output_features"),
        ("lm_head", "input_features"),
    ],
)
def test_rejects_invalid_boundary_values(component, field, value) -> None:
    attributes = {field: value}
    if component == "lm_head":
        attributes["tie_to_embedding"] = False
    with pytest.raises(ValueError, match=f"{component}.{field}"):
        make_model(**{component: SimpleNamespace(**attributes)})


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_features", 0),
        ("output_features", True),
        ("residual", 1),
    ],
)
def test_rejects_invalid_custom_block_values(field, value) -> None:
    attributes = {"input_features": 4, "output_features": 4, "residual": False}
    attributes[field] = value
    with pytest.raises(ValueError, match=field):
        make_model(blocks=[SimpleNamespace(**attributes)])


def test_rejects_non_boolean_tying() -> None:
    with pytest.raises(ValueError, match="tie_to_embedding"):
        make_model(lm_head=SimpleNamespace(input_features=4, tie_to_embedding=1))


def test_rejects_connection_residual_and_head_mismatches() -> None:
    with pytest.raises(ValueError, match="previous component"):
        make_model(blocks=[Linear(8, 4)])
    with pytest.raises(ValueError, match="residual"):
        make_model(blocks=[Linear(4, 8, True)])
    with pytest.raises(ValueError, match="body output"):
        make_model(lm_head=LMHead(8))
    with pytest.raises(ValueError, match="Tied"):
        make_model(blocks=[Linear(4, 8)], lm_head=LMHead(8, True))


def test_revalidation_catches_mutated_block_list() -> None:
    model = make_model()
    model.blocks.append(Linear(8, 4))
    with pytest.raises(ValueError, match="previous component"):
        model.validate()


def test_valid_nested_repeats_and_single_width_changing_repeat() -> None:
    make_model(
        blocks=[Repeat(3, [Linear(4, 8), Repeat(2, [Linear(8, 8)]), Linear(8, 4)])]
    )
    make_model(blocks=[Repeat(1, [Linear(4, 8)])], lm_head=LMHead(8))


@pytest.mark.parametrize("times", [0, -1, True, 1.5])
def test_repeat_rejects_invalid_counts(times) -> None:
    with pytest.raises(ValueError, match="times"):
        make_model(blocks=[Repeat(times, [Linear(4, 4)])])


@pytest.mark.parametrize("block", [Repeat(1, []), Repeat(2, [Repeat(1, [])])])
def test_repeat_rejects_empty_children_before_property_access(block) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_model(blocks=[block])


def test_repeat_checks_internal_neighbors_and_repetition_boundary() -> None:
    with pytest.raises(ValueError, match=r"blocks\[0\].blocks\[1\]"):
        make_model(blocks=[Repeat(2, [Linear(4, 8), Linear(4, 4)])])
    with pytest.raises(ValueError, match="repeat output"):
        make_model(blocks=[Repeat(2, [Linear(4, 8)])], lm_head=LMHead(8))


def test_repeat_checks_its_residual_and_revalidates_mutated_children() -> None:
    with pytest.raises(ValueError, match="residual"):
        make_model(blocks=[Repeat(1, [Linear(4, 8)], True)], lm_head=LMHead(8))
    with pytest.raises(ValueError, match="boolean"):
        make_model(blocks=[Repeat(1, [Linear(4, 4)], residual=1)])
    repeat = Repeat(2, [Linear(4, 4)])
    model = make_model(blocks=[repeat])
    repeat.blocks.clear()
    with pytest.raises(ValueError, match="must not be empty"):
        model.validate()


def test_sequential_validates_nested_connections_and_residual() -> None:
    make_model(
        blocks=[
            Sequential([Linear(4, 8), Repeat(2, [Linear(8, 8)]), Linear(8, 4)], True)
        ]
    )
    with pytest.raises(ValueError, match=r"blocks\[0\].blocks\[1\]"):
        make_model(blocks=[Sequential([Linear(4, 8), Linear(4, 4)])])
    with pytest.raises(ValueError, match="residual"):
        make_model(blocks=[Sequential([Linear(4, 8)], True)], lm_head=LMHead(8))


@pytest.mark.parametrize(
    "block", [Sequential([]), Repeat(2, [Sequential([])]), Sequential([Repeat(1, [])])]
)
def test_empty_nested_compositions_are_rejected(block) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_model(blocks=[block])
