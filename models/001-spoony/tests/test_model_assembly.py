import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.model.assembler import Model
from src.model.definitions import Embedding, LMHead, Repeat, Sequential, SwiGLU


def definition(tied=True):
    return Model(
        tokenizer=SimpleNamespace(vocab_size=13),
        input=Embedding(4),
        blocks=[Repeat(2, [Sequential([SwiGLU(4, 8, 4, residual=True)])])],
        lm_head=LMHead(4, tied),
    )


@pytest.mark.parametrize("tied", [False, True])
def test_assembled_model_forward_backward_registration_and_tying(tied) -> None:
    model, entries = definition(tied).assemble()
    assert entries == []
    logits = model(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert logits.shape == (2, 3, 13)
    assert (model.lm_head.weight is model.embedding.weight) == tied
    assert model.body[0].branch.gate.weight is not model.body[1].branch.gate.weight
    logits.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(name.startswith("body.") for name in model.state_dict())


@pytest.mark.parametrize("tied", [False, True])
def test_state_dict_round_trip_preserves_outputs_and_tying(tied) -> None:
    recipe = definition(tied)
    model, _ = recipe.assemble()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored, _ = recipe.assemble()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    tokens = torch.tensor([[1, 2]])
    torch.testing.assert_close(restored(tokens), model(tokens))
    assert (restored.lm_head.weight is restored.embedding.weight) == tied


def test_empty_body_and_revalidation_before_build() -> None:
    recipe = Model(SimpleNamespace(vocab_size=7), Embedding(4), [], LMHead(4))
    model, entries = recipe.assemble()
    assert entries == []
    assert model(torch.tensor([[1, 2]])).shape == (1, 2, 7)
    recipe.blocks.append(SwiGLU(8, 16, 4))
    with patch.object(Embedding, "build") as build:
        with pytest.raises(ValueError, match="previous component"):
            recipe.assemble()
        build.assert_not_called()


def test_assembly_collects_cache_entries_for_each_attention_occurrence() -> None:
    from src.model.attention import GQAAttentionImpl
    from src.model.cache_storage import CacheStorage
    from src.model.definitions import GQAAttention

    attention = GQAAttention(4, 4, 2, 1, 2, 0, 0.0)
    recipe = Model(
        SimpleNamespace(vocab_size=13),
        Embedding(4),
        [Repeat(2, [Sequential([attention])]), attention],
        LMHead(4),
    )
    model, entries = recipe.assemble()
    layers = [m for m in model.modules() if isinstance(m, GQAAttentionImpl)]
    assert [e.layer_id for e in entries] == [m.layer_id for m in layers]
    assert len({e.layer_id for e in entries}) == 3
    assert len({id(e.keys) for e in entries}) == 3
    assert all(e.keys.shape == (0, 1, 0, 2) for e in entries)
    storage = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    keys = torch.ones(2, 1, 3, 2, dtype=torch.float16)
    storage.write(entries[0].layer_id, keys, keys + 1)
    restored = storage.read_all(entries[0].layer_id)
    torch.testing.assert_close(restored.keys, keys)
    torch.testing.assert_close(restored.values, keys + 1)
    assert restored.count == 3
    assert storage.step(entries[1].layer_id) == 0
