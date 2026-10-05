from functools import partial
from types import SimpleNamespace

import pytest

from src.model.assembler import Model
from src.model.attention import GQAAttentionImpl
from src.model.composition import ResidualImpl, build_block
from src.model.definitions import Embedding, GQAAttention, Linear, LMHead, Repeat
from src.model.position import rope


@pytest.mark.parametrize("residual", [False, True])
def test_repeated_attention_ids_match_cache_entries(residual):
    attention = GQAAttention(
        4,
        4,
        2,
        1,
        2,
        0,
        0.0,
        position_rotation=partial(rope, base=10000.0),
        residual=residual,
    )
    recipe = Model(
        SimpleNamespace(vocab_size=7),
        Embedding(4),
        [Repeat(2, [attention]), attention],
        LMHead(4),
    )
    model, entries = recipe.assemble()
    branches = [m.branch if isinstance(m, ResidualImpl) else m for m in model.body]
    assert all(isinstance(m, GQAAttentionImpl) for m in branches)
    assert [m.layer_id for m in branches] == [e.layer_id for e in entries]
    assert len({m.layer_id for m in branches}) == 3
    assert len({id(m) for m in branches}) == 3


def test_ordinary_block_accepts_id_without_assign_id():
    module = build_block(Linear(4, 4, residual=True), 3)
    assert isinstance(module, ResidualImpl)
