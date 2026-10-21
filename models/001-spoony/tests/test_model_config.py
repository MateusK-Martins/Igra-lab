import io
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from configs.model.model import build_model
from src.data.tokenizer import ByteLevelBPETokenizer
from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheStorage
from src.model.composition import ResidualImpl
from src.model.stability import RMSNormImpl
from src.model.token_mlp import TokenMLPImpl


@pytest.fixture(autouse=True)
def deterministic_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


@pytest.fixture
def tokenizer(tmp_path):
    trained = ByteLevelBPETokenizer.train(
        ["A small model reads a story.", "Another story for the model."],
        target_vocab_size=300,
        min_frequency=1,
        special_tokens=("<|unk|>", "<|eot|>"),
    )
    path = tmp_path / "tokenizer.json"
    trained.save(path)
    return ByteLevelBPETokenizer.load(path)


def test_config_uses_loaded_tokenizer_without_retraining(tokenizer):
    with patch.object(ByteLevelBPETokenizer, "train") as train:
        model, entries = build_model(tokenizer)
    train.assert_not_called()
    assert model.embedding.num_embeddings == tokenizer.vocab_size
    assert model.lm_head.out_features == tokenizer.vocab_size
    assert model.lm_head.weight is model.embedding.weight
    assert len(model.body) == 24
    for index in range(0, 24, 4):
        norm1, attention, feedforward, norm2 = model.body[index : index + 4]
        assert isinstance(attention, ResidualImpl)
        assert isinstance(attention.branch, GQAAttentionImpl)
        assert isinstance(feedforward, ResidualImpl)
        assert isinstance(feedforward.branch, TokenMLPImpl)
        assert isinstance(norm1, RMSNormImpl)
        assert isinstance(norm2, RMSNormImpl)
        assert attention.branch.q_proj.in_features == 384
        assert attention.branch.q_proj.out_features == 384
        assert attention.branch.k_proj.out_features == 96
        assert attention.branch.v_proj.out_features == 96
        assert attention.branch.window_size == 256
        assert attention.branch.dropout == 0
    assert [entry.layer_id for entry in entries] == [
        model.body[index].branch.layer_id for index in range(1, 24, 4)
    ]
    assert len({entry.layer_id for entry in entries}) == 6
    assert all(entry.keys.shape == (0, 2, 0, 48) for entry in entries)


def test_config_forward_and_next_token_loss_backward(tokenizer):
    model, _ = build_model(tokenizer)
    tokens = torch.randint(tokenizer.vocab_size, (2, 257))
    logits = model(tokens[:, :-1])
    assert logits.shape == (2, 256, tokenizer.vocab_size)
    loss = F.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size), tokens[:, 1:].reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_repeated_builds_and_layers_have_independent_parameters_and_cache(tokenizer):
    first, first_entries = build_model(tokenizer)
    second, second_entries = build_model(tokenizer)
    weights = [first.body[index].branch.q_proj.weight for index in range(1, 24, 4)]
    assert len({weight.data_ptr() for weight in weights}) == 6
    assert len(first.token_mlp_projections) == 2
    assert not hasattr(first, "token_mlp_embeddings")
    assert all(not list(first.body[index].parameters()) for index in range(2, 24, 4))
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert all(
        a is not b and a.keys is not b.keys and a.values is not b.values
        for a, b in zip(first_entries, second_entries, strict=True)
    )
    cache = CacheStorage(first_entries, torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        first(torch.tensor([tokenizer.encode("A story.")]), cache=cache)
    assert all(entry.count > 0 for entry in first_entries)
    assert all(entry.count == 0 for entry in second_entries)


def test_config_cached_chunks_match_full_logits_across_window_boundary(tokenizer):
    model, entries = build_model(tokenizer)
    # Different chunk sizes use different matrix kernels; compare in float64
    # to distinguish cache semantics from accumulated float32 rounding.
    model.double().eval()
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    tokens = torch.randint(tokenizer.vocab_size, (1, 261))
    with torch.no_grad():
        expected = model(tokens)
        actual = torch.cat(
            [
                model(tokens[:, :255], cache=cache),
                model(tokens[:, 255:258], cache=cache),
                model(tokens[:, 258:], cache=cache),
            ],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
    assert all(entry.count == 261 for entry in entries)
    assert all(entry.keys.shape == (1, 2, 261, 48) for entry in entries)


def test_config_state_dict_round_trip_preserves_logits_and_tying(tokenizer):
    model, _ = build_model(tokenizer)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored, _ = build_model(tokenizer)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    tokens = torch.tensor([tokenizer.encode("A small story.")])
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens), model(tokens))
    assert restored.lm_head.weight is restored.embedding.weight


def test_configured_embedding_scale_and_initial_next_token_loss(tokenizer):
    import math

    model, _ = build_model(tokenizer)
    assert model.embedding.weight.std().item() == pytest.approx(0.02, rel=0.02)
    tokens = torch.randint(tokenizer.vocab_size, (2, 17))
    with torch.no_grad():
        loss = F.cross_entropy(
            model(tokens[:, :-1]).reshape(-1, tokenizer.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
    # The wider tied head has higher initial self-token confidence; reject the
    # old unscaled initialization, allowing two nats above uniform here.
    assert abs(loss.item() - math.log(tokenizer.vocab_size)) < 2.0
