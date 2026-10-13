import pytest
import torch

from src.model.definitions import Embedding, Linear, LMHead


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_boundaries_reject_invalid_features(value) -> None:
    with pytest.raises(ValueError, match="output_features"):
        Embedding(value)
    with pytest.raises(ValueError, match="input_features"):
        LMHead(value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_boundaries_reject_invalid_vocab_size(value) -> None:
    embedding = Embedding(4).build(vocab_size=10)
    with pytest.raises(ValueError, match="vocab_size"):
        Embedding(4).build(vocab_size=value)
    with pytest.raises(ValueError, match="vocab_size"):
        LMHead(4).build(vocab_size=value, embedding=embedding)


def test_head_requires_boolean_tying_option() -> None:
    with pytest.raises(ValueError, match="boolean"):
        LMHead(4, tie_to_embedding=1)


def test_embedding_and_tied_head_shapes_and_shared_gradient() -> None:
    embedding = Embedding(4).build(vocab_size=10)
    head = LMHead(4, tie_to_embedding=True).build(vocab_size=10, embedding=embedding)
    assert head.weight is embedding.weight
    assert head.bias is None
    hidden = embedding(torch.tensor([[1, 2, 3]]))
    assert hidden.shape == (1, 3, 4)
    logits = head(hidden)
    assert logits.shape == (1, 3, 10)
    logits.sum().backward()
    assert embedding.weight.grad is not None
    assert head.weight.grad is embedding.weight.grad


def test_tied_head_rejects_width_or_vocab_mismatch() -> None:
    embedding = Embedding(4).build(vocab_size=10)
    with pytest.raises(ValueError, match="width"):
        LMHead(8, True).build(vocab_size=10, embedding=embedding)
    with pytest.raises(ValueError, match="vocabulary"):
        LMHead(4, True).build(vocab_size=11, embedding=embedding)


def test_untied_head_has_independent_weights_and_may_use_different_width() -> None:
    embedding = Embedding(4).build(vocab_size=10)
    head = LMHead(8).build(vocab_size=10, embedding=embedding)
    assert head.weight is not embedding.weight
    assert head.weight.shape == (10, 8)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_linear_rejects_invalid_dimensions(value) -> None:
    with pytest.raises(ValueError, match="input_features"):
        Linear(value, 4)
    with pytest.raises(ValueError, match="output_features"):
        Linear(4, value)


def test_linear_flag_and_fresh_builds() -> None:
    with pytest.raises(ValueError, match="boolean"):
        Linear(4, 4, residual=1)
    recipe = Linear(4, 8)
    first, second = recipe.build(), recipe.build()
    assert first.weight is not second.weight
    assert first(torch.ones(2, 3, 4)).shape == (2, 3, 8)


@pytest.mark.parametrize(
    "std", [0, -0.02, True, None, float("nan"), float("inf"), -float("inf"), "0.02"]
)
def test_embedding_rejects_invalid_initialization_std(std):
    with pytest.raises(ValueError, match="init_std"):
        Embedding(4, init_std=std)


def test_default_embedding_preserves_pytorch_initialization_and_rng():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        expected = torch.nn.Embedding(100, 16)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        actual = Embedding(16).build(vocab_size=100)
        torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)


def test_embedding_scales_initial_weights_without_consuming_extra_random_numbers():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        default = Embedding(64).build(vocab_size=1000)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        scaled = Embedding(64, init_std=0.02).build(vocab_size=1000)
        torch.testing.assert_close(scaled.weight, default.weight * 0.02, rtol=0, atol=0)
        assert scaled.weight.std().item() == pytest.approx(0.02, rel=0.02)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert scaled.weight.requires_grad


def test_scaled_embedding_keeps_shared_head_weights_and_gradients():
    embedding = Embedding(4, init_std=0.02).build(vocab_size=10)
    expected = embedding.weight.detach().clone()
    head = LMHead(4, tie_to_embedding=True).build(vocab_size=10, embedding=embedding)
    assert head.weight is embedding.weight
    torch.testing.assert_close(head.weight, expected, rtol=0, atol=0)
    head(embedding(torch.tensor([[1, 2, 3]]))).sum().backward()
    assert head.weight.grad is embedding.weight.grad
    assert torch.isfinite(embedding.weight.grad).all()
    assert embedding.weight.grad.abs().sum() > 0
