import pytest
import torch

from src.model.definitions import CumulativeMixer
from src.model.scan import decayed_prefix_sum


def sequential(x, decay, initial):
    state = initial[:, 0]
    states = []
    for token in range(x.shape[1]):
        state = decay * state + x[:, token]
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("length", [1, 2, 7, 256, 1024])
def test_parallel_scan_matches_recurrence_at_boundary_decays(length):
    torch.manual_seed(123)
    x = torch.randn(2, length, 6, dtype=torch.float64)
    decay = torch.tensor([0, 1e-5, 0.1, 0.5, 0.99, 1], dtype=torch.float64)
    initial = torch.randn(2, 1, 6, dtype=torch.float64)
    expected = sequential(x, decay, initial)
    actual = decayed_prefix_sum(x, decay, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_parallel_scan_gradients_include_per_feature_decay_and_initial_state():
    x = torch.randn(2, 9, 4, dtype=torch.float64, requires_grad=True)
    decay = torch.tensor(
        [0.01, 0.3, 0.7, 0.99], dtype=torch.float64, requires_grad=True
    )
    initial = torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(
        decayed_prefix_sum(x, decay, initial).square().sum(), (x, decay, initial)
    )
    second = torch.autograd.grad(
        sequential(x, decay, initial).square().sum(), (x, decay, initial)
    )
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(decayed_prefix_sum, (x, decay, initial))


def test_disabled_decay_is_cumsum_and_unit_decay_matches_it():
    x = torch.randn(2, 19, 4, dtype=torch.float64)
    torch.testing.assert_close(decayed_prefix_sum(x), x.cumsum(1))
    torch.testing.assert_close(
        decayed_prefix_sum(x, torch.ones(4, dtype=torch.float64)), x.cumsum(1)
    )


def test_decay_is_learnable_independent_per_feature_and_does_not_change_projection_rng():
    torch.manual_seed(42)
    plain = CumulativeMixer(8).build()
    torch.manual_seed(42)
    decayed = CumulativeMixer(8, key_decay=True, value_decay=True).build()
    for name in ("q_proj", "k_proj", "v_proj"):
        torch.testing.assert_close(
            getattr(plain, name).weight, getattr(decayed, name).weight
        )
    assert decayed.key_decay_logits.shape == (8,)
    assert decayed.value_decay_logits.shape == (8,)
    assert decayed.key_decay_logits is not decayed.value_decay_logits
    torch.testing.assert_close(
        decayed.value_decay_logits.sigmoid(), torch.full((8,), 0.99)
    )
    decayed(torch.randn(2, 13, 8)).square().mean().backward()
    assert torch.isfinite(decayed.key_decay_logits.grad).all()
    assert torch.isfinite(decayed.value_decay_logits.grad).all()
    assert decayed.value_decay_logits.grad.unique().numel() > 1
