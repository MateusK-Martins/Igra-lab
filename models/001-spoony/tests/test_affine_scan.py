import pytest
import torch

from src.model.decay import TokenDecayImpl
from src.model.definitions import CumulativeMixer
from src.model.scan import affine_prefix_sum, decayed_prefix_sum


def sequential(update, decay, initial):
    state = initial[:, 0]
    states = []
    for token in range(update.shape[1]):
        state = decay[:, token] * state + update[:, token]
        states.append(state)
    return torch.stack(states, 1)


@pytest.mark.parametrize("length", [1, 2, 7, 256, 1024])
def test_affine_scan_matches_varying_coefficients_and_nonzero_initial(length):
    torch.manual_seed(42)
    update = torch.randn(2, length, 4, dtype=torch.float64)
    decay = torch.rand_like(update)
    decay[:, ::3, 0] = 0
    decay[:, ::5, 1] = 1
    initial = torch.randn(2, 1, 4, dtype=torch.float64)
    torch.testing.assert_close(
        affine_prefix_sum(update, decay, initial),
        sequential(update, decay, initial),
        atol=1e-12,
        rtol=1e-12,
    )


def test_affine_scan_all_gradients_match_sequential_reference():
    update = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    decay = torch.rand_like(update, requires_grad=True)
    initial = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    actual = torch.autograd.grad(
        affine_prefix_sum(update, decay, initial).square().sum(),
        (update, decay, initial),
    )
    expected = torch.autograd.grad(
        sequential(update, decay, initial).square().sum(), (update, decay, initial)
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert torch.autograd.gradcheck(affine_prefix_sum, (update, decay, initial))


def test_affine_constant_gate_matches_diagonal_and_zero_gate_resets_history():
    x = torch.randn(2, 17, 4, dtype=torch.float64)
    decay = torch.tensor([0, 0.3, 0.99, 1], dtype=torch.float64)
    torch.testing.assert_close(
        affine_prefix_sum(x, decay.expand_as(x)), decayed_prefix_sum(x, decay)
    )
    torch.testing.assert_close(
        affine_prefix_sum(x, torch.zeros_like(x), torch.ones(2, 1, 4, dtype=x.dtype)), x
    )


def test_token_gate_initialization_preserves_rng_and_learns_input_dependence():
    torch.manual_seed(42)
    plain = CumulativeMixer(4).build()
    torch.manual_seed(42)
    token = CumulativeMixer(
        4, key_decay=True, value_decay=True, decay_mode="token"
    ).build()
    for name in ("q_proj", "k_proj", "v_proj"):
        torch.testing.assert_close(
            getattr(plain, name).weight, getattr(token, name).weight
        )
    gate = token.value_decay_gate
    assert token.key_decay_gate.weight is not gate.weight
    x = torch.randn(2, 5, 4)
    torch.testing.assert_close(gate(x), torch.full_like(x, 0.99))
    gate(x).sum().backward()
    assert gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all()
    with torch.no_grad():
        gate.weight.copy_(torch.eye(4))
    actual = gate(x)
    expected = torch.sigmoid(x + gate.bias)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[:, 0], actual[:, 1])
    assert (actual > 0).all() and (actual < 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gate_autocast_keeps_initial_coefficient_in_fp32(dtype):
    gate = TokenDecayImpl(4, 0.99).cuda()
    with torch.autocast("cuda", dtype=dtype):
        result = gate(torch.randn(2, 3, 4, device="cuda"))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full_like(result, 0.99))
