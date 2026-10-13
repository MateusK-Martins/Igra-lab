import pytest
import torch

from src.model.definitions import CumulativeMixer
from src.model.stability import RMSNormImpl, SoftRMSNormImpl


def test_soft_norm_reference_feature_axis_and_small_signal_preservation():
    module = SoftRMSNormImpl(3)
    x = torch.tensor([[[1e-5, -1e-5, 1e-5], [3.0, -4.0, 0.0]]], dtype=torch.float64)
    expected = x / torch.sqrt(1 + x.square().mean(-1, keepdim=True))
    torch.testing.assert_close(module(x), expected)
    assert torch.all(module(x).abs() <= x.abs())
    torch.testing.assert_close(module(x)[:, :1], x[:, :1], atol=1e-14, rtol=1e-9)
    torch.testing.assert_close(module(x)[:, :1], module(x[:, :1]))
    assert list(module.parameters()) == []
    rms = RMSNormImpl(3, 1e-6)
    assert rms(x[:, :1]).abs().max() > module(x[:, :1]).abs().max() * 900


def test_soft_norm_gradcheck_and_identity_jacobian_at_zero():
    module = SoftRMSNormImpl(3)
    x = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    zeros = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(module, zeros),
        torch.eye(3, dtype=torch.float64),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_soft_norm_preserves_dtype_and_handles_large_and_zero_inputs(dtype):
    module = SoftRMSNormImpl(2)
    x = torch.tensor([[[0, 0], [1000, -2000]]], dtype=dtype)
    actual = module(x)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.all(actual.float().square().mean(-1) <= 1.01)
    torch.testing.assert_close(
        actual,
        (x.float() * torch.rsqrt(1 + x.float().square().mean(-1, keepdim=True))).to(
            dtype
        ),
    )


def test_norm_choice_preserves_common_initial_projection_weights():
    torch.manual_seed(42)
    original = CumulativeMixer(8, normalization="rms").build()
    torch.manual_seed(42)
    new = CumulativeMixer(8, normalization="soft_rms").build()
    for name in ("q_proj", "k_proj", "v_proj"):
        torch.testing.assert_close(
            getattr(original, name).weight, getattr(new, name).weight
        )
    assert (
        sum(p.numel() for p in original.parameters())
        - sum(p.numel() for p in new.parameters())
        == 16
    )
