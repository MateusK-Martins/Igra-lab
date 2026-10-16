"""CUDA affine scan comparisons against an independent sequential recurrence."""

import pytest
import torch

pytest.importorskip("triton")
from src.model.scan_triton import affine_prefix_sum_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def sequential(update, decay, initial):
    state = update.new_zeros(update.shape[0], update.shape[2], dtype=torch.float32)
    if initial is not None:
        state = initial[:, 0].float()
    states = []
    for t in range(update.shape[1]):
        state = decay[:, t].float() * state + update[:, t].float()
        states.append(state)
    return torch.stack(states, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 7, 19), (2, 256, 384), (1, 512, 384)])
@pytest.mark.parametrize("cached", [False, True])
def test_forward_and_backward(dtype, shape, cached):
    torch.manual_seed(42)
    u = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
    g = torch.rand(shape, device="cuda", dtype=dtype)
    g[:, ::3, 0] = 0
    g[:, ::5, -1] = 1
    g.requires_grad_()
    s = (
        torch.randn(shape[0], 1, shape[2], device="cuda", requires_grad=True)
        if cached
        else None
    )
    inputs = (u, g, s) if cached else (u, g)
    expected = sequential(u, g, s)
    actual = affine_prefix_sum_triton(u, g, s)
    incoming = torch.randn_like(actual)
    eg = torch.autograd.grad(expected, inputs, incoming)
    ag = torch.autograd.grad(actual, inputs, incoming)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(
            a,
            e,
            rtol=0.02 if dtype == torch.bfloat16 else 3e-5,
            atol=0.01 if dtype == torch.bfloat16 else 3e-5,
        )
        assert torch.isfinite(a).all()
        assert a.dtype == e.dtype


@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_chunks_preserve_outputs_and_gradients(chunk_size, dtype):
    torch.manual_seed(42)
    u = torch.randn(2, 17, 19, device="cuda", dtype=dtype).requires_grad_()
    g = torch.full_like(u, 0.99).requires_grad_()
    s = torch.randn(2, 1, 19, device="cuda", requires_grad=True)
    full = affine_prefix_sum_triton(u, g, s)
    state = s
    pieces = []
    for t in range(0, 17, chunk_size):
        piece = affine_prefix_sum_triton(
            u[:, t : t + chunk_size].contiguous(),
            g[:, t : t + chunk_size].contiguous(),
            state,
        )
        pieces.append(piece)
        state = piece[:, -1:].contiguous()
    chunked = torch.cat(pieces, 1)
    torch.testing.assert_close(chunked, full, rtol=3e-5, atol=3e-5)
    incoming = torch.randn_like(full)
    for a, e in zip(
        torch.autograd.grad(chunked, (u, g, s), incoming),
        torch.autograd.grad(full, (u, g, s), incoming),
        strict=True,
    ):
        torch.testing.assert_close(
            a,
            e,
            rtol=0.03 if dtype == torch.bfloat16 else 3e-5,
            atol=0.03 if dtype == torch.bfloat16 else 3e-5,
        )


@pytest.mark.parametrize("shape", [(2, 0, 19), (0, 7, 19), (2, 7, 0)])
def test_empty_inputs(shape):
    u = torch.empty(shape, device="cuda", requires_grad=True)
    g = torch.empty_like(u, requires_grad=True)
    s = torch.randn(shape[0], 1, shape[2], device="cuda", requires_grad=True)
    out = affine_prefix_sum_triton(u, g, s)
    assert out.shape == u.shape
    out.sum().backward()
    assert u.grad.shape == u.shape and g.grad.shape == g.shape
    torch.testing.assert_close(s.grad, torch.zeros_like(s))


def test_causality_and_noncontiguous_incoming_gradient():
    u = torch.randn(2, 17, 19, device="cuda", requires_grad=True)
    g = torch.rand_like(u, requires_grad=True)
    actual = affine_prefix_sum_triton(u, g)
    du, dg = torch.autograd.grad(actual[:, :5].sum(), (u, g))
    assert torch.count_nonzero(du[:, 5:]) == 0
    assert torch.count_nonzero(dg[:, 5:]) == 0
    incoming = torch.randn(2, 19, 17, device="cuda").transpose(1, 2)
    assert not incoming.is_contiguous()
    actual = affine_prefix_sum_triton(u, g)
    expected = sequential(u, g, None)
    for a, e in zip(
        torch.autograd.grad(actual, (u, g), incoming),
        torch.autograd.grad(expected, (u, g), incoming),
        strict=True,
    ):
        torch.testing.assert_close(a, e, rtol=3e-5, atol=3e-5)


def test_recurrent_module_gradients_match_reference(monkeypatch):
    import src.model.scan_triton as scan_module
    from src.model.no_k_variant import NoKVariantImpl
    from src.model.scan import affine_prefix_sum

    torch.manual_seed(42)
    model = NoKVariantImpl(16, 1e-6, 0.99, "write_gate", 4).cuda()
    x = torch.randn(2, 17, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(x)
        actual.float().square().mean().backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(scan_module, "affine_prefix_sum_triton", affine_prefix_sum)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(x)
        expected.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
    for n, p in model.named_parameters():
        torch.testing.assert_close(gradients[n], p.grad, rtol=0.05, atol=0.002)
