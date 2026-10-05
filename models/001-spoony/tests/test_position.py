import math

import pytest
import torch

from src.model.position import rope


def test_rope_matches_scalar_pair_rotations_and_preserves_norms():
    x = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    positions = torch.tensor([0, 1, 12, 103])
    expected = torch.empty_like(x)
    for token, position in enumerate(positions.tolist()):
        for feature in range(0, 6, 2):
            angle = position / 10000.0 ** (feature / 6)
            a, b = x[:, :, token, feature], x[:, :, token, feature + 1]
            expected[:, :, token, feature] = a * math.cos(angle) - b * math.sin(angle)
            expected[:, :, token, feature + 1] = a * math.sin(angle) + b * math.cos(
                angle
            )
    actual = rope(x, positions, base=10000.0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(actual[:, :, 0], x[:, :, 0])


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_rope_preserves_dtype_device_and_chunk_offsets(dtype):
    x = torch.randn(2, 3, 5, 8, dtype=dtype)
    positions = torch.arange(17, 22)
    actual = rope(x, positions, base=10000.0)
    chunks = torch.cat(
        [
            rope(x[:, :, :2], positions[:2], base=10000.0),
            rope(x[:, :, 2:], positions[2:], base=10000.0),
        ],
        dim=2,
    )
    assert actual.dtype == dtype
    assert actual.device == x.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, chunks)
    if dtype in (torch.float16, torch.bfloat16):
        expected = rope(x.float(), positions, base=10000.0).to(dtype)
        torch.testing.assert_close(actual, expected)


def test_rope_gradients_and_relative_position_property():
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([2, 7, 13])
    assert torch.autograd.gradcheck(lambda z: rope(z, positions, base=10000.0), (x,))
    k = torch.randn_like(x)
    before = (
        rope(x, positions, base=10000.0) * rope(k, positions + 5, base=10000.0)
    ).sum(-1)
    shifted = (
        rope(x, positions + 19, base=10000.0) * rope(k, positions + 24, base=10000.0)
    ).sum(-1)
    torch.testing.assert_close(before, shifted)


@pytest.mark.parametrize("base", [0, -1, True, "bad", float("nan"), float("inf")])
def test_rope_rejects_invalid_base(base):
    with pytest.raises(ValueError, match="base"):
        rope(torch.ones(1, 1, 2, 4), torch.arange(2), base=base)


@pytest.mark.parametrize("shape", [(1, 2, 4), (1, 1, 2, 3), (1, 1, 2, 0)])
def test_rope_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="shape|head width"):
        rope(torch.ones(shape), torch.arange(2), base=10000.0)


@pytest.mark.parametrize("positions", [torch.zeros(2, 1), torch.arange(3)])
def test_rope_rejects_invalid_positions(positions):
    with pytest.raises(ValueError, match="positions"):
        rope(torch.ones(1, 1, 2, 4), positions, base=10000.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rope_cuda_matches_cpu():
    x = torch.randn(2, 3, 4, 6)
    positions = torch.arange(10, 14)
    expected = rope(x, positions, base=10000.0)
    actual = rope(x.cuda(), positions.cuda(), base=10000.0)
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected)
