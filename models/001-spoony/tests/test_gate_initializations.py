import copy

import pytest
import torch

from src.model.definitions import NoKVariant
from tests.test_no_k_variant import reference


@pytest.mark.parametrize("recipe", ["mean", "range", "minimum"])
def test_initial_bias_profiles_values_rng_and_unchanged_parameters(recipe):
    features = 12
    if recipe == "mean":
        g, i = (0.95625,) * 4, (0.0925,) * 4
    elif recipe == "minimum":
        g, i = (0.90,) * 4, (0.02,) * 4
    else:
        g = tuple(0.90 + 0.095 * j / (features - 1) for j in range(features))
        i = tuple(0.20 - 0.18 * j / (features - 1) for j in range(features))
    definition = NoKVariant(
        features,
        num_heads=4,
        normalization_scope="global",
        retention_initials=g,
        write_initials=i,
    )
    torch.manual_seed(42)
    model = definition.build()
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    baseline = NoKVariant(
        features,
        num_heads=4,
        normalization_scope="global",
        retention_initials=(0.9, 0.95, 0.98, 0.995),
        write_initials=(0.2, 0.1, 0.05, 0.02),
    ).build()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in baseline.state_dict().items():
        if name not in ("value_decay_gate.bias", "write_gate.bias"):
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    for gate, probabilities in ((model.value_decay_gate, g), (model.write_gate, i)):
        expected = torch.tensor(probabilities).repeat_interleave(
            features // len(probabilities)
        )
        assert gate.bias.requires_grad and gate.weight.requires_grad
        assert torch.count_nonzero(gate.weight) == 0
        torch.testing.assert_close(
            gate(torch.randn(2, 7, features))[0, 0], expected, rtol=1e-6, atol=1e-7
        )
    model = model.double()
    with torch.no_grad():
        model.value_decay_gate.weight.normal_(0, 0.2)
        model.write_gate.weight.normal_(0, 0.2)
    other = copy.deepcopy(model)
    x = torch.randn(2, 17, features, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    out, ref = model(x), reference(other, y)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    (ref * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-8)
    assert model.value_decay_gate.weight.grad.abs().sum() > 0
    assert model.write_gate.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ranged_biases_preserve_fp32_gate_precision_in_bf16():
    g = tuple(0.90 + 0.095 * j / 383 for j in range(384))
    i = tuple(0.20 - 0.18 * j / 383 for j in range(384))
    m = (
        NoKVariant(
            384,
            num_heads=4,
            normalization_scope="global",
            retention_initials=g,
            write_initials=i,
        )
        .build()
        .cuda()
    )
    x = torch.randn(1, 3, 384, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for gate, expected in ((m.value_decay_gate, g), (m.write_gate, i)):
            values = gate(x)
            assert values.dtype == torch.float32
            torch.testing.assert_close(
                values[0, 0],
                torch.tensor(expected, device="cuda"),
                rtol=1e-6,
                atol=1e-7,
            )
        m(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_ranged_endpoints_means_and_preference_length_validation():
    g = torch.tensor([0.90 + 0.095 * j / 383 for j in range(384)], dtype=torch.float64)
    i = torch.tensor([0.20 - 0.18 * j / 383 for j in range(384)], dtype=torch.float64)
    assert g[0].item() == pytest.approx(0.90) and g[-1].item() == pytest.approx(0.995)
    assert i[0].item() == pytest.approx(0.20) and i[-1].item() == pytest.approx(0.02)
    assert (g.diff() > 0).all() and (i.diff() < 0).all()
    assert g.mean().item() == pytest.approx(0.9475)
    assert i.mean().item() == pytest.approx(0.11)
    for probabilities in ((0.5,) * 3, (0.5,) * 7, (0.5,) * 11):
        with pytest.raises(ValueError, match="head or feature"):
            NoKVariant(12, num_heads=4, retention_initials=probabilities)
