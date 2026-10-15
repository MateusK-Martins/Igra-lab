import copy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.cumulative import CumulativeMixerImpl
from src.model.definitions import MultiKeyMixer
from src.model.multikey import MultiKeyMixerImpl


def sequential(m, x):
    q = m.q_proj(x).sigmoid()
    reader = q if m.tied_query else m.key_query_proj(x).sigmoid()
    k = m.k_proj(x).tanh().reshape(*x.shape[:2], m.num_keys, m.features)
    v = m.v_proj(x).sigmoid()
    decay = m.key_decay_logits.sigmoid().reshape(m.num_keys, m.features)
    gate = (x @ m.value_decay_gate.weight.T + m.value_decay_gate.bias).sigmoid()
    state = torch.zeros_like(k[:, 0])
    value = torch.zeros_like(v[:, 0])
    result = []
    for t in range(x.shape[1]):
        state = state * decay + k[:, t]
        combined = (state * m.key_output_weights).sum(1) * reader[:, t]
        normalized = (
            combined
            * (combined.square().mean(-1, keepdim=True) + m.key_norm.eps).rsqrt()
            * m.key_norm.weights
        )
        value = gate[:, t] * value + v[:, t] * normalized
        normalized_v = (
            value
            * (value.square().mean(-1, keepdim=True) + m.value_norm.eps).rsqrt()
            * m.value_norm.weights
        )
        result.append(q[:, t] * normalized_v)
    return torch.stack(result, 1)


@pytest.mark.parametrize("tied", [True, False])
def test_reference_values_and_all_gradients(tied):
    torch.manual_seed(42)
    m = MultiKeyMixer(4, num_keys=3, tied_query=tied).build().double()
    with torch.no_grad():
        m.key_decay_logits.uniform_(-1, 3)
        m.value_decay_gate.weight.normal_(0, 0.2)
        m.key_norm.weights.uniform_(0.5, 1.5)
    other = copy.deepcopy(m)
    x = torch.randn(2, 11, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = m(x), sequential(other, y)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    weights = torch.randn_like(a)
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-8, atol=1e-9)
    for (name, p), (_, reference) in zip(
        m.named_parameters(), other.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, reference.grad, rtol=1e-8, atol=1e-9)
    assert m.key_decay_logits.grad.abs().sum() > 0
    assert m.key_output_weights.grad.abs().sum() > 0
    assert m.value_decay_gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("tied", [True, False])
def test_cache_chunks_single_tokens_snapshot_reset_and_size(tied, tmp_path):
    definition = MultiKeyMixer(4, num_keys=3, tied_query=tied)
    m = build_block(definition, 0).double()
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 25, 4, dtype=torch.float64)
    with torch.no_grad():
        full = m(x)
        first = m(x[:, :7], cache=cache)
        path = tmp_path / "state.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        middle = m(x[:, 7:16], cache=restored)
        tail = torch.cat([m(x[:, t : t + 1], cache=restored) for t in range(16, 25)], 1)
        torch.testing.assert_close(
            torch.cat([first, middle, tail], 1), full, rtol=1e-8, atol=1e-9
        )
        state = restored.read_state(m.layer_id)
        assert state.key_sum.shape == (2, 1, 12)
        assert state.value_sum.shape == (2, 1, 4)
        size = state.key_sum.numel() + state.value_sum.numel()
        m(x, cache=restored)
        assert (
            size
            == restored.read_state(m.layer_id).key_sum.numel()
            + restored.read_state(m.layer_id).value_sum.numel()
        )
        fresh = CacheStorage(
            [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
        )
        torch.testing.assert_close(m(x, cache=fresh), full)
    with pytest.raises(RuntimeError, match="no_grad"):
        m(x, cache=cache)


@pytest.mark.parametrize("tied", [True, False])
def test_causality_and_gradcheck(tied):
    m = MultiKeyMixer(3, num_keys=2, tied_query=tied).build().double()
    x = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    original = m(x)
    changed = x.detach().clone()
    changed[:, 3:] += 2
    torch.testing.assert_close(original[:, :3], m(changed)[:, :3])
    gradient = torch.autograd.grad(original[:, :3].sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    assert torch.autograd.gradcheck(
        m, (x[:, :2].detach().requires_grad_(),), fast_mode=True
    )


def test_query_tying_and_common_initialization():
    torch.manual_seed(42)
    a = MultiKeyMixerImpl(4, 3, True, 1e-6, 0.99)
    end_a = torch.random.get_rng_state()
    torch.manual_seed(42)
    b = MultiKeyMixerImpl(4, 3, False, 1e-6, 0.99)
    assert torch.equal(end_a, torch.random.get_rng_state())
    assert a.key_query_proj is None and b.key_query_proj is not None
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(42)
    original = CumulativeMixerImpl(
        4, 1e-6, bias=False, value_decay=True, decay_mode="token"
    )
    assert torch.equal(end_a, torch.random.get_rng_state())
    for name in (
        "q_proj.weight",
        "v_proj.weight",
        "value_decay_gate.weight",
        "value_decay_gate.bias",
        "key_norm.weights",
        "value_norm.weights",
    ):
        torch.testing.assert_close(
            a.state_dict()[name], original.state_dict()[name], rtol=0, atol=0
        )
    torch.testing.assert_close(a.key_decay_logits.sigmoid(), torch.full((12,), 0.99))
    assert not torch.equal(a.k_proj.weight[:4], a.k_proj.weight[4:8])
    assert a.key_output_weights.std() > 0


@pytest.mark.parametrize("tied", [True, False])
def test_residual_exactly_once_and_legacy_checkpoint_rejected(tied):
    d = MultiKeyMixer(4, num_keys=3, tied_query=tied, residual=True)
    wrapped = build_block(d, 0)
    x = torch.randn(2, 9, 4)
    torch.testing.assert_close(wrapped(x), x + wrapped.branch(x))
    old = CumulativeMixerImpl(4, 1e-6, bias=False, value_decay=True, decay_mode="token")
    with pytest.raises(RuntimeError):
        wrapped.branch.load_state_dict(old.state_dict())


@pytest.mark.parametrize("tied", [True, False])
def test_long_sequence_zero_and_finite_gradients(tied):
    m = MultiKeyMixer(4, num_keys=3, tied_query=tied).build()
    assert torch.equal(m(torch.zeros(1, 2049, 4)), torch.zeros(1, 2049, 4))
    result = m(torch.randn(1, 2049, 4))
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fp32_scan_and_decay_gradients(tied):
    d = MultiKeyMixer(8, num_keys=10, tied_query=tied)
    m = build_block(d, 0).cuda()
    x = torch.randn(2, 256, 8, device="cuda")
    cache = CacheStorage([d.cache(0)], torch.device("cuda"), torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = m(x)
        result.square().mean().backward()
        with torch.no_grad():
            m(x, cache=cache)
    state = cache.read_state(m.layer_id)
    assert state.key_sum.dtype == state.value_sum.dtype == torch.float32
    assert m.key_decay_logits.sigmoid().mean().item() == pytest.approx(0.99, abs=1e-6)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_keys": 0},
        {"num_keys": True},
        {"tied_query": 1},
        {"eps": 0},
        {"decay_init": 1},
        {"input_features": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MultiKeyMixer(**({"input_features": 4} | kwargs))


def test_full_parameter_budget():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    m, _ = build_model(Tokenizer())
    tied = m.body[1].branch.tied_query
    assert sum(p.numel() for p in m.parameters()) == (16446720 if tied else 17331456)
