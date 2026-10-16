from types import SimpleNamespace

import torch

from configs.data.dataloader import chunking_config, train_loader_config
from configs.model.model import build_model
from configs.training.scheduler import scheduler_config
from src.model.ac_swiglu import ACSwiGLUImpl


def test_scaled_architecture_exact_parameters_and_preserved_mechanism():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 29_705_216
    assert len(model.body) == 28 and len(entries) == 14
    for layer in model.body[::2]:
        assert isinstance(layer, ACSwiGLUImpl)
        assert layer.ffn.gate.weight.shape == (1152, 512)
        assert layer.condition.weight.shape == (1152, 16)
        assert torch.count_nonzero(layer.condition.weight) == 0
        assert layer.attention.q_proj.weight.shape == (16, 512)
        assert layer.attention.rope_base == 10000.0 and layer.alpha == 0.1
    assert all(e.keys.shape == (0, 1, 0, 16) for e in entries)
    assert chunking_config.context_length == 512 and train_loader_config.batch_size == 8
    assert scheduler_config.total_steps == 6500


def test_scaled_cuda_bf16_full_backward():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, _ = build_model(SimpleNamespace(vocab_size=8192))
    model.cuda()
    tokens = torch.randint(8192, (1, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(tokens)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
