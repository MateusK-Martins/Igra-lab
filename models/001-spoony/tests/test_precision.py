from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from configs.model.model import build_model
from src.config.errors import ConfigError
from src.config.types import PrecisionConfig, SchedulerConfig
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler
from src.training.step import train_step


@pytest.mark.parametrize("mode", ["bad", "", None, True, 16])
def test_precision_config_rejects_unknown_modes(mode):
    with pytest.raises(ConfigError, match="mode"):
        PrecisionConfig(mode)


@pytest.mark.parametrize(
    "device,mode,expected_dtype",
    [
        ("cpu", "fp32", torch.float32),
        ("cpu", "bf16", torch.bfloat16),
        ("cuda", "fp32", torch.float32),
        ("cuda", "bf16", torch.bfloat16),
        ("cuda", "fp16", torch.float16),
    ],
)
def test_context_reusable_and_parameters_remain_fp32(device, mode, expected_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and mode == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    context, scaler = build_precision(torch.device(device), PrecisionConfig(mode))
    model = torch.nn.Linear(4, 7).to(device)
    for _ in range(2):
        with context:
            output = model(torch.ones(2, 4, device=device))
        assert output.dtype == expected_dtype
        assert not torch.is_autocast_enabled(device)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert scaler.is_enabled() == (mode == "fp16")


def test_cpu_fp16_and_unsupported_device_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        build_precision(torch.device("cpu"), PrecisionConfig("fp16"))
    with pytest.raises(ValueError, match="CPU and CUDA"):
        build_precision(torch.device("meta"), PrecisionConfig("fp32"))


def test_unavailable_cuda_and_unsupported_bf16_rejected():
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(ValueError, match="unavailable"),
    ):
        build_precision(torch.device("cuda"), PrecisionConfig("fp32"))
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device", return_value=nullcontext()),
        patch("torch.cuda.is_bf16_supported", return_value=False),
        pytest.raises(ValueError, match="BF16"),
    ):
        build_precision(torch.device("cuda"), PrecisionConfig("bf16"))


@pytest.mark.parametrize("mode", ["bf16", "fp16"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_configured_model_mixed_precision_training_update(mode):
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    torch.manual_seed(42)
    device = torch.device("cuda")
    model, _ = build_model(SimpleNamespace(vocab_size=17))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 3, 0.1))
    context, scaler = build_precision(device, PrecisionConfig(mode))
    if mode == "fp16":
        # A small scale exercises a real update without depending on early
        # dynamic-scale reductions for randomly initialized model weights.
        scaler = torch.amp.GradScaler("cuda", init_scale=16)
    before = model.embedding.weight.detach().clone()
    batch = (torch.randint(17, (1, 8)), torch.randint(17, (1, 8)))
    loss, updated = train_step(
        model, [batch], optimizer, scheduler, context, scaler, device, 1.0
    )
    assert updated and scheduler.last_epoch == 1
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.embedding.weight)
    assert all(
        p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters()
    )
