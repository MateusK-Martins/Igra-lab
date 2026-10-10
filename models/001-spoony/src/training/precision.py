from contextlib import AbstractContextManager, nullcontext

import torch

from src.config.types import PrecisionConfig


def build_precision(
    device: torch.device,
    config: PrecisionConfig,
) -> tuple[AbstractContextManager, torch.amp.GradScaler]:
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Precision supports CPU and CUDA")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")

        with torch.cuda.device(device):
            if config.mode == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support BF16")

    if config.mode == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA")

    if config.mode == "fp32":
        context = nullcontext()
    else:
        dtype = torch.bfloat16 if config.mode == "bf16" else torch.float16
        context = torch.autocast(device_type=device.type, dtype=dtype)

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=config.mode == "fp16",
    )

    return context, scaler
