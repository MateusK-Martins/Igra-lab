"""Fixed validation-batch TDM diagnostics without optimizer updates."""

import json
import sys

import torch

from configs.data.dataloader import chunking_config, validation_loader_config
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config
from src.data.dataloader import build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.model.tdm import TemporalDifferenceMixerImpl
from src.training.loss import next_token_loss


def rms(x):
    return x.detach().float().square().mean().sqrt().item()


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "initial"
    if stage not in ("initial", "final"):
        raise ValueError("stage must be initial or final")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = ByteLevelBPETokenizer.load(
        preparation_config.output_directory / "tokenizer.json"
    )
    model, _ = build_model(tokenizer)
    if stage == "final":
        saved = torch.load(
            checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(saved["model"])
        del saved
    model.cuda().eval()
    x, y = next(
        iter(
            build_split_dataloader(
                preparation_config.output_directory / "validation",
                chunking_config,
                validation_loader_config,
            )
        )
    )
    x, y = x.cuda(), y.cuda()
    records = {}
    handles = []
    for name, m in model.named_modules():
        if not isinstance(m, TemporalDifferenceMixerImpl):
            continue

        def record(module, args, output, name=name):
            incoming = args[0]
            differences = module.temporal_differences(incoming)
            alpha = module.router(incoming).float().softmax(-1)
            records[name] = {
                "scales": list(module.scales),
                "routing_mean": alpha.mean((0, 1)).cpu().tolist(),
                "routing_entropy_mean": (
                    -(alpha * alpha.clamp_min(1e-30).log()).sum(-1)
                )
                .mean()
                .item(),
                "difference_rms_per_scale": differences.float()
                .square()
                .mean((0, 1, 3))
                .sqrt()
                .cpu()
                .tolist(),
                "scale_weight_norms": module.scale_weights.float()
                .norm(dim=(1, 2))
                .cpu()
                .tolist(),
                "input_rms": rms(incoming),
                "output_rms": rms(output),
                "all_finite": bool(
                    torch.isfinite(output).all()
                    and torch.isfinite(differences).all()
                    and torch.isfinite(alpha).all()
                ),
            }

        handles.append(m.register_forward_hook(record))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = next_token_loss(logits, y).item()
    for handle in handles:
        handle.remove()
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        next_token_loss(model(x), y).backward()
    gradient_norm = (
        sum(
            p.grad.float().square().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        ** 0.5
    )
    for name, m in model.named_modules():
        if isinstance(m, TemporalDifferenceMixerImpl):
            records[name]["scale_gradient_norms"] = (
                m.scale_weights.grad.float().norm(dim=(1, 2)).cpu().tolist()
            )
    report = {
        "stage": stage,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fixed_batch_loss": loss,
        "gradient_norm": gradient_norm,
        "all_gradients_finite": all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        ),
        "layers": records,
        "note": "One held-out batch; routing preferences are not proof of semantic or timescale specialization.",
    }
    path = logging_config.directory.parent / f"{stage}-probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
