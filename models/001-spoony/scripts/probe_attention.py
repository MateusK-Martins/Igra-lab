"""Fixed-batch initial/final stability and compressed-memory diagnostics."""

import hashlib
import json
import sys

import torch

from configs.data.dataloader import chunking_config, validation_loader_config
from configs.model.model import MODE, build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config
from src.data.dataloader import build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.model.cache_storage import CacheStorage
from src.model.position import rope
from src.training.loss import next_token_loss


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data = preparation_config.output_directory
    tokenizer = ByteLevelBPETokenizer.load(data / "tokenizer.json")
    model, entries = build_model(tokenizer)
    backbone = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not any(name.startswith(f"body.{i}.") for i in range(1, 24, 4)):
            backbone.update(name.encode())
            backbone.update(value.numpy().tobytes())
    phase = sys.argv[1]
    if phase == "final":
        checkpoint = torch.load(
            checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    loader = build_split_dataloader(
        data / "validation", chunking_config, validation_loader_config
    )
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.cuda(), targets.cuda()
    stats = []
    handles = []
    for index in range(1, 24, 4):

        def hook(module, args, output, index=index):
            x = args[0]
            q = module.q_proj(x)
            positions = torch.arange(x.shape[1], device=x.device)
            if module.rope_base is not None:
                q = rope(q.unsqueeze(1), positions, base=module.rope_base).squeeze(1)
            q = q.float()
            if MODE in ("sma4", "sma4_key", "sma4_key_gelu16"):
                matrices = (
                    module.matrix_proj(x)
                    .float()
                    .reshape(
                        *x.shape[:2], module.latent_features, module.latent_features
                    )
                )
                if module.rope_base is not None:
                    matrices = rope(
                        matrices.transpose(1, 2), positions, base=module.rope_base
                    ).transpose(1, 2)
                responses = torch.einsum("bsij,btj->btsi", matrices, q)
                if MODE in ("sma4_key", "sma4_key_gelu16"):
                    k = module.k_proj(x)
                    if module.rope_base is not None:
                        k = rope(
                            k.unsqueeze(1), positions, base=module.rope_base
                        ).squeeze(1)
                    scores = (
                        q @ k.float().transpose(-1, -2) / module.latent_features**0.5
                    )
                else:
                    scores = (
                        torch.einsum("btsi,i->bts", responses, module.scoring.float())
                        / module.latent_features**0.5
                    )
                singular = torch.linalg.svdvals(matrices.detach())
                extra = {
                    "mean_singular_values": singular.mean((0, 1)).tolist(),
                    "mean_matrix_norm": matrices.detach()
                    .norm(dim=(-2, -1))
                    .mean()
                    .item(),
                    **(
                        {"scoring_vector_norm": module.scoring.detach().norm().item()}
                        if MODE == "sma4"
                        else {}
                    ),
                }
            else:
                k = module.k_proj(x)
                if module.rope_base is not None:
                    k = rope(k.unsqueeze(1), positions, base=module.rope_base).squeeze(
                        1
                    )
                k = k.float()
                scores = q @ k.transpose(-1, -2) / module.latent_features**0.5
                extra = {}
            mask = torch.ones(
                x.shape[1], x.shape[1], device=x.device, dtype=torch.bool
            ).tril()
            weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            stats.append(
                {
                    "layer": index,
                    "output_rms": output.detach().float().square().mean().sqrt().item(),
                    "query_rms": q.detach().square().mean().sqrt().item(),
                    "mean_attention_entropy": (
                        -(weights * weights.clamp_min(1e-30).log()).sum(-1)
                    )
                    .detach()
                    .mean()
                    .item(),
                    **extra,
                }
            )

        handles.append(model.body[index].branch.register_forward_hook(hook))
    # Diagnostics use FP32 contractions even while the model trains in BF16.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = next_token_loss(logits, targets)
    loss.backward()
    for handle in handles:
        handle.remove()
    all_finite = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.float().norm() for p in model.parameters()])
    ).item()
    cache = CacheStorage(entries, torch.device("cuda"), torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cached = torch.cat(
            [model(part, cache=cache) for part in inputs[:1, :32].split(7, 1)], 1
        )
        full = model(inputs[:1, :32])
    report = {
        "mode": MODE,
        "phase": phase,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_shared_backbone_sha256": backbone.hexdigest(),
        "fixed_batch_loss": loss.item(),
        "finite_logits": torch.isfinite(logits).all().item(),
        "finite_gradients": all_finite,
        "gradient_norm_before_clipping": grad_norm,
        "layers": stats,
        "bf16_cached_max_logit_difference": (cached.float() - full.float())
        .abs()
        .max()
        .item(),
        "cached_tokens": 32,
        "cache_tensor_elements": sum(
            e.keys.numel() + e.values.numel() for e in cache.entries.values()
        )
        + sum(
            e.matrices.numel() + (0 if e.keys is None else e.keys.numel())
            for e in cache.matrices.values()
        ),
    }
    if not all_finite or not report["finite_logits"]:
        raise RuntimeError("Nonfinite diagnostic")
    path = logging_config.directory.parent / f"{phase}-probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
