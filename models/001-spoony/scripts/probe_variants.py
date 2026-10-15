"""Fixed held-out batch diagnostics; no optimizer updates or training-data changes."""

import copy
import json
import math
import sys

import torch

from configs.data.dataloader import chunking_config, validation_loader_config
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.precision import precision_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config
from src.data.dataloader import build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.model.cache_storage import CacheStorage
from src.model.state_mechanisms import MassNormalizedImpl, SignedStateImpl
from src.training.loss import next_token_loss as compute_loss
from src.training.precision import build_precision


def statistics(x):
    x = x.detach().float().cpu()
    return {
        "rms": x.square().mean().sqrt().item(),
        "mean": x.mean().item(),
        "variance": x.var(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "finite": bool(torch.isfinite(x).all()),
    }


def heads_statistics(x, heads=6):
    x = x.detach().float().cpu().reshape(-1, heads, x.shape[-1] // heads)
    return {
        "rms": x.square().mean((0, 2)).sqrt().tolist(),
        "variance": x.var((0, 2), unbiased=False).tolist(),
        "mean_absolute_contribution": x.abs().mean((0, 2)).tolist(),
    }


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "initial"
    if stage not in ("initial", "final"):
        raise ValueError("stage must be initial or final")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = ByteLevelBPETokenizer.load(
        preparation_config.output_directory / "tokenizer.json"
    )
    model, entries = build_model(tokenizer)
    if stage == "final":
        saved = torch.load(
            checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(saved["model"])
        del saved
    model.to("cuda").eval()
    context, _ = build_precision(torch.device("cuda"), precision_config)
    loader = build_split_dataloader(
        preparation_config.output_directory / "validation",
        chunking_config,
        validation_loader_config,
    )
    x, y = next(iter(loader))
    x = x.cuda()
    y = y.cuda()
    records = {}
    handles = []
    inputs = {}
    modules = {
        name: m
        for name, m in model.named_modules()
        if hasattr(m, "value_decay_gate") or hasattr(m, "row_gate")
    }
    for name, module in modules.items():
        record = {}
        records[name] = record

        def save_output(m, args, output, record=record):
            record["output"] = statistics(output)
            if (
                not isinstance(m, (SignedStateImpl, MassNormalizedImpl))
                and not hasattr(m, "row_gate")
                and not hasattr(m, "key_output_weights")
            ):
                record["per_head_output"] = heads_statistics(output)

        handles.append(module.register_forward_hook(save_output))

        def save_input(m, args, name=name):
            inputs[name] = args[0].detach()

        handles.append(module.register_forward_pre_hook(save_input))
        for label in ("key_norm", "value_norm", "value_decay_gate"):
            target = getattr(module, label, None)
            if target is not None:

                def save(m, args, output, record=record, label=label, module=module):
                    record[label] = statistics(output)
                    if label != "value_decay_gate" and not hasattr(
                        module, "key_output_weights"
                    ):
                        record[label]["per_head"] = heads_statistics(output)
                    elif label == "value_decay_gate":
                        record[label]["token_variation_std"] = (
                            output.detach()
                            .float()
                            .std(dim=1, unbiased=False)
                            .mean()
                            .item()
                        )

                handles.append(target.register_forward_hook(save))
    cache = CacheStorage(
        copy.deepcopy(entries), torch.device("cuda"), torch.device("cuda")
    )
    with torch.no_grad(), context:
        logits = model(x, cache=cache)
        for handle in handles:
            handle.remove()
        loss = compute_loss(logits, y).item()
        for name, module in modules.items():
            record = records[name]
            if hasattr(module, "gain"):
                raw = cache.read_state(module.layer_id)
                record["denominator_mode"] = module.mode
                record["denominator"] = statistics(raw.key_sum)
                record["gain"] = statistics(module.gain)
                divisor = raw.key_sum + module.eps
                if module.mode == "squared":
                    divisor = divisor.sqrt()
                record["normalized_last_state"] = statistics(
                    module.gain * raw.value_sum / divisor
                )
                writes = module.v_proj(inputs[name]).tanh().float()
                if module.write_gate is not None:
                    gate = module.write_gate(inputs[name])
                    record["write_gate"] = statistics(gate)
                    record["write_gate"]["token_variation_std"] = (
                        gate.std(1, unbiased=False).mean().item()
                    )
                    writes = gate * writes
                record["writes"] = statistics(writes)
                record["write_feature_mean_absolute"] = (
                    writes.mean((0, 1)).abs().mean().item()
                )
                centered = writes - writes.mean((0, 1), keepdim=True)
                left, right = centered[:, :-1], centered[:, 1:]
                scale = (
                    (left.square().mean((0, 1)) * right.square().mean((0, 1)))
                    .sqrt()
                    .clamp_min(1e-12)
                )
                record["write_lag1_correlation"] = statistics(
                    (left * right).mean((0, 1)) / scale
                )
            if isinstance(module, SignedStateImpl):
                state = cache.read_state(module.layer_id)
                record["signed_state"] = statistics(state.value_sum)
                record["token_decay"] = module.token_decay
                if not module.token_decay:
                    record["diagonal_decay"] = statistics(
                        module.value_decay_logits.sigmoid()
                    )
            if isinstance(module, MassNormalizedImpl):
                state = cache.read_state(module.layer_id)
                kn, kd = state.key_sum.chunk(2, -1)
                vn, vd = state.value_sum.chunk(2, -1)
                record["key_numerator"] = statistics(kn)
                record["key_denominator"] = statistics(kd)
                record["value_numerator"] = statistics(vn)
                record["value_denominator"] = statistics(vd)
                record["denominator_floor_fraction"] = (
                    (vd < module.eps).float().mean().item()
                )
            if hasattr(module, "key_output_weights"):
                raw = cache.read_state(module.layer_id).key_sum
                heads = raw.reshape(raw.shape[0], module.num_keys, module.features)
                weights = module.key_output_weights.float()
                record["key_state_rms_per_head"] = (
                    heads.float().square().mean((0, 2)).sqrt().cpu().tolist()
                )
                record["weighted_key_rms_per_head"] = (
                    (heads.float() * weights)
                    .square()
                    .mean((0, 2))
                    .sqrt()
                    .cpu()
                    .tolist()
                )
                record["key_output_weight_norms"] = weights.norm(dim=1).cpu().tolist()
                record["key_decay"] = statistics(module.key_decay_logits.sigmoid())
                record["key_decay_mean_per_head"] = (
                    module.key_decay_logits.sigmoid()
                    .reshape(module.num_keys, -1)
                    .mean(1)
                    .cpu()
                    .tolist()
                )
                record["tied_query"] = module.tied_query
            if getattr(module, "pair_p", None) is not None:
                record["pair_p_norms"] = (
                    module.pair_p.float().norm(dim=(1, 2)).cpu().tolist()
                )
                record["pair_w_norms"] = (
                    module.pair_w.float().norm(dim=(1, 2)).cpu().tolist()
                )
            if getattr(module, "head_mlp", None) is not None:
                raw = cache.read_state(module.layer_id).value_sum
                values = module.value_norm(raw).to(torch.bfloat16)
                h = values.reshape(*values.shape[:-1], module.num_heads, -1).transpose(
                    -1, -2
                )
                delta = module.head_mlp(h)
                record["head_mlp_delta"] = statistics(delta)
                record["head_mlp_relative_change"] = statistics(delta)["rms"] / max(
                    statistics(h)["rms"], 1e-30
                )
                record["head_mlp_parameter_norms"] = {
                    n: p.float().norm().item()
                    for n, p in module.head_mlp.named_parameters()
                }
            if hasattr(module, "row_gate"):
                matrix = cache.read_matrix_state(module.layer_id).matrix
                record["matrix"] = statistics(matrix)
                record["matrix"]["frobenius_per_sample"] = (
                    matrix.float().norm(dim=(1, 2)).cpu().tolist()
                )
                singular = torch.linalg.svdvals(matrix[0].float()).cpu()
                p = singular / singular.sum().clamp_min(1e-30)
                record["matrix"]["singular_values_sample0"] = singular.tolist()
                record["matrix"]["effective_rank_sample0"] = float(
                    torch.exp(-(p * p.clamp_min(1e-30).log()).sum())
                )
                q, _, _, a, b = module.projections(inputs[name])
                record["row_gate"] = statistics(a)
                record["column_gate"] = statistics(b)
                record["matrix_gate_sample"] = statistics(
                    a[:, :16, :, None] * b[:, :16, None, :]
                )
                record["matrix_gate_sample"]["token_variation_std"] = (
                    (a[:, :16, :, None] * b[:, :16, None, :])
                    .std(1, unbiased=False)
                    .mean()
                    .item()
                )
                current = (q[:, -1, :, None] * matrix).sum(1) / math.sqrt(
                    module.key_features
                )
                other = (q[:, -1].flip(-1)[:, :, None] * matrix).sum(1) / math.sqrt(
                    module.key_features
                )
                record["query_sensitivity_rms_difference"] = statistics(
                    current - other
                )["rms"]
    for handle in handles:
        handle.remove()
    model.train()
    model.zero_grad(set_to_none=True)
    with context:
        probe_loss = compute_loss(model(x), y)
    probe_loss.backward()
    gradients = {
        name: statistics(p.grad)
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    report = {
        "stage": stage,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fixed_batch_validation_loss": loss,
        "layers": records,
        "gradient_probe": {
            "loss": probe_loss.item(),
            "all_finite": all(row["finite"] for row in gradients.values()),
            "total_norm": sum(
                p.grad.float().square().sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5,
            "parameters": gradients,
        },
        "notes": "One fixed held-out batch, BF16; head amplitudes are not semantic specialization evidence. Matrix SVD uses final cached state sample0; gate matrix statistics sample first16tokens. Head MLP relative change samples final recurrent vector.",
    }
    root = logging_config.directory.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stage}-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": stage,
                "parameters": report["parameters"],
                "loss": loss,
                "gradient_norm": report["gradient_probe"]["total_norm"],
                "all_finite": report["gradient_probe"]["all_finite"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
