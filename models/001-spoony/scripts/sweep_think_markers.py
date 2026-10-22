"""Sweep supplied think-marker counts using the same model and prompts."""

import json
from pathlib import Path

import torch

from configs.model.model import build_model
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.checkpoint import load_model_weights
from src.training.manifest import file_sha256


def main():
    torch.set_num_threads(4)
    out = Path("results/sft/lantern70m-instruct1m-think3-consume")
    cases = json.loads((out / "evaluation-examples.json").read_text())
    tok = ByteLevelBPETokenizer.load(out / "model/tokenizer.json")
    model, _ = build_model(tok)
    load_model_weights(
        out / "model/model.pt", model=model, tokenizer_path=out / "model/tokenizer.json"
    )
    model.cuda().eval()
    rows = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            for mode in ["greedy", "top_k"]:
                for markers in [0, 3, 6, 12, 24]:
                    prompt = (
                        "User: "
                        + case["instruction"]
                        + "\nAssistant: "
                        + ("<think>" * markers)
                    )
                    x = torch.tensor([tok.encode(prompt)], device="cuda")
                    generated = []
                    seed = 42 + index
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    for _ in range(128):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x[:, -2048:])[:, -1].float()
                        if mode == "greedy":
                            next_id = logits.argmax(-1, keepdim=True)
                        else:
                            values, ids = torch.topk(logits / 0.8, 40, dim=-1)
                            next_id = ids.gather(
                                -1,
                                torch.multinomial(
                                    values.softmax(-1), 1, generator=generator
                                ),
                            )
                        token = next_id.item()
                        if token == tok.token_to_id("<|eot|>"):
                            break
                        generated.append(token)
                        x = torch.cat([x, next_id], dim=1)
                    row = {
                        **case,
                        "mode": mode,
                        "supplied_think_tokens": markers,
                        "seed": seed,
                        "continuation": tok.decode(generated),
                        "generated_tokens": len(generated),
                        "generated_think_tokens": generated.count(8192),
                    }
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "model_sha256": file_sha256(out / "model/model.pt"),
        "marker_targets": 0,
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_k": 40,
        "samples": rows,
    }
    (out / "marker-count-sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    text = """# Consume-think model: marker-count sweep

Same frozen model, nine dataset prompts, greedy and seeded top-k40/temperature0.8
sampling, 128-token generation cap. Compare 0, 3, 6, 12 and 24 supplied markers.
Only three-marker prefixes were used during this model's stage-2 training.
Groups: stage2_seen, stage1_only (excluded from stage2), held_out (neither stage).
These qualitative samples do not measure reasoning improvement.

"""
    for case in cases:
        text += f"## {case['group']} — row {case['source_index']}\n\nInstruction:\n```text\n{case['instruction']}\n```\n\nDataset answer:\n```text\n{case['response']}\n```\n\n"
        for mode in ["greedy", "top_k"]:
            for markers in [0, 3, 6, 12, 24]:
                row = next(
                    r
                    for r in rows
                    if r["group"] == case["group"]
                    and r["source_index"] == case["source_index"]
                    and r["mode"] == mode
                    and r["supplied_think_tokens"] == markers
                )
                text += f"### {mode}, {markers} supplied think tokens\n\n```text\n{row['continuation']}\n```\n\n"
    (out / "MARKER_COUNT_SWEEP.md").write_text(text)


if __name__ == "__main__":
    main()
