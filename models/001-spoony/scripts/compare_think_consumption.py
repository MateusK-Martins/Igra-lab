"""Compare supplied versus absent think markers on seen/excluded/held-out inputs."""

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
                for markers in [False, True]:
                    prompt = (
                        "User: "
                        + case["instruction"]
                        + "\nAssistant: "
                        + ("<think>" * 3 if markers else "")
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
                        "supplied_think_tokens": 3 if markers else 0,
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
    (out / "marker-comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    text = """# Stage 2: consume think markers, without predicting them

Corrected run restarts from stage 1, not from the earlier marker-prediction run.
All 8,970 training records contain three input <think> tokens and **zero marker
targets**. Answer tokens and EOT alone receive supervision. Longest 25% of stage-1
instructions (2,364 unique examples) are cycled to 999,995 total corpus tokens.

The comparisons below use this corrected stage-2 model, with and without three
supplied marker tokens. Stage2_seen examples were trained in both stages;
stage1_only examples were excluded from stage 2; held_out examples were trained
in neither stage. Reference answers come from the dataset, not an accuracy audit.
Greedy and seeded top-k sampling use the same seed per input across conditions.
These are qualitative probes, not a benchmark of reasoning ability.

"""
    for case in cases:
        text += f"## {case['group']} — row {case['source_index']}\n\nInstruction:\n```text\n{case['instruction']}\n```\n\nDataset answer:\n```text\n{case['response']}\n```\n\n"
        for mode in ["greedy", "top_k"]:
            for markers in [False, True]:
                row = next(
                    r
                    for r in rows
                    if r["group"] == case["group"]
                    and r["source_index"] == case["source_index"]
                    and r["mode"] == mode
                    and bool(r["supplied_think_tokens"]) == markers
                )
                text += f"### {mode}, {'three supplied think tokens' if markers else 'no think tokens'}\n\n```text\n{row['continuation']}\n```\n\n"
    (out / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
