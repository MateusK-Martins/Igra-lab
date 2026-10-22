"""Matched story-generation probes across pretraining, instruction SFT and TinyStories."""

import json
from pathlib import Path

import torch

from configs.inference import samples as config
from configs.model.model import build_model
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.checkpoint import load_model_weights
from src.training.manifest import file_sha256


def main():
    torch.set_num_threads(4)
    out = Path("results/finetuning/lantern70m-instruct-tinystories1m")
    bundles = {
        "pretrained600m": Path(
            "/home/mateus/.codex/worktrees/lantern-70m-wikipedia-600m/Igra-lab/models/001-spoony/results/pretraining/wikipedia-en-600m/model"
        ),
        "instruction1m": Path(
            "/home/mateus/.codex/worktrees/lantern70m-sft1m-normal/Igra-lab/models/001-spoony/results/sft/lantern70m-instruct1m-normal/model"
        ),
        "tinystories1m": out / "model",
    }
    rows = []
    hashes = {}
    tokenizer_hashes = set()
    for stage, bundle in bundles.items():
        tok = ByteLevelBPETokenizer.load(bundle / "tokenizer.json")
        tokenizer_hashes.add(file_sha256(bundle / "tokenizer.json"))
        model, _ = build_model(tok)
        load_model_weights(
            bundle / "model.pt", model=model, tokenizer_path=bundle / "tokenizer.json"
        )
        model.cuda().eval()
        hashes[stage] = file_sha256(bundle / "model.pt")
        with torch.inference_mode():
            for index, prompt in enumerate(config.PROMPTS):
                for mode in ("greedy", "top_k"):
                    seed = 42 + index
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    tokens = torch.tensor([tok.encode(prompt)], device="cuda")
                    generated = []
                    for _ in range(128):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(tokens[:, -2048:])[:, -1].float()
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
                        tokens = torch.cat([tokens, next_id], dim=1)
                    row = {
                        "stage": stage,
                        "prompt": prompt,
                        "mode": mode,
                        "seed": seed,
                        "generated_tokens": len(generated),
                        "continuation": tok.decode(generated),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    assert len(tokenizer_hashes) == 1
    report = {
        "model_hashes": hashes,
        "tokenizer_sha256": next(iter(tokenizer_hashes)),
        "temperature": 0.8,
        "top_k": 40,
        "max_new_tokens": 128,
        "samples": rows,
    }
    (out / "stage-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    text = """# Matched samples: pretrained, instruction-tuned, TinyStories-tuned

Three checkpoints: original Lantern70M after 600M Wikipedia tokens; then normal
1M-token instruction SFT; then 1M-token TinyStories fine-tuning from that normal
instruction model. No think-token experiment is used here. Same 8192-token
vocabulary, six prompts, greedy and top-k40/temperature0.8 decoding, 128 new-token
cap. Sampling uses seed42+prompt index for every model. Continuations are unedited.
These are qualitative probes; they are not a factual or story-quality benchmark.

"""
    for prompt in config.PROMPTS:
        text += f"## Prompt\n\n```text\n{prompt}\n```\n\n"
        for mode in ("greedy", "top_k"):
            for stage in bundles:
                row = next(
                    r
                    for r in rows
                    if r["prompt"] == prompt
                    and r["mode"] == mode
                    and r["stage"] == stage
                )
                text += (
                    f"### {stage}, {mode}\n\n```text\n{row['continuation']}\n```\n\n"
                )
    (out / "STAGE_COMPARISON.md").write_text(text)


if __name__ == "__main__":
    main()
