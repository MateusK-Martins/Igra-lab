"""Generate fixed factual prompts with greedy and seeded top-k decoding."""

import json

import torch

from configs.inference import samples as config
from configs.model.model import build_model
from configs.training.precision import precision_config
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.checkpoint import load_model_weights
from src.training.manifest import file_sha256
from src.training.precision import build_precision


def main():
    torch.set_num_threads(4)
    device = torch.device(config.DEVICE)
    tokenizer_path = config.BUNDLE / "tokenizer.json"
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    model, _ = build_model(tokenizer)
    load_model_weights(
        config.BUNDLE / "model.pt", model=model, tokenizer_path=tokenizer_path
    )
    model.to(device).eval()
    context, _ = build_precision(device, precision_config)
    end_id = tokenizer.token_to_id("<|eot|>")
    samples = []
    with torch.inference_mode():
        for mode in ("greedy", "top_k"):
            for index, prompt in enumerate(config.PROMPTS):
                generator = torch.Generator(device=device).manual_seed(
                    config.SEED + index
                )
                tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
                generated = []
                stopped = False
                for _ in range(config.MAX_NEW_TOKENS):
                    with context:
                        logits = model(tokens[:, -config.CONTEXT_LENGTH :])[
                            :, -1
                        ].float()
                    if mode == "greedy":
                        next_id = logits.argmax(dim=-1, keepdim=True)
                    else:
                        values, indices = torch.topk(
                            logits / config.TEMPERATURE, config.TOP_K, dim=-1
                        )
                        choice = torch.multinomial(
                            values.softmax(dim=-1), 1, generator=generator
                        )
                        next_id = indices.gather(-1, choice)
                    token_id = next_id.item()
                    if token_id == end_id:
                        stopped = True
                        break
                    generated.append(token_id)
                    tokens = torch.cat((tokens, next_id), dim=-1)
                sample = {
                    "mode": mode,
                    "seed": config.SEED + index,
                    "prompt": prompt,
                    "continuation": tokenizer.decode(generated),
                    "generated_tokens": len(generated),
                    "stopped_on_eot": stopped,
                }
                samples.append(sample)
                print(json.dumps(sample), flush=True)
    report = {
        "model_sha256": file_sha256(config.BUNDLE / "model.pt"),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "precision": precision_config.mode,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_k": config.TOP_K,
        "samples": samples,
    }
    config.OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
