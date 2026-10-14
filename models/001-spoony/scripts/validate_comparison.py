"""Evaluate the final configured checkpoint separately on stories and QA."""

import copy
import json
import re
import string
from collections import Counter

import torch

from configs.data.dataloader import chunking_config, validation_loader_config
from configs.data.qa_corpus import GENERATION_EXAMPLES, MAX_NEW_TOKENS
from configs.model.model import build_model
from configs.sft.data import data_config as qa_config
from configs.sft.scheduler import scheduler_config
from configs.sft.training import checkpoint_config
from configs.tokenizers.preparation import preparation_config
from configs.training.precision import precision_config
from src.data.dataloader import build_dataloader, build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.model.cache_storage import CacheStorage
from src.sft.data import SFTDataset
from src.training.precision import build_precision
from src.training.validation import validate


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    tokenizer = ByteLevelBPETokenizer.load(qa_config.tokenizer_path)
    model, entries = build_model(tokenizer)
    path = checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    del saved
    model.to(device).eval()
    context, _ = build_precision(device, precision_config)
    stories = build_split_dataloader(
        preparation_config.output_directory / "validation",
        chunking_config,
        validation_loader_config,
    )
    qa = build_dataloader(
        SFTDataset(qa_config.validation_path, tokenizer, qa_config),
        validation_loader_config,
    )
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": scheduler_config.total_steps,
    }
    for name, loader in [("stories", stories), ("qa_answers", qa)]:
        loss, perplexity = validate(model, loader, context, device)
        report[name] = {
            "loss": loss,
            "perplexity": perplexity,
            "examples": len(loader.dataset),
        }
    predictions = []
    records = [
        json.loads(line) for line in qa_config.validation_path.read_text().splitlines()
    ]
    end = tokenizer.token_to_id(qa_config.end_token)
    for record in records[:GENERATION_EXAMPLES]:
        prompt = tokenizer.encode(
            qa_config.instruction_prefix
            + record["instruction"]
            + qa_config.response_prefix
        )
        cache = CacheStorage(copy.deepcopy(entries), device, device)
        ids = torch.tensor([prompt], device=device)
        generated = []
        with context:
            for _ in range(
                min(MAX_NEW_TOKENS, qa_config.context_length - len(prompt) + 1)
            ):
                token = model(ids, cache=cache)[0, -1].argmax().item()
                if token == end:
                    break
                generated.append(token)
                ids = torch.tensor([[token]], device=device)
        prediction = tokenizer.decode(generated)
        predicted = normalize(prediction).split()
        exact = f1 = 0.0
        for answer in record["answers"]:
            actual = normalize(answer).split()
            shared = sum((Counter(predicted) & Counter(actual)).values())
            exact = max(exact, float(normalize(prediction) == normalize(answer)))
            f1 = max(
                f1,
                2 * shared / (len(predicted) + len(actual))
                if shared
                else float(not predicted and not actual),
            )
        predictions.append(
            {
                "id": record["id"],
                "prediction": prediction,
                "answers": record["answers"],
                "exact_match": exact,
                "f1": f1,
            }
        )
    report["qa_generation"] = {
        key: 100 * sum(row[key] for row in predictions) / len(predictions)
        for key in ("exact_match", "f1")
    }
    report["qa_generation"]["examples"] = len(predictions)
    report["protocol"] = (
        "Final checkpoint; QA loss supervises first answer plus EOS; greedy EM/F1 over filtered held-out subset, maximum over references; one seed."
    )
    root = checkpoint_config.directory.parent
    (root / "qa-predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (root / "validation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
