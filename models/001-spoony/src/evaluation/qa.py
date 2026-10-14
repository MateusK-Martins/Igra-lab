"""SQuAD-v1-style answer normalization, EM/F1 and causal cached generation."""

import copy
import json
import re
import string
from collections import Counter

import torch

from src.model.cache_storage import CacheStorage


def normalize_answer(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, answers):
    exact = f1 = 0.0
    predicted = normalize_answer(prediction).split()
    for answer in answers:
        actual = normalize_answer(answer).split()
        exact = max(
            exact, float(normalize_answer(prediction) == normalize_answer(answer))
        )
        shared = sum((Counter(predicted) & Counter(actual)).values())
        value = 2 * shared / (len(predicted) + len(actual)) if shared else 0.0
        f1 = max(f1, value)
    return exact, f1


@torch.no_grad()
def evaluate_answers(model, entries, tokenizer, config, comparison, context, device):
    records = [
        json.loads(line)
        for line in config.validation_path.read_text().splitlines()
        if line.strip()
    ]
    results = []
    was_training = model.training
    model.eval()
    try:
        for record in records[: comparison.qa_generation_examples]:
            prompt = tokenizer.encode(
                config.instruction_prefix
                + record["instruction"]
                + config.response_prefix
            )
            limit = min(
                comparison.qa_max_new_tokens, config.context_length - len(prompt) + 1
            )
            if limit <= 0:
                raise ValueError("QA prompt has no generation room")
            cache = CacheStorage(copy.deepcopy(entries), device, device)
            ids = torch.tensor([prompt], device=device, dtype=torch.long)
            generated = []
            with context:
                logits = model(ids, cache=cache)
                for step in range(limit):
                    token = logits[0, -1].argmax().item()
                    if token == tokenizer.token_to_id(config.end_token):
                        break
                    generated.append(token)
                    if step + 1 < limit:
                        logits = model(
                            torch.tensor([[token]], device=device), cache=cache
                        )
            prediction = tokenizer.decode(generated)
            exact, f1 = answer_scores(prediction, record["answers"])
            results.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "prediction": prediction,
                    "answers": record["answers"],
                    "exact_match": exact,
                    "f1": f1,
                    "generated_tokens": len(generated),
                    "generation_limit": limit,
                }
            )
    finally:
        model.train(was_training)
    if not results:
        raise ValueError("QA evaluation requires examples")
    return {
        "examples": len(results),
        "exact_match": 100 * sum(row["exact_match"] for row in results) / len(results),
        "f1": 100 * sum(row["f1"] for row in results) / len(results),
        "predictions": results,
    }
