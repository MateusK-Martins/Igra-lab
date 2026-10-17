# T16 70M — Wikipedia data

Source: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
subset `20231101.en`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
The user selected regular English Wikipedia after Simple English Wikipedia
proved too small for 600M tokens.

- Train: exactly 600,000,000 stored tokens from 550,016 articles.
- Validation: exactly 2,000,000 stored tokens from 1,530 articles.
- Deduplicate stripped text by SHA256, then use a seed-42 content-hash partition
  with 1% assigned to validation before tokenizer preparation.
- Selection follows source shard/row order. It is a deterministic slice rather
  than a globally random sample. No documents are repeated to fill the budget.
- The bundled 8192-token byte-level BPE vocabulary was prepared from the first
  20,000 unique training articles only. Its exact bytes and hash are checked.
- Token budgets include end-of-document tokens. The last article in each token
  store is truncated to meet the exact budget; raw JSONL retains its full text.
- At context/stride 2048 there are 292,968 full training windows and 976
  validation windows. Trailing tokens without a full window are unused.
- Raw rows retain article IDs, URLs and titles. Dataset licenses are
  CC-BY-SA-3.0 and GFDL; retain provenance/attribution when redistributing.

`configs/data/wikipedia.py` selects the corpus and token budgets.
`configs/data/wikipedia-artifacts.json` records expected tokenizer, token-store,
offset and metadata hashes. `configs/tokenizers/wikipedia-en-8192.json` is the
canonical vocabulary. Both model branches use identical data artifacts.

The training script prepares missing data automatically and verifies every
expected artifact hash before model construction. Preparation stages output in
`.preparing` paths and can retry after interruption. Existing completed data is
never overwritten. To prepare without training:

```bash
.venv/bin/python -m scripts.prepare_data
```

Artifacts live under `data/raw/wikipedia-en/` and
`data/processed/wikipedia-en-600m/`; they are ignored by Git. Small preparation,
verification and benchmark reports live under `results/data/wikipedia-en-600m/`.
Historical TinyStories materialization utilities are retained but are not part
of this experiment's run path.
