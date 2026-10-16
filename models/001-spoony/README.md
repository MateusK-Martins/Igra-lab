# Tide template

A self-contained template for Tide: signed, selectively written recurrent memory
with token-dependent retention, global RMSNorm and continuous gate initialization.

Default recipe: width384, six layers, SwiGLU1024, vocabulary8192,
context256, BF16, seed42,9000updates. The model has13,774,080parameters.
Each layer caches384FP32 state values (1.5KiB/sample), independent of sequence length.

    g = sigmoid(W_g x + b_g)
    i = sigmoid(W_I x + b_I)
    S = g * S_previous + i * tanh(W_V x)
    output = x + sigmoid(W_Q x) * RMSNorm(S)

The residual wrapper adds x once. The existing parallel affine scan is unchanged.
For j=0..D-1, u=j/(D-1), retention starts at0.90+0.095u and writing
at0.20-0.18u. All gates remain learned and input-dependent.

## Files

- configs/model/model.py: editable model assembly and initialization.
- src/model/tide.py: standalone TideMixerImpl; definitions.py: TideMixer.
- configs/data/, configs/tokenizers/: corpus, loaders and tokenizer.
- configs/training/, configs/sft/: pretraining and optional instruction tuning.
- tests/: Tide mathematical/cache/precision tests plus shared infrastructure.
- results/: empty destination for new runs; no experimental histories.

## Commands

Run from this model directory using the existing model .venv or install
requirements/local-cu128.txt in a new .venv when setting up another checkout.

    .venv/bin/python -m pytest -q
    .venv/bin/ruff check src configs scripts tests
    .venv/bin/ruff format --check src configs scripts tests
    .venv/bin/python -m scripts.materialize_corpus
    .venv/bin/python -m scripts.prepare_data
    .venv/bin/python -m scripts.benchmark
    .venv/bin/python -m scripts.run_baseline

Data preparation refuses to overwrite existing artifacts. The default pinned
TinyStories subset is30000training/256validation documents. Prepared files live
under data/processed/tinystories; runs under results/pretraining/tide.
Data and large checkpoints are ignored by Git. SFT requires the JSONL files
configured in configs/sft/data.py and a matching pretrained checkpoint.

The cleanup preserves parameter names, initial weights and RNG consumption.
Historical Tide model weight dictionaries load directly. External cache IDs are
now TideMixerImpl-<layer_id>; reset/recreate old experimental cache snapshots.
Old full training manifests refer to removed source/config files and are not
resume-compatible with this cleaned template; start a new run or initialize
from weights. Earlier architectures and results remain on their model branches.

See MODEL_ASSEMBLY.md, CHECKPOINTS.md and SFT.md for the shared contracts.

Cleanup verification:419 tests and Ruff passed. Unused attention KV-cache storage
and the fixed-decay scan helper have been removed; only recurrent state storage
and the token-dependent affine scan remain. Recurrent format-v1 snapshots remain
readable; snapshots containing attention KV entries are explicitly rejected. Exact seeded initial parameters, RNG state and FP32 logits match the original Tide under matched CPU threading.
