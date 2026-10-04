# Spoony

Spoony is model 001 in Igra-lab: a small, independent decoder-only Transformer
used as the first research baseline.

Its purpose is to provide a fast, understandable local training loop on the
RTX 4050 while establishing contracts that later decoder Transformer models can
reuse through the template branch.

## Environment

Each model owns its virtual environment. Create it in the model directory and
install the dependency set that matches the machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/local-cu128.txt
```

`requirements/base.txt` lists packages used on every machine. The local CUDA
file adds the PyTorch build tested on the RTX 4050. Cloud dependency files are
added only after selecting the cloud GPU and its compatible CUDA build.

## Status

Implemented and tested:

- Byte-level BPE training, persistence, and a generic tokenizer protocol.
- Source, DataSet, and DataSetPipe contracts with one dataset per split.
- Train-only tokenizer training, document encoding, and end-of-document tokens.
- Binary token storage, document offsets, and tokenizer hashes.
- Disk-backed next-token windows and seeded PyTorch DataLoaders.
- Declarative model validation and assembly with Sequential, Repeat, residuals,
  embeddings, Linear, SwiGLU, and an optionally tied LM head.

No real corpus is selected: `configs/data/datasets.py` defines an empty pipe.
Attention, RMSNorm, positional encoding, a final architecture configuration,
training, evaluation, and SFT remain to be implemented. Linear/SwiGLU assembly
tests establish the machinery; they are not a complete Transformer.

See [ROADMAP.md](ROADMAP.md) and [MODEL_ASSEMBLY.md](MODEL_ASSEMBLY.md) for the
remaining work and composition contracts.

## Commands

Run from this model directory with its virtual environment activated:

```bash
python -m pytest -q
ruff check src configs scripts tests
ruff format --check src configs scripts tests
python -m scripts.prepare_data
```

Preparation requires train datasets to be defined first. It trains BPE, saves
the tokenizer, and processes train and validation. Existing split artifacts
are rejected rather than overwritten; select a fresh output directory for a
new preparation.

## Directory contract

- `src/model/` defines block contracts, construction, and the runtime model.
- `src/config/types.py` defines the typed configuration contracts.
- `configs/data/` holds dataset definitions.
- `configs/data/dataloader.py` holds window and train/validation batch settings.
- `configs/tokenizers/` holds tokenizer preparation settings.
- `src/data/`, `src/training/`, `src/evaluation/`, and `src/inference/` hold
  separate stages of the model lifecycle.
- `tests/` contains tests that run without a full training job.
- `results/` stores small, versioned experiment records only.
