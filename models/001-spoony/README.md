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

The model architecture, tokenizer, data source, and first training
configuration are intentionally undecided. They will be documented before the
first implementation is added.

## Directory contract

- `src/model.py` defines the architecture for this model.
- `src/config/types.py` defines the typed configuration contracts.
- `configs/data/` holds dataset definitions.
- `configs/tokenizers/` holds tokenizer preparation settings.
- `src/data/`, `src/training/`, `src/evaluation/`, and `src/inference/` hold
  separate stages of the model lifecycle.
- `tests/` contains tests that run without a full training job.
- `results/` stores small, versioned experiment records only.
