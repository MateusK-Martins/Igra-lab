# Decoder Transformer model template

This directory is the starting point for an independent decoder-only
Transformer language model. Copy it into `models/<id>-<name>/` when creating a
new model branch, then replace placeholders with model-specific values.

The template deliberately contains no production model implementation yet.
The first implementation will establish the public contracts for training,
evaluation, inference, configuration, and results.

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

## Directory contract

- `src/model.py` defines the architecture for this model.
- `configs/` holds every experiment choice.
- `src/data/`, `src/training/`, `src/evaluation/`, and `src/inference/` hold
  separate stages of the model lifecycle.
- `tests/` contains tests that run without a full training job.
- `results/` stores small, versioned experiment records only.
