# Run Lantern 70M on a rental GPU

Use a Linux x86-64 host with **Python 3.13**, an NVIDIA GPU supporting BF16,
a driver compatible with CUDA 12.8, at least 16GB RAM and preferably 50GB free
persistent disk per model checkout. The CUDA wheel includes its user-space
libraries; no local CUDA compiler is required. The commands below target one GPU.

## Clone, install, train

```bash
git clone --single-branch --branch codex/lantern-70m-wikipedia https://github.com/MateusK-Martins/Igra-lab.git lantern-70m
cd lantern-70m/models/001-spoony
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/cloud-cu128.txt
.venv/bin/python -m scripts.train
```

The training command downloads the required pinned Wikipedia shards, prepares
600M training tokens and 2M validation tokens with the bundled tokenizer,
verifies the prepared artifact hashes, and then starts a freshly initialized
70,006,272-parameter model. The dataset and pretrained weights do not come from
Git. Public data downloads require network access; a Hugging Face token is optional.
Preparation and dependency installation add setup time to the training estimates.

## Default run

- Width 768; nine layers; FFN width 2432; vocabulary 8192; tied embedding/head.
- Causal attention with no sliding-window restriction, trained on 2048 tokens.
- Microbatch 2, accumulation 4: effective batch 8 sequences / 16,384 target tokens.
- 292,968 full training windows; 146,484 batches; **36,621 optimizer updates**.
- One pass covers 599,998,464 supervised target tokens. The 600M stored-token
  corpus also contains the initial shift token and an unused trailing fragment.
- AdamW LR 3e-4; 1000-update warmup; cosine decay to 0.1 of base LR; BF16.
- Validation every 5000 updates and at the end; checkpoints every 10000 updates
  and at the end. Progress prints every 25 updates; TensorBoard logs are saved.

Accumulated loss is weighted by each batch's actual target-token count. There is no QA mixing, SFT or distributed training.
The schedule is computed from the data-window, batch and accumulation settings.
If changing microbatch size, adjust accumulation to preserve the desired effective
batch; do not change those settings while resuming an existing checkpoint.

## Resume and output

Keep `results/` and prepared data on persistent storage. After interruption:

```bash
.venv/bin/python -m scripts.train --resume-latest
```

An explicit checkpoint can instead be supplied with `--resume PATH`.
Resume restores model, optimizer, scheduler, scaler, counters, RNG and loader
position. It requires unchanged code/settings/data, software environment and GPU
identity/count as recorded in the run manifest. Resume on the same rental host
image; moving between GPU models is not exact-resume compatible.
A fresh invocation refuses existing checkpoints, preventing accidental restart.

Outputs are under `results/pretraining/wikipedia-en-600m/`:

- `manifest.json`: data, code, settings and environment identity.
- `tensorboard/`: training and validation metrics.
- `checkpoints/step-00036621.pt`: final full resumable checkpoint.
- `model/model.pt`: final inference weights (no optimizer).
- `model/tokenizer.json`, `model/manifest.json`, `model/training-state.json`.

The exported model is the final update's model. Earlier validation checkpoints
remain available for selecting a better validation result. Copy the model bundle
and keep this exact branch/commit for its runtime and architecture definitions.

To reload the final bundle from this model directory:

```python
from pathlib import Path
import torch
from configs.model.model import build_model
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.checkpoint import load_model_weights

bundle = Path("results/pretraining/wikipedia-en-600m/model")
tokenizer = ByteLevelBPETokenizer.load(bundle / "tokenizer.json")
model, entries = build_model(tokenizer)
load_model_weights(bundle / "model.pt", model=model,
                   tokenizer_path=bundle / "tokenizer.json")
model = model.to("cuda").eval()
```

## Measure before a long run

```bash
.venv/bin/python -m scripts.prepare_data  # only when data is not prepared yet
.venv/bin/python -m scripts.benchmark
```

If data is already present, skip preparation. The benchmark uses real 2048-token
batches, performs 3 warmup and 10 measured optimizer updates, validates the model,
and checks weight save/load logits. It discards its weights and writes only
`results/data/wikipedia-en-600m/benchmark-2048.json`. It does not produce a
pretrained checkpoint or contaminate the subsequent freshly initialized run.

Observed on the local 45W RTX 4050 laptop: Lantern 20.8k tokens/sec and 3.35GB
allocated VRAM; T16 11.1k tokens/sec and 5.23GB. These are short measurements.
For 600M tokens, compute-only planning estimates for a full single GPU are:

| GPU | Lantern | T16 |
| --- | --- | --- |
| RTX 4090 24GB | 80–160k tok/s; 1.0–2.1h | 40–80k tok/s; 2.1–4.2h |
| A100 80GB | 90–180k tok/s; 0.9–1.9h | 50–100k tok/s; 1.7–3.3h |
| H100 80GB | 120–240k tok/s; 0.7–1.4h | 70–140k tok/s; 1.2–2.4h |

Rental figures are unmeasured extrapolations from the local benchmark and GPU
compute/memory capabilities, with broad allowances for this small workload's
kernel overhead. PCIe/SXM variants, power limits and host/storage performance
matter. Budget another 10–25% for validation, checkpoints and throughput drift,
plus setup/data preparation. Use the on-host benchmark to replace the estimate.
A 600M-token run verifies this chosen training budget, not a guaranteed quality
or convergence target.

Hardware references for the planning ranges: [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/),
[A100 datasheet](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf),
[H100 specifications](https://www.nvidia.com/en-gb/data-center/h100/).
These sources describe hardware capabilities; they do not benchmark this code.
