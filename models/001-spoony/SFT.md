# Supervised fine-tuning

SFT starts a new training stage from a pretrained checkpoint and the exact same
saved tokenizer. The model architecture and `scripts.train` pretraining recipe
are unchanged. The SFT script reuses the existing training loop, mixed precision,
optimizer/scheduler builders, logging, and checkpoint/resume machinery.

## Data and configuration

Supply separate UTF-8 JSONL files at the paths in `configs/sft/data.py`. Each
nonblank line is one example:

```json
{"instruction": "Who is Romeo?", "response": "Romeo is a protagonist of Romeo and Juliet."}
```

The default prompt is `User: {instruction}\nAssistant: `. Use that same prompt
format for inference. Prefixes are configurable strings, not new vocabulary
entries. Preparation loads the existing tokenizer; it never trains or expands it.
Prompt and response are encoded separately so their token boundary is explicit.
A saved end token is appended to the response and supervised along with it.

`src/sft/data.py` produces fixed-length `(inputs, targets)` tensors, shifted once
for next-token prediction. Prompt and right-padding targets are `-100`; the last
prompt position predicts the first response token. The existing causal model
prevents later padding from influencing real tokens. Examples are independent:
there is no packing, cross-example attention, or silent truncation. Examples
requiring more than `context_length` input positions fail with their file/line.
The initial implementation holds encoded examples in RAM and supports single
instruction/response pairs. Conversation datasets and large disk-backed SFT
stores require further data support.

Editable settings are separate from pretraining:

- `configs/sft/data.py`: files, prompt format, context, and loaders.
- `configs/sft/run.py`: initialization checkpoint, device, optional SFT resume.
- `configs/sft/optimizer.py`, `scheduler.py`, and `precision.py`: update recipe.
- `configs/sft/training.py`: accumulation, validation, logging, checkpoint paths.

The supplied recipe is a starting point, not a validated SFT experiment. Select
suitable examples and a schedule for their size before training. The default
initialization points to the best Shakespeare v4 checkpoint. No SFT dataset is
bundled and no SFT run is launched by this change.

## Start and resume

From the model directory, after supplying data and reviewing configuration:

```bash
.venv/bin/python -m scripts.sft
```

With `run_config.resume_from=None`, `load_model_weights` checks the tokenizer hash
and loads only model weights. Optimizer, scheduler, scaler and counters start
fresh. Initialization does not restore the pretraining RNG or data cursor.
Outputs default to `results/sft/instructions-v1`, leaving pretraining outputs
alone. A manifest identifies the JSONL files, tokenizer, original checkpoint
hash, resolved settings, source code, Git revision and environment. Different
runs cannot overwrite an existing manifest.

To resume interrupted SFT, set `run_config.resume_from` to an SFT checkpoint and
keep the recipe, data, code and environment compatible. This restores the full
SFT training state. The original pretraining checkpoint is no longer required;
its recorded hash is retained as provenance. Resume does not reload base weights.

The project's existing strict resume contract also checks source hashes and Git
revision. Old pretraining runs remain resumable with their matching code/revision;
this feature does not bypass that check after code changes. Their checkpoints
can initialize SFT when architecture and tokenizer match.

## Loss and counters

Cross-entropy ignores `-100` labels. Accumulated training loss and validation
metrics are weighted by the number of supervised targets, not padded lengths.
Empty-supervision microbatches are skipped; an entirely unsupervised update or
validation set raises before an optimizer update. `TrainingState.tokens` retains
its existing consumed-position meaning, including masked/padded positions in
SFT; it is not a count of supervised response tokens. SFT perplexity describes
response targets and should not be compared directly to pretraining perplexity.

Tests cover shifts/masks/end tokens, invalid and oversized examples, unequal
supervision during accumulation/validation, weight initialization, tokenizer
mismatch, fresh SFT, exact interrupted/resumed SFT, and changed-data rejection.
The existing pretraining tests run alongside these checks.
