# Igra-lab project structure

## Purpose

Igra-lab is a research repository for independently developed language models.
Each model must remain runnable from its own directory, including its code,
configuration, tests, documentation, and recorded results.

## Branches

| Branch pattern | Purpose |
| --- | --- |
| `main` | Mature, validated model releases. |
| `develop` | Integration branch for work that is still developing. |
| `template/base` | Generic project conventions shared by model templates. |
| `template/<family>` | Reusable scaffold for one architecture family. |
| `model/<id>-<name>` | Development branch for one independent model. |
| `release/<id>-<name>-v<version>` | Candidate used for first full runs, benchmarks, and result recording. |

Branches are kept as project history and are not deleted.

## Template hierarchy

`template/base` defines the universal project contract. A family template,
such as `template/decoder-transformer`, adds the files and workflows
needed by that architecture family.

A new model branch starts from its applicable family template. It copies the
template into its own model directory. Models never import source code from
another model directory or use a live external shared-library dependency.

When an improvement is shown to be reusable, it is deliberately brought from a
model into its family template. If it applies to every architecture family, it
can then be brought into `template/base`. Existing models are updated only
when that update is intentionally chosen.

## Model layout

Every integrated model lives under `models/<id>-<name>/` and follows this
layout:

```text
models/<id>-<name>/
├── src/                 # model, data, training, evaluation, inference, utils
├── configs/             # model, data, local-training, cloud-training settings
├── tests/               # model-local tests
├── results/             # small reports, metrics, charts, and benchmark records
├── .env.example         # documented environment variables; never secrets
├── pyproject.toml       # model-local Python project and dependencies
└── README.md            # purpose, commands, experiments, and limitations
```

Datasets, checkpoints, large artifacts, private credentials, and real `.env`
files are never committed. Each result records the model version, Git commit,
configuration, dataset version, hardware, and important metrics.

## Configuration rule

Training code must not hide experiment choices in source. Batch size, learning
rate, precision, context length, dataset paths, tokenizer settings, checkpoint
locations, logging, and hardware choices belong in editable Python dataclass
instances under the model's `configs/` directory. Dataclass definitions and
validation code belong under `src/`; there is no experiment configuration
parser or generic dictionary merge layer. Environment variables are for
credentials and environment integration when needed.

The model definition explicitly describes architecture and composition using
typed block definitions. Each block owns build(), while the assembler validates
connections and applies residual wrappers. Dataset definitions belong in
`configs/data/`, tokenizer preparation settings in `configs/tokenizers/`, and
the upcoming editable architecture in `configs/model/`.

Templates may lag behind a model's implementation until reusable changes are
intentionally promoted. Their scaffold is not a live dependency of existing
models.

## Promotion flow

```text
template/<family>
        └── model/<id>-<name>
                └── develop
                        └── release/<id>-<name>-v<version>
                                └── main
```

`develop` receives a working model. A release branch is used for initial serious
training, benchmarks, and result documentation. Only a validated release is
merged into `main` and tagged.
