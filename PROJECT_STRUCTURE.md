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
| `model-template` | Generic project conventions shared by model templates. |
| `model-template/<family>` | Reusable scaffold for one architecture family. |
| `model/<id>-<name>` | Development branch for one independent model. |
| `release/<id>-<name>-v<version>` | Candidate used for first full runs, benchmarks, and result recording. |

Branches are kept as project history and are not deleted.

## Template hierarchy

`model-template` defines the universal project contract. A family template,
such as `model-template/decoder-transformer`, adds the files and workflows
needed by that architecture family.

A new model branch starts from its applicable family template. It copies the
template into its own model directory. Models never import source code from
another model directory or use a live external shared-library dependency.

When an improvement is shown to be reusable, it is deliberately brought from a
model into its family template. If it applies to every architecture family, it
can then be brought into `model-template`. Existing models are updated only
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
locations, logging, and hardware choices belong in configuration files or
environment variables.

The model definition may explicitly describe its architecture and composition.
Its numeric parameters still come from configuration whenever they should be
experimented with.

## Promotion flow

```text
model-template/<family>
        └── model/<id>-<name>
                └── develop
                        └── release/<id>-<name>-v<version>
                                └── main
```

`develop` receives a working model. A release branch is used for initial serious
training, benchmarks, and result documentation. Only a validated release is
merged into `main` and tagged.
