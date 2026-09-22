# MVCT research code

This repository contains the cleaned research implementation of the
Multi-View Consistency-Aware Transformer (MVCT). The maintained pipeline is
the token-level CAT implementation in `src/`; legacy prototypes, local model
snapshots, raw data, checkpoints, caches, and manuscript working files are not
part of this release.

## Repository layout

```text
MVCT/
├── src/                 # Core model, training, evaluation, and experiment code
├── tests/               # Unit and protocol-integrity tests
├── docs/                # Detailed experiment and reproduction notes
├── figures/             # Figure-generation source code and small source data
├── data/README.md       # Required dataset schema and placement
├── models/README.md     # Required local model snapshots
├── outputs/             # Generated locally; ignored by Git
├── requirements.txt     # Reproducible runtime dependencies
├── requirements-dev.txt # Test dependency
└── pyproject.toml       # Project and pytest configuration
```

## Setup

Python 3.10 or 3.11 is recommended. Create a virtual environment, install the
CUDA-compatible PyTorch build for your machine, and then install the remaining
dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

The pinned environment mirrors the completed cloud experiments. If the pinned
CUDA build is not compatible with your host, install the appropriate PyTorch
wheel first, then install the rest of the requirements.

Follow [data/README.md](data/README.md) and [models/README.md](models/README.md)
to provide the excluded inputs. Stanford CoreNLP must be reachable at
`http://localhost:9000` for OpenIE-backed cache construction.

## Quick checks

Run commands from the repository root:

```bash
python -m pytest -q
python src/build_mvc_cache.py --limit 100 --no-use-nli
python src/main_cat.py
```

For a full cache with NLI:

```bash
python src/build_mvc_cache.py --use-nli
```

The primary entry points are:

- `src/main_cat.py`: train and evaluate the maintained MVCT model.
- `src/build_mvc_cache.py`: build the versioned MVCM cache.
- `src/run_ablation_cat.py`: run the main ablation suite.
- `src/run_strict_replication.py`: execute the sealed replication protocol.
- `src/run_locked_three_channel_test.py`: run the locked three-channel test.
- `src/run_strict_baselines_seed42.py`: run controlled baselines.
- `src/run_p0_analysis.py`: assemble subgroup and efficiency analyses.

Detailed operational notes are in `docs/`.

## Configuring paths

Defaults live in `src/config.py`. These environment variables avoid editing
source files when data or models are stored elsewhere:

```text
MVCT_DATA_PATH
MVCT_BERT_MODEL
MVCT_SBERT_MODEL
MVCT_NLI_MODEL
MVCT_CACHE_DIR
MVCT_OUTPUT_DIR
```

## What is intentionally excluded

- `data_all.csv` and any other raw/private dataset;
- Hugging Face model snapshots and `*.safetensors` weights;
- `*.pt` checkpoints, MVC caches, split manifests, logs, and outputs;
- virtual environments, IDE metadata, audit copies, archives, and manuscript
  production files.

If public redistribution of the dataset or trained checkpoints is permitted,
publish them as a versioned GitHub Release or an external archival dataset and
record checksums and download links here. Do not add multi-gigabyte artifacts
to normal Git history.

## Reproducibility notes

The strict runners preserve protocol hashes, reject incompatible source
artifacts, and separate validation-time selection from locked test evaluation.
Generated result directories remain local by default. See
`docs/README_STRICT_REPLICATION.md` and `docs/README_REPRO_CHECK.md` for the
full procedure.

## License and citation

No license is included because the project owner must choose one and verify
that all dataset, model, and third-party-code terms are compatible before the
repository is made public. Add a `LICENSE` file and citation metadata before a
public release.
