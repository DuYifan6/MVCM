# Token-level MVCT reconstruction

The reconstructed implementation is isolated from the legacy experiment.

- Entry point: `main_cat.py`
- MVCM-only precomputation: `build_mvc_cache.py`
- Token-level model: `model_cat.py`
- Hybrid consistency builder: `consistency_hybrid.py`
- New cache: `mvc_cache_v2/`
- New outputs: `outputs/cat_v1/`

The old `main.py`, `model.py`, `dataset.py`, `mvc_cache/`, and `outputs/` are not
overwritten.

On an AutoDL host where `/root/autodl-tmp` exists, the reconstructed code
automatically stores only the large generated artifacts on the data disk:

```text
/root/autodl-tmp/MVCT_runtime/mvc_cache_v2
/root/autodl-tmp/MVCT_runtime/outputs/cat_v1
```

The dataset, model directories, and split manifests remain relative to the
project directory. The two generated paths can be overridden explicitly with
the environment variables `MVCT_CACHE_DIR` and `MVCT_OUTPUT_DIR`.

## Cloud setup

Install the environment on the data disk rather than the 30 GB system disk:

```bash
pip install -r requirements.txt
```

Place the local models at the paths configured in `config.py`:

```text
models/bert-base-uncased
models/all-mpnet-base-v2
models/nli-model
```

Recommended NLI checkpoint:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

Its output order is entailment=0, neutral=1, contradiction=2. The project
therefore sets `NLI_CONTRADICTION_ID = 2` explicitly.

The NLI directory must contain a sequence-classification model whose
`id2label` metadata identifies the contradiction class. If it does not, set
`NLI_CONTRADICTION_ID` explicitly.

## Cache construction on RTX 4080 SUPER

First run a small smoke test:

```bash
python build_mvc_cache.py --limit 100 --use-nli
```

After checking the failure count and GPU memory, build the complete cache:

```bash
python build_mvc_cache.py --use-nli
```

The defaults target a 16-core CPU and a single RTX 4080 SUPER:

- 3 concurrent OpenIE requests;
- SBERT batch size 64;
- NLI batch size 16;
- automatic batch-size reduction after a CUDA out-of-memory error;
- atomic, content-addressed cache files;
- restart-safe reuse of completed samples.

If CoreNLP becomes unstable, reduce `OPENIE_WORKERS` to 2. If GPU utilization
is low while OpenIE is responsive, increase `SBERT_BATCH_SIZE` to 96. Do not
run several cache-building processes on the same GPU because each process
would load another copy of SBERT and NLI.

After a successful cache build, set `USE_NLI = True` in `config.py` so the
training entry point expects exactly the same builder signature.

## Training and experiments

### Paired seed replication (no hyperparameter search)

```bash
python run_repeated_validation.py --max-new-runs 3
```

The frozen plan is seeds 42/43/44 crossed with full, without_mvcm, and
missing_evidence: nine fresh runs. Each seed uses independently seeded fresh
data loaders with identical ordering across the three variants. The split,
threshold, losses, learning rate and early stopping come from the original
`ablation/full/best_checkpoint.pt`; old single-seed runs are not mixed into
this suite. Evaluation is validation-only. Every diagnostic reports the
checkpoint epoch and verifies reproduction of its stored validation loss.

The default output is `outputs/cat_v1/replications/missing_evidence_v1` under
the configured output root (the AutoDL data disk by default). Completion
markers let the same command skip completed runs; `--max-new-runs 3` limits
each invocation to three additional runs. Interrupted individual runs start
fresh in a new attempt directory; they are not resumed from a training epoch,
and the previous attempt is preserved. Use one suite process at a time.
Code/config/data fingerprints prevent accidental mixing after changes.
The script checks disk space based on reference checkpoint size.

`replication_results.csv` and `replication_summary.json` are refreshed after
each completed run. The summary reports sample standard deviation across
seeds, matched-seed differences versus without_mvcm, and completion counts.
Three seeds on the same validation split are not independent test evidence.
Upload the new runner and updated `diagnose_validation.py`; retain the updated
`model_cat.py` and `run_missing_evidence.py` from the previous experiment.

To test the missing-evidence policy as a single-factor, validation-only run:

```bash
python run_missing_evidence.py --reference-run /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/ablation/full
```

This requires the updated `model_cat.py` and `diagnose_validation.py`. It reads
the trusted reference checkpoint's configuration and starts fresh training at
seed 42 with the same settings. Only the missing-evidence policy is enabled:
for a view pair with factual approximately 0 AND logical approximately 0.5,
omit those two channels and renormalize the remaining active channel weights.
The tolerance is 1e-6. Zero available weight yields neutral (zero) bias. Other
pairs, diagonal bias, attention regularization and thresholds are unchanged.
This is a missing-evidence proxy, not a measured OpenIE failure indicator.
The experiment checks the existing cache signature and saved splits, never
rebuilds features, and writes a unique sibling `missing_evidence_*` run folder.
After training it runs validation diagnostics only, with no test evaluation.
The policy is recorded in the checkpoint configuration and restored by the
validation diagnostic. Original models keep their existing default behavior.

To audit existing caches without inference or repairs:

```bash
python audit_mvc_cache.py --run-dir /root/autodl-tmp/MVCT_runtime/outputs/cat_v1
```

Run from the project directory with `diagnose_validation.py` present. CPU mode
is sufficient; no OpenIE, NLI, SBERT, or BERT inference is run. The audit checks
the saved run/cache signature, saved split coverage and overlap, and every
expected matrix's shape, range, symmetry and diagonal. Distribution and label
profiles use only training/validation data and three undirected view pairs.
It writes a new `cache_audit_*` folder with a summary, detailed JSON, file
issues and validation text examples. It never rewrites caches or splits.
Factual-zero/logical-half rates are missing-comparison proxies, not measured
OpenIE failure rates; raw extraction provenance was not retained in the cache.

After a completed main run, diagnose the best checkpoint on its saved validation
split (run from the project directory):

```bash
python diagnose_validation.py
```

This reads the checkpoint's configuration, checks its cache signature, restores
the saved `val.csv` IDs and checks their labels and group IDs. It requires only
BERT and existing MVC caches; it does not load SBERT/NLI or contact OpenIE. It
writes independent task losses, validation metrics, predictions, and a fixed
threshold scan into a new timestamped folder inside the run directory. The
selection objective is validation Fake/Real macro-F1. It does not evaluate the
test split, change the training configuration, or overwrite earlier results.
Validation-selected scores are not unbiased held-out performance estimates.

```bash
python main_cat.py
python run_ablation_cat.py
```

Every run writes a full configuration, best checkpoint, metrics, per-sample
predictions, learned MVCM weights, CAT bias strengths, and split manifests.
The reconstructed data pipeline removes incomplete rows, exact duplicates,
and identical inputs with conflicting labels before producing group-safe
train/validation/test splits.
