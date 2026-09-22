# ELECTRA-base dual-head baseline extension

This extension adds `google/electra-base-discriminator` without modifying the
sealed four-baseline v1 suite.  It uses the same saved train/validation/test
splits, title/body/description token budgets, dual heads, loss functions,
seed 42, validation-loss checkpoint selection, and strict Fake/Real threshold
`p(Fake) > 0.55` as the existing controlled baselines.

The extension is post hoc: it was specified after the existing main-model and
baseline test results were observed.  The protocol records this timing, and the
paper must not describe the baseline as preregistered.

## 1. Download the fixed pretrained model

From the project root on the GPU server:

```bash
hf download google/electra-base-discriminator \
  --local-dir models/electra-base-discriminator
```

If the environment provides the older command name, use:

```bash
huggingface-cli download google/electra-base-discriminator \
  --local-dir models/electra-base-discriminator
```

## 2. Read-only preflight

```bash
python run_strict_electra_baseline_seed42.py --check-only
```

The preflight must report the fixed train/validation sizes, the completed
five-variant locked MVCT suite, the completed four-baseline v1 suite, and a
complete local ELECTRA model directory.

## 3. Train and freeze on validation

```bash
python run_strict_electra_baseline_seed42.py
```

This stage trains ELECTRA and selects the checkpoint only by minimum combined
validation task loss.  It does not evaluate the test split.

## 4. Evaluate the locked test split once

Run this only after the validation checkpoint is complete:

```bash
python run_strict_electra_baseline_seed42.py --evaluate-test
```

The main outputs are:

```text
outputs/cat_v1/replications/strict_electra_baseline_seed42_v1/
  protocol.json
  electra_validation_results.csv
  electra_test_results.csv
  electra/
  locked_test/electra/
```

## Manuscript positioning

Use wording such as:

> We additionally adapted ELECTRA-base to the same three-view, dual-head input
> and evaluation protocol. ELECTRA is a general pretrained encoder rather than
> a task-specific architecture; its inclusion follows its use as a fake-news
> detection backbone in the original GossipCop++ study.

Do not call ELECTRA a newly proposed or recent task-specific method.  Its
original pretraining method was published in 2020; the task relevance comes
from the 2024 GossipCop++ evaluation setting.
