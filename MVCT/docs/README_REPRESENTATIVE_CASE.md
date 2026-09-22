# Representative case selection and visualization

## Why the old heatmaps cannot be reused

The old example was the first item yielded by an earlier test loader. It used a four-channel model and did not retain a sample identifier, ground-truth labels, prediction metadata, or a link to the final locked protocol. The final paper therefore requires a new example selected from the frozen three-channel test predictions.

## Reproducible selection policy

The primary illustrative case represents the subgroup in which the aggregate veracity gain is strongest:

1. the true group is human-written fake news (HF);
2. MVCT predicts both veracity and provenance correctly;
3. the architecture-matched model without MVCM and RoBERTa both miss veracity while retaining correct provenance;
4. among all qualifying cases, select the sample closest to the median MVCT true-label probability advantage, with only a small penalty for atypical article length;
5. never use the visual appearance of the heatmaps or attention pattern to select the sample.

If the strict set is empty, the script records and applies a predefined fallback tier. It also selects one HR boundary case in which MVCT produces a false positive while RoBERTa is correct. Showing both cases is preferable when space permits because it mirrors the subgroup trade-off rather than presenting only a success.

## Selection command

Run this in the final AutoDL project environment:

```bash
cd /tmp/pycharm_project_173
python select_representative_case.py \
  --main-suite /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/locked_three_channel_test_v1 \
  --baseline-suite /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_baselines_seed42_v1 \
  --data-csv data_all.csv \
  --output /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/p0_submission_evidence/representative_case_selection.json
```

The command reads existing frozen predictions only. It does not retrain, reevaluate, change thresholds, or modify the locked test suite.

## Extract final-model matrices and attention

After the selector reports the sample ID, run:

```bash
python extract_representative_case.py \
  --selection /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/p0_submission_evidence/representative_case_selection.json \
  --main-suite /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/locked_three_channel_test_v1 \
  --case primary \
  --device cuda \
  --output /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/p0_submission_evidence/representative_case_evidence.json
```

To extract the balancing HR failure case, replace `--case primary` with `--case boundary` and use a different output filename. The extractor performs only two one-sample frozen forward passes and reads the existing cache; it does not rebuild OpenIE, SBERT, or NLI features.

## Evidence to extract after selection

For the selected sample, load the frozen `three_channel_full` checkpoint and its existing MVC cache, then save:

- sample ID, joint label, predictions and probabilities for MVCT, w/o MVCM, and RoBERTa;
- the semantic, logical, and local 3 × 3 matrices from the cache;
- the learned masked-softmax channel weights;
- the signed fused view-level bias produced by `model.combine_mvc`;
- the final CAT-layer view-aggregated attention matrix;
- the same outputs for the architecture-matched w/o-MVCM checkpoint;
- checkpoint, cache, prediction-file, and selection-report hashes.

The final figure should use one row per selected case and the following panel sequence:

`sample and predictions → three active MVC matrices → fused signed bias → view-level attention/control comparison`

Describe it as an illustrative diagnostic example, not a faithful explanation or causal proof.
