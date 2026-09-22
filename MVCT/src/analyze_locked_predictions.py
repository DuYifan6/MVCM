"""Paired statistical analysis of frozen MVCT test predictions.

The script compares the three-channel MVCT with (1) the strongest controlled
external baseline, RoBERTa, and (2) the matched w/o-MVCM internal baseline.
It verifies source protocols and prediction alignment, then reports paired,
joint-label-stratified bootstrap confidence intervals, paired permutation tests
for macro-F1, and exact McNemar tests for accuracy.  Holm correction is applied
separately to the four primary macro-F1 tests and four secondary accuracy tests.
No model, checkpoint, threshold, or prediction is changed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import run_locked_three_channel_test as locked
import run_strict_baselines_seed42 as baselines
import run_strict_replication as strict


SOURCE_DIR = Path(__file__).resolve().parent
ANALYSIS_SEED = 20260917
BOOTSTRAP_REPS = 10000
PERMUTATION_REPS = 10000
CONFIDENCE_LEVEL = 0.95
COMPARISONS = {
    "mvct_vs_roberta": ("three_channel_full", "roberta"),
    "mvct_vs_without_mvcm": ("three_channel_full", "without_mvcm"),
}
TASKS = {
    "fake_real": ("true_fake", "pred_fake"),
    "ai_human": ("true_ai", "pred_ai"),
}
METRICS = ("macro_f1", "accuracy")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary_macro_f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive = 0.0 if positive_denominator == 0 else 2 * tp / positive_denominator
    negative = 0.0 if negative_denominator == 0 else 2 * tn / negative_denominator
    return float((positive + negative) / 2.0)


def metric_value(metric, labels, predictions):
    if metric == "macro_f1":
        return binary_macro_f1(labels, predictions)
    if metric == "accuracy":
        return float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    raise ValueError(f"Unknown metric: {metric}")


def holm_adjust(p_values):
    """Holm step-down adjusted p values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm correction requires finite p values in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def aligned_predictions(named_rows):
    """Return identically ordered arrays; fail on duplicate IDs/label drift."""
    tables = {}
    for name, rows in named_rows.items():
        by_id = {}
        for row in rows:
            sample_id = int(row["sample_id"])
            if sample_id in by_id:
                raise ValueError(f"Duplicate sample_id in {name}: {sample_id}")
            by_id[sample_id] = row
        tables[name] = by_id
    id_sets = {name: set(table) for name, table in tables.items()}
    reference_name = next(iter(tables))
    reference_ids = id_sets[reference_name]
    for name, ids in id_sets.items():
        if ids != reference_ids:
            raise ValueError(
                f"Prediction sample IDs differ: {reference_name} vs {name}"
            )
    ids = np.asarray(sorted(reference_ids), dtype=np.int64)
    arrays = {"sample_id": ids}
    for truth_key in ("true_fake", "true_ai"):
        reference = np.asarray(
            [int(tables[reference_name][value][truth_key]) for value in ids],
            dtype=np.int8,
        )
        for name, table in tables.items():
            candidate = np.asarray(
                [int(table[value][truth_key]) for value in ids], dtype=np.int8
            )
            if not np.array_equal(reference, candidate):
                raise ValueError(f"Ground-truth mismatch for {truth_key}: {name}")
        arrays[truth_key] = reference
    for name, table in tables.items():
        for prediction_key in ("pred_fake", "pred_ai"):
            arrays[f"{name}:{prediction_key}"] = np.asarray(
                [int(table[value][prediction_key]) for value in ids], dtype=np.int8
            )
    return arrays


def stratified_bootstrap_indices(true_fake, true_ai, rng):
    joint = np.asarray(true_fake, dtype=np.int8) * 2 + np.asarray(true_ai, dtype=np.int8)
    parts = []
    for label in range(4):
        members = np.flatnonzero(joint == label)
        if members.size:
            parts.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(parts)


def bootstrap_intervals(arrays, bootstrap_reps=BOOTSTRAP_REPS, seed=ANALYSIS_SEED):
    rng = np.random.default_rng(seed)
    keys = [
        (comparison, task, metric)
        for comparison in COMPARISONS
        for task in TASKS
        for metric in METRICS
    ]
    samples = {key: np.empty(bootstrap_reps, dtype=np.float64) for key in keys}
    for replicate in range(bootstrap_reps):
        index = stratified_bootstrap_indices(
            arrays["true_fake"], arrays["true_ai"], rng
        )
        for comparison, (left, right) in COMPARISONS.items():
            for task, (truth_key, prediction_key) in TASKS.items():
                labels = arrays[truth_key][index]
                left_prediction = arrays[f"{left}:{prediction_key}"][index]
                right_prediction = arrays[f"{right}:{prediction_key}"][index]
                for metric in METRICS:
                    samples[(comparison, task, metric)][replicate] = (
                        metric_value(metric, labels, left_prediction)
                        - metric_value(metric, labels, right_prediction)
                    )
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        key: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "confidence_level": CONFIDENCE_LEVEL,
            "repetitions": bootstrap_reps,
            "method": "paired percentile bootstrap stratified by joint true labels",
        }
        for key, values in samples.items()
    }


def permutation_p_value(labels, left, right, repetitions, rng):
    observed = binary_macro_f1(labels, left) - binary_macro_f1(labels, right)
    exceedances = 0
    for _ in range(repetitions):
        swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
        permuted_left = np.where(swap, right, left)
        permuted_right = np.where(swap, left, right)
        difference = (
            binary_macro_f1(labels, permuted_left)
            - binary_macro_f1(labels, permuted_right)
        )
        exceedances += abs(difference) >= abs(observed) - 1e-15
    return float((exceedances + 1) / (repetitions + 1))


def mcnemar_exact(labels, left, right):
    labels = np.asarray(labels)
    left_correct = np.asarray(left) == labels
    right_correct = np.asarray(right) == labels
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(left_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant": discordant,
        "raw_p": p_value,
        "method": "exact two-sided McNemar/binomial test",
    }


def analyze(arrays, bootstrap_reps=BOOTSTRAP_REPS, permutation_reps=PERMUTATION_REPS, seed=ANALYSIS_SEED):
    intervals = bootstrap_intervals(arrays, bootstrap_reps, seed)
    rng = np.random.default_rng(seed + 1)
    report = {
        "n": int(len(arrays["sample_id"])),
        "sample_unit": "one frozen test-set article",
        "seed": seed,
        "comparisons": {},
    }
    primary_entries, secondary_entries = [], []
    for comparison, (left_name, right_name) in COMPARISONS.items():
        comparison_report = {
            "left": left_name, "right": right_name, "tasks": {}
        }
        for task, (truth_key, prediction_key) in TASKS.items():
            labels = arrays[truth_key]
            left = arrays[f"{left_name}:{prediction_key}"]
            right = arrays[f"{right_name}:{prediction_key}"]
            task_report = {}
            for metric in METRICS:
                left_value = metric_value(metric, labels, left)
                right_value = metric_value(metric, labels, right)
                task_report[metric] = {
                    "left": left_value,
                    "right": right_value,
                    "difference_left_minus_right": left_value - right_value,
                    "bootstrap_ci": intervals[(comparison, task, metric)],
                }
            permutation = {
                "raw_p": permutation_p_value(
                    labels, left, right, permutation_reps, rng
                ),
                "repetitions": permutation_reps,
                "method": "paired two-sided random-swap permutation test",
            }
            task_report["macro_f1"]["test"] = permutation
            mcnemar = mcnemar_exact(labels, left, right)
            task_report["accuracy"]["test"] = mcnemar
            primary_entries.append((comparison, task, permutation))
            secondary_entries.append((comparison, task, mcnemar))
            comparison_report["tasks"][task] = task_report
        report["comparisons"][comparison] = comparison_report

    primary_adjusted = holm_adjust([entry[2]["raw_p"] for entry in primary_entries])
    secondary_adjusted = holm_adjust([entry[2]["raw_p"] for entry in secondary_entries])
    for entries, adjusted in (
        (primary_entries, primary_adjusted),
        (secondary_entries, secondary_adjusted),
    ):
        for (comparison, task, test), adjusted_p in zip(entries, adjusted):
            test["holm_adjusted_p"] = adjusted_p
            test["holm_family_size"] = len(entries)
            test["reject_at_0_05"] = bool(adjusted_p < 0.05)
    report["multiple_comparisons"] = {
        "primary_family": "four macro-F1 tests (two comparisons x two tasks)",
        "secondary_family": "four accuracy tests (two comparisons x two tasks)",
        "method": "Holm step-down correction, separately by metric family",
    }
    report["scope_note"] = (
        "Intervals and tests quantify paired sample-level uncertainty for fixed "
        "seed-42 models on one frozen test split. They do not include training-seed "
        "uncertainty and do not establish population-wide or cross-domain superiority."
    )
    return report


def source_predictions(main_suite, baseline_suite):
    main_protocol = strict.verified_protocol(main_suite / "protocol.json")
    baseline_protocol = strict.verified_protocol(baseline_suite / "protocol.json")
    full = locked.completed_record(
        main_suite / "three_channel_full", main_protocol, "three_channel_full"
    )
    without = locked.completed_record(
        main_suite / "without_mvcm", main_protocol, "without_mvcm"
    )
    roberta = baselines.test_record(
        baseline_suite / "locked_test" / "roberta", baseline_protocol, "roberta"
    )
    rows = {
        "three_channel_full": read_json(Path(full["evaluation_dir"]) / "test_predictions.json"),
        "without_mvcm": read_json(Path(without["evaluation_dir"]) / "test_predictions.json"),
        "roberta": read_json(Path(roberta["output_dir"]) / "test_predictions.json"),
    }
    arrays = aligned_predictions(rows)
    evidence = {
        "main_protocol_id": main_protocol["protocol_id"],
        "baseline_protocol_id": baseline_protocol["protocol_id"],
        "sources": {
            "three_channel_full": {
                "completion_sha256": strict.file_hash(main_suite / "three_channel_full" / "completed.json"),
                "predictions_sha256": strict.file_hash(Path(full["evaluation_dir"]) / "test_predictions.json"),
            },
            "without_mvcm": {
                "completion_sha256": strict.file_hash(main_suite / "without_mvcm" / "completed.json"),
                "predictions_sha256": strict.file_hash(Path(without["evaluation_dir"]) / "test_predictions.json"),
            },
            "roberta": {
                "completion_sha256": strict.file_hash(baseline_suite / "locked_test" / "roberta" / "completed.json"),
                "predictions_sha256": strict.file_hash(Path(roberta["output_dir"]) / "test_predictions.json"),
            },
        },
    }
    return arrays, evidence


def analysis_protocol(main_suite, baseline_suite, arrays, evidence):
    payload = {
        "schema": 1,
        "purpose": "Paired uncertainty analysis of frozen seed-42 test predictions",
        "main_suite": str(main_suite),
        "baseline_suite": str(baseline_suite),
        "source_evidence": evidence,
        "entry_sha256": {Path(__file__).name: strict.file_hash(__file__)},
        "n": int(len(arrays["sample_id"])),
        "sample_id_sha256": hashlib.sha256(arrays["sample_id"].tobytes()).hexdigest(),
        "sample_unit": "one frozen test-set article",
        "comparisons": {
            name: list(models) for name, models in COMPARISONS.items()
        },
        "tasks": list(TASKS),
        "primary_metric": "macro-F1",
        "secondary_metric": "accuracy",
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPS,
            "confidence_level": CONFIDENCE_LEVEL,
            "method": "paired percentile bootstrap stratified by joint true labels",
        },
        "tests": {
            "macro_f1": "paired two-sided random-swap permutation test",
            "accuracy": "exact two-sided McNemar/binomial test",
            "permutation_repetitions": PERMUTATION_REPS,
        },
        "multiplicity": "Holm correction separately across four macro-F1 and four accuracy tests",
        "random_seed": ANALYSIS_SEED,
        "limitations": (
            "Fixed-model sample uncertainty only; no training-seed uncertainty, "
            "event-cluster uncertainty, or external-domain inference."
        ),
    }
    return strict.seal(payload)


def flat_rows(report):
    rows = []
    for comparison, comparison_report in report["comparisons"].items():
        for task, task_report in comparison_report["tasks"].items():
            for metric, result in task_report.items():
                test = result["test"]
                rows.append({
                    "comparison": comparison,
                    "left": comparison_report["left"],
                    "right": comparison_report["right"],
                    "task": task,
                    "metric": metric,
                    "left_score": result["left"],
                    "right_score": result["right"],
                    "difference_left_minus_right": result["difference_left_minus_right"],
                    "ci_lower": result["bootstrap_ci"]["lower"],
                    "ci_upper": result["bootstrap_ci"]["upper"],
                    "raw_p": test["raw_p"],
                    "holm_adjusted_p": test["holm_adjusted_p"],
                    "reject_at_0_05": test["reject_at_0_05"],
                    "test_method": test["method"],
                })
    return rows


def write_csv(path, rows):
    fields = (
        "comparison", "left", "right", "task", "metric", "left_score",
        "right_score", "difference_left_minus_right", "ci_lower", "ci_upper",
        "raw_p", "holm_adjusted_p", "reject_at_0_05", "test_method",
    )
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def protect_output(root, sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Analysis output must not overlap a frozen source suite.")


def main(argv=None):
    from config import cfg

    replications = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-suite", default=str(replications / "locked_three_channel_test_v1")
    )
    parser.add_argument(
        "--baseline-suite", default=str(replications / "strict_baselines_seed42_v1")
    )
    parser.add_argument(
        "--output-dir", default=str(Path(cfg.OUTPUT_DIR) / "analysis" / "paired_statistics_v1")
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    main_suite = Path(args.main_suite).resolve()
    baseline_suite = Path(args.baseline_suite).resolve()
    root = Path(args.output_dir).resolve()
    protect_output(root, (main_suite, baseline_suite))

    arrays, evidence = source_predictions(main_suite, baseline_suite)
    protocol = analysis_protocol(main_suite, baseline_suite, arrays, evidence)
    print(
        f"Preflight passed: n={len(arrays['sample_id'])}; two paired model "
        "comparisons x two tasks; 10,000 bootstrap and permutation repetitions.",
        flush=True,
    )
    if args.check_only:
        print("Check only: no statistical output written.")
        return
    if (root / "protocol.json").exists():
        strict.same_protocol(strict.verified_protocol(root / "protocol.json"), protocol)
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty analysis directory has no protocol; use a new directory.")
    completion_path = root / "completed.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        if completion.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("Completed statistical analysis protocol mismatch.")
        for name, digest in completion["artifact_sha256"].items():
            if strict.file_hash(root / name) != digest:
                raise ValueError(f"Completed statistical artifact changed: {name}")
        print("Paired statistical analysis already completed:", root)
        return

    root.mkdir(parents=True, exist_ok=True)
    strict.atomic_json(root / "protocol.json", protocol)
    report = analyze(arrays)
    report.update({
        "protocol_id": protocol["protocol_id"],
        "analysis_status": "completed",
    })
    strict.atomic_json(root / "paired_statistics.json", report)
    write_csv(root / "paired_statistics.csv", flat_rows(report))
    artifacts = ("protocol.json", "paired_statistics.json", "paired_statistics.csv")
    completion = {
        "protocol_id": protocol["protocol_id"],
        "artifact_sha256": {
            name: strict.file_hash(root / name) for name in artifacts
        },
    }
    strict.atomic_json(completion_path, completion)
    print("Paired statistical analysis completed:", root)


if __name__ == "__main__":
    main()
