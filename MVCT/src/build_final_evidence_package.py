"""Build the immutable, non-submission final evidence package for MVCT.

The package consolidates already frozen test results, channel ablations,
three-seed architecture-selection evidence, and paired statistics.  It does
not train/evaluate a model, recompute a metric, or create journal submission
materials.  Source and output hashes make this the single numerical source for
subsequent manuscript revision.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import analyze_locked_predictions as paired
import run_locked_three_channel_test as locked
import run_strict_baselines_seed42 as baselines
import run_strict_factual_confirmation as factual_confirmation
import run_strict_replication as strict


SOURCE_DIR = Path(__file__).resolve().parent
METRICS = ("fake_accuracy", "fake_macro_f1", "ai_accuracy", "ai_macro_f1")
BASELINE_ORDER = ("tfidf_logreg", "textcnn", "bert", "roberta")
MAIN_ORDER = (*BASELINE_ORDER, "without_mvcm", "three_channel_full")
ABLATION_ORDER = (
    "three_channel_full", "without_semantic", "without_logical",
    "without_local", "without_mvcm",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_completed_artifacts(root):
    completion = read_json(root / "completed.json")
    protocol = strict.verified_protocol(root / "protocol.json")
    if completion.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError(f"Completion/protocol mismatch: {root}")
    for name, digest in completion.get("artifact_sha256", {}).items():
        if strict.file_hash(root / name) != digest:
            raise ValueError(f"Completed artifact changed: {root / name}")
    return protocol, completion


def verify_sources(main_suite, baseline_suite, factual_suite, statistics_dir):
    main_protocol = strict.verified_protocol(main_suite / "protocol.json")
    main_rows = [
        locked.completed_record(main_suite / variant, main_protocol, variant)
        for variant in locked.VARIANTS
    ]
    rebuilt_main = locked.summarize(main_rows)
    saved_main = read_json(main_suite / "locked_test_summary.json")
    if rebuilt_main != saved_main:
        raise ValueError("Locked main-test summary differs from completed records.")

    baseline_protocol = strict.verified_protocol(baseline_suite / "protocol.json")
    baseline_rows = [
        baselines.test_record(
            baseline_suite / "locked_test" / variant, baseline_protocol, variant
        )
        for variant in baselines.VARIANTS
    ]
    rebuilt_baselines = baselines.summarize(baseline_rows, "locked_test")
    saved_baselines = read_json(baseline_suite / "baseline_test_summary.json")
    if rebuilt_baselines != saved_baselines:
        raise ValueError("Baseline test summary differs from completed records.")

    factual_protocol = strict.verified_protocol(factual_suite / "protocol.json")
    factual_rows = [
        factual_confirmation.completed_record(
            factual_suite / f"seed_{seed}" / factual_confirmation.VARIANT,
            factual_protocol,
            seed,
        )
        for seed in factual_confirmation.SEEDS
    ]
    if any(row is None for row in factual_rows):
        raise ValueError("Three-seed factual-channel confirmation runs are incomplete.")
    factual = read_json(factual_suite / "factual_confirmation_summary.json")
    if factual.get("completed_pairs") != 3 or factual.get("completed_new_runs") != 2:
        raise ValueError("Three-seed factual-channel confirmation is incomplete.")

    statistics_protocol, _ = verify_completed_artifacts(statistics_dir)
    statistics = read_json(statistics_dir / "paired_statistics.json")
    if statistics.get("analysis_status") != "completed" or statistics.get("n") != 1948:
        raise ValueError("Paired statistical analysis is incomplete or has unexpected n.")

    evidence = {
        "main_test": {
            "protocol_id": main_protocol["protocol_id"],
            "protocol_sha256": strict.file_hash(main_suite / "protocol.json"),
            "summary_sha256": strict.file_hash(main_suite / "locked_test_summary.json"),
        },
        "baselines": {
            "protocol_id": baseline_protocol["protocol_id"],
            "protocol_sha256": strict.file_hash(baseline_suite / "protocol.json"),
            "summary_sha256": strict.file_hash(baseline_suite / "baseline_test_summary.json"),
        },
        "factual_confirmation": {
            "protocol_id": factual_protocol["protocol_id"],
            "protocol_sha256": strict.file_hash(factual_suite / "protocol.json"),
            "summary_sha256": strict.file_hash(factual_suite / "factual_confirmation_summary.json"),
        },
        "paired_statistics": {
            "protocol_id": statistics_protocol["protocol_id"],
            "protocol_sha256": strict.file_hash(statistics_dir / "protocol.json"),
            "results_sha256": strict.file_hash(statistics_dir / "paired_statistics.json"),
        },
    }
    return saved_main, saved_baselines, factual, statistics, evidence


def metric_values(row):
    return {metric: float(row[metric]) for metric in METRICS}


def build_main_table(main, baseline):
    sources = {
        **baseline["by_variant"],
        "without_mvcm": main["by_variant"]["without_mvcm"],
        "three_channel_full": main["by_variant"]["three_channel_full"],
    }
    full = sources["three_channel_full"]
    rows = []
    for variant in MAIN_ORDER:
        values = metric_values(sources[variant])
        row = {"variant": variant, **values}
        for metric in METRICS:
            row[f"full_minus_{metric}"] = full[metric] - values[metric]
        rows.append(row)
    return rows


def build_ablation_table(main):
    full = main["by_variant"]["three_channel_full"]
    rows = []
    for variant in ABLATION_ORDER:
        values = metric_values(main["by_variant"][variant])
        row = {"variant": variant, **values}
        for metric in METRICS:
            row[f"full_minus_{metric}"] = full[metric] - values[metric]
        rows.append(row)
    return rows


def build_architecture_table(factual):
    mapping = {
        "four_channel_full": "full",
        "three_channel_full": "without_factual",
    }
    rows = []
    for label, source in mapping.items():
        source_row = factual["by_variant"][source]
        row = {"variant": label, "n_seeds": int(source_row["n"])}
        for metric in (*METRICS, "val_loss"):
            row[f"{metric}_mean"] = source_row[metric]["mean"]
            row[f"{metric}_sd"] = source_row[metric]["sd"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def package_protocol(paths, evidence):
    payload = {
        "schema": 1,
        "purpose": "Immutable numerical evidence package for manuscript revision",
        "sources": {name: str(path) for name, path in paths.items()},
        "source_evidence": evidence,
        "entry_sha256": {Path(__file__).name: strict.file_hash(__file__)},
        "tables": {
            "main_test": list(MAIN_ORDER),
            "ablation_test": list(ABLATION_ORDER),
            "architecture_selection_validation": [
                "four_channel_full", "three_channel_full"
            ],
            "paired_statistics": "two comparisons x two tasks x two metrics",
        },
        "reporting_boundary": (
            "Test tables use one frozen seed-42 model. Three-seed values are "
            "validation-only architecture-selection evidence. Paired intervals "
            "quantify sample-level, not training-seed, uncertainty."
        ),
        "submission_materials": False,
    }
    return strict.seal(payload)


def protect_output(root, sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Evidence output must not overlap a source directory.")


def main(argv=None):
    from config import cfg

    replications = Path(cfg.OUTPUT_DIR) / "replications"
    analysis_root = Path(cfg.OUTPUT_DIR) / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-suite", default=str(replications / "locked_three_channel_test_v1"))
    parser.add_argument("--baseline-suite", default=str(replications / "strict_baselines_seed42_v1"))
    parser.add_argument("--factual-suite", default=str(replications / "strict_factual_confirmation_v1"))
    parser.add_argument("--statistics-dir", default=str(analysis_root / "paired_statistics_v1"))
    parser.add_argument("--output-dir", default=str(analysis_root / "final_evidence_v1"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    paths = {
        "main_suite": Path(args.main_suite).resolve(),
        "baseline_suite": Path(args.baseline_suite).resolve(),
        "factual_suite": Path(args.factual_suite).resolve(),
        "statistics_dir": Path(args.statistics_dir).resolve(),
    }
    root = Path(args.output_dir).resolve()
    protect_output(root, paths.values())
    main_result, baseline_result, factual, statistics, evidence = verify_sources(
        paths["main_suite"], paths["baseline_suite"],
        paths["factual_suite"], paths["statistics_dir"],
    )
    protocol = package_protocol(paths, evidence)
    main_rows = build_main_table(main_result, baseline_result)
    ablation_rows = build_ablation_table(main_result)
    architecture_rows = build_architecture_table(factual)
    statistics_rows = paired.flat_rows(statistics)
    print(
        "Preflight passed: 6 main-test rows, 5 ablation rows, "
        "2 architecture-selection rows, and 8 paired-statistics rows.",
        flush=True,
    )
    if args.check_only:
        print("Check only: no evidence-package files written.")
        return
    if (root / "protocol.json").exists():
        strict.same_protocol(strict.verified_protocol(root / "protocol.json"), protocol)
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty evidence directory has no protocol; use a new directory.")
    if (root / "completed.json").exists():
        completion = read_json(root / "completed.json")
        if completion.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("Completed evidence package protocol mismatch.")
        for name, digest in completion["artifact_sha256"].items():
            if strict.file_hash(root / name) != digest:
                raise ValueError(f"Completed evidence artifact changed: {name}")
        print("Final evidence package already completed:", root)
        return

    root.mkdir(parents=True, exist_ok=True)
    strict.atomic_json(root / "protocol.json", protocol)
    write_csv(root / "main_test_results.csv", main_rows)
    write_csv(root / "ablation_test_results.csv", ablation_rows)
    write_csv(root / "architecture_selection_validation.csv", architecture_rows)
    write_csv(root / "paired_statistics.csv", statistics_rows)
    payload = {
        "protocol_id": protocol["protocol_id"],
        "main_test": main_rows,
        "ablation_test": ablation_rows,
        "architecture_selection_validation": architecture_rows,
        "paired_statistics": statistics_rows,
        "source_evidence": evidence,
        "reporting_boundary": protocol["reporting_boundary"],
    }
    strict.atomic_json(root / "final_evidence.json", payload)
    artifacts = (
        "protocol.json", "main_test_results.csv", "ablation_test_results.csv",
        "architecture_selection_validation.csv", "paired_statistics.csv",
        "final_evidence.json",
    )
    strict.atomic_json(root / "completed.json", {
        "protocol_id": protocol["protocol_id"],
        "artifact_sha256": {
            name: strict.file_hash(root / name) for name in artifacts
        },
    })
    print("Final evidence package completed:", root)


if __name__ == "__main__":
    main()
