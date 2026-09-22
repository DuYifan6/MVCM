"""One-time locked test evaluation for the frozen three-channel MVCM study.

This entry never trains, scans thresholds, selects checkpoints, or rebuilds
caches.  It verifies the sealed validation protocols and evaluates exactly five
predeclared seed-42, loss-selected checkpoints on the saved test split at the
fixed Fake/Real threshold 0.55.  Each checkpoint is evaluated in a fresh
process and the small JSON outputs are integrity sealed for safe resumption.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import run_strict_replication as strict
import run_strict_three_channel_ablation as three
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEED = 42
FAKE_THRESHOLD = 0.55
VARIANT_SOURCES = {
    "three_channel_full": ("channel", "without_factual"),
    "without_mvcm": ("channel", "without_mvcm"),
    "without_semantic": ("three", "three_without_semantic"),
    "without_logical": ("three", "three_without_logical"),
    "without_local": ("three", "three_without_local"),
}
VARIANTS = tuple(VARIANT_SOURCES)
EXPECTED_MASKS = {
    "three_channel_full": [True, False, True, True],
    "without_mvcm": [False, False, False, False],
    "without_semantic": [False, False, True, True],
    "without_logical": [True, False, False, True],
    "without_local": [True, False, True, False],
}
MEASURES = ("fake_accuracy", "fake_macro_f1", "ai_accuracy", "ai_macro_f1")
OUTPUT_ARTIFACTS = (
    "test_metrics.json",
    "test_predictions.json",
    "evaluation_metadata.json",
)


def source_row(suite, protocol, variant):
    row = strict.completed_record(
        suite / f"seed_{SEED}" / variant,
        protocol,
        variant,
        SEED,
    )
    if row is None:
        raise ValueError(f"Missing completed source checkpoint: {variant}")
    return row


def restore_saved_split(config, split_name="test"):
    """Restore an exact saved split and attach existing caches without writes."""
    import pandas as pd
    from diagnose_validation import signature_from_config
    from dataset_cat import _cache_key, _clean_and_group_dataframe

    signature = signature_from_config(config)
    cache_dir = Path(config.MVC_CACHE_DIR)
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = strict.read_json(manifest_path)
    if manifest.get("builder_signature") != signature:
        raise ValueError(
            "Checkpoint and cache signatures differ; caches must not be rebuilt."
        )

    split_path = Path(config.SPLIT_DIR) / f"{split_name}.csv"
    saved = pd.read_csv(split_path, dtype={"_split_group": str})
    frame = _clean_and_group_dataframe(
        config.DATA_PATH,
        drop_incomplete=config.DROP_INCOMPLETE,
        max_body_chars=config.MAX_BODY_CHARS,
        group_column=config.GROUP_COLUMN,
        drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES,
    )
    if not frame.ID.is_unique or not saved.ID.is_unique:
        raise ValueError("Sample IDs must be unique to restore the saved split.")
    missing_ids = sorted(set(saved.ID) - set(frame.ID))
    if missing_ids:
        raise ValueError(f"Saved split contains {len(missing_ids)} unknown IDs.")
    frame = frame.set_index("ID").loc[saved.ID].reset_index()
    for column in ("label_fake", "label_ai", "_split_group"):
        if frame[column].astype(str).tolist() != saved[column].astype(str).tolist():
            raise ValueError(f"Saved {split_name} split mismatch: {column}")

    builder = SimpleNamespace(cache_signature=signature)
    frame["mvc_path"] = [
        str(cache_dir / f"mvc_{_cache_key(row, builder)}.npy")
        for row in frame.itertuples(index=False)
    ]
    missing = [path for path in frame.mvc_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} {split_name} caches; no features were recomputed."
        )
    return frame, split_path, cache_dir, manifest_path, signature


def prepare(baseline, channel_suite, three_suite):
    """Verify all frozen sources and seal the one-time test protocol."""
    source_protocol, _, train, val, _ = strict.prepare(baseline, (SEED,))
    channel_protocol = strict.verified_protocol(channel_suite / "protocol.json")
    three_protocol = strict.verified_protocol(three_suite / "protocol.json")
    three.verify_source_protocol(source_protocol, channel_protocol)
    three.verify_source_protocol(source_protocol, three_protocol)
    if three_protocol.get("source_channel_protocol_id") != channel_protocol["protocol_id"]:
        raise ValueError("Three-channel suite does not point to the verified channel suite.")

    source_rows = {}
    completion_evidence = {}
    for label, (suite_name, source_variant) in VARIANT_SOURCES.items():
        suite = channel_suite if suite_name == "channel" else three_suite
        protocol = channel_protocol if suite_name == "channel" else three_protocol
        row = source_row(suite, protocol, source_variant)
        checkpoint = Path(row["run_dir"]) / "best_checkpoint.pt"
        recorded = row["artifact_sha256"].get("best_checkpoint.pt")
        # source_row() already verifies every artifact digest, including this
        # checkpoint.  Do not hash the 1.4-GiB file a second time here.
        if not recorded:
            raise ValueError(f"Checkpoint digest is absent: {checkpoint}")
        source_rows[label] = row
        completion_path = suite / f"seed_{SEED}" / source_variant / "completed.json"
        completion_evidence[label] = {
            "source_suite": str(suite),
            "source_variant": source_variant,
            "completion_sha256": strict.file_hash(completion_path),
            "checkpoint_sha256": recorded,
            "checkpoint": str(checkpoint.resolve()),
        }

    reference_config = SimpleNamespace(**strict.read_json(
        Path(source_rows["three_channel_full"]["run_dir"]) / "config.json"
    ))
    if float(reference_config.FAKE_THRESHOLD) != FAKE_THRESHOLD:
        raise ValueError("The frozen reference threshold is not 0.55.")
    frame, split_path, _, manifest_path, signature = restore_saved_split(
        reference_config, "test"
    )
    test_cache_hashes = {
        str(int(sample_id)): strict.file_hash(path)
        for sample_id, path in zip(frame.ID.tolist(), frame.mvc_path.tolist())
    }

    payload = {
        key: value for key, value in source_protocol.items()
        if key != "protocol_id"
    }
    entries = dict(payload["entry_sha256"])
    for name in (
        Path(__file__).name,
        "evaluate_cat.py",
        "diagnose_validation.py",
        "dataset_cat.py",
        "model_cat.py",
        "utils.py",
    ):
        entries[name] = strict.file_hash(SOURCE_DIR / name)
    payload.update({
        "purpose": "One-time locked test evaluation of the frozen three-channel MVCM",
        "seeds": [SEED],
        "variants": list(VARIANTS),
        "entry_sha256": entries,
        "channel_protocol_id": channel_protocol["protocol_id"],
        "three_channel_protocol_id": three_protocol["protocol_id"],
        "source_completion_evidence": completion_evidence,
        "expected_channel_masks": EXPECTED_MASKS,
        "selection": (
            "All checkpoints were previously selected by minimum combined validation "
            "task loss. Test labels cannot select models, epochs, or thresholds."
        ),
        "primary_metric": "Test Fake/Real macro-F1 at fixed strict > 0.55",
        "fake_threshold": FAKE_THRESHOLD,
        "ai_decision_rule": "argmax of the two AI/Human logits",
        "threshold_scan": "Forbidden for test evaluation",
        "test_evaluation": True,
        "test_split": str(split_path.resolve()),
        "test_split_sha256": strict.file_hash(split_path),
        "test_manifest": str(manifest_path.resolve()),
        "test_manifest_sha256": strict.file_hash(manifest_path),
        "test_cache_signature": signature,
        "test_cache_sha256_by_sample_id": test_cache_hashes,
        "sample_counts": {
            "train": len(train), "validation": len(val), "test": len(frame)
        },
        "reporting_scope": (
            "Seed-42 locked test estimates for one predeclared three-channel model "
            "and four predeclared ablations; descriptive, not a significance test."
        ),
    })
    return strict.seal(payload), source_rows


def metric_row(label, source, metrics, output):
    return {
        "variant": label,
        "seed": SEED,
        "source_variant": VARIANT_SOURCES[label][1],
        "checkpoint_epoch": source["checkpoint_epoch"],
        "fake_threshold": FAKE_THRESHOLD,
        "fake_accuracy": metrics["fake_report"]["accuracy"],
        "fake_macro_f1": metrics["fake_report"]["macro avg"]["f1-score"],
        "ai_accuracy": metrics["ai_report"]["accuracy"],
        "ai_macro_f1": metrics["ai_report"]["macro avg"]["f1-score"],
        "evaluation_dir": str(output.resolve()),
    }


def completed_record(slot, protocol, label, record=None):
    done = slot / "completed.json"
    if record is None and not done.exists():
        return None
    row = strict.read_json(done) if record is None else record
    if (row.get("protocol_id"), row.get("variant"), row.get("seed")) != (
        protocol["protocol_id"], label, SEED
    ):
        raise ValueError(f"Completed identity mismatch: {done}")
    output = Path(row["evaluation_dir"]).resolve()
    if output.parent != slot.resolve() or not output.name.startswith("evaluation_"):
        raise ValueError("Completion points outside its assigned evaluation slot.")
    if set(row.get("artifact_sha256", {})) != set(OUTPUT_ARTIFACTS):
        raise ValueError("Incomplete test artifact evidence.")
    for name, digest in row["artifact_sha256"].items():
        if strict.file_hash(output / name) != digest:
            raise ValueError(f"Completed test artifact changed: {output / name}")
    metrics = strict.read_json(output / "test_metrics.json")
    expected = metric_row(label, row, metrics, output)
    for key in (
        "variant", "seed", "source_variant", "checkpoint_epoch",
        "fake_threshold", *MEASURES,
    ):
        if row.get(key) != expected[key]:
            raise ValueError(f"Completed test metric mismatch: {key}")
    metadata = strict.read_json(output / "evaluation_metadata.json")
    if metadata.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Evaluation metadata protocol mismatch.")
    return row


def summarize(rows):
    by_name = {row["variant"]: row for row in rows}
    if len(by_name) != len(rows) or any(name not in VARIANTS for name in by_name):
        raise ValueError("Duplicate or unplanned locked test result.")
    result = {
        "planned_evaluations": len(VARIANTS),
        "completed_evaluations": len(rows),
        "seed": SEED,
        "fake_threshold": FAKE_THRESHOLD,
        "primary_metric": "Test Fake/Real macro-F1 at fixed 0.55",
        "by_variant": {
            name: (
                {measure: by_name[name][measure] for measure in MEASURES}
                if name in by_name else None
            )
            for name in VARIANTS
        },
        "three_channel_full_minus_ablation": {},
        "note": (
            "One locked test split and one pre-specified seed; descriptive final "
            "test estimates, not a significance test. No test-based checkpoint or "
            "threshold selection was performed."
        ),
    }
    full = by_name.get("three_channel_full")
    if full is not None:
        for name in VARIANTS:
            if name == "three_channel_full" or name not in by_name:
                continue
            result["three_channel_full_minus_ablation"][name] = {
                measure: full[measure] - by_name[name][measure]
                for measure in MEASURES
            }
    return result


def write_summary(root, rows):
    strict.atomic_json(root / "locked_test_summary.json", summarize(rows))
    temporary = root / "locked_test_results.csv.tmp"
    columns = (
        "variant", "seed", "source_variant", "checkpoint_epoch",
        "fake_threshold", *MEASURES, "evaluation_dir",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, root / "locked_test_results.csv")


def evaluate_worker(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from dataset_cat import MVCNewsDatasetCAT
    from evaluate_cat import evaluate_cat
    from model_cat import MVCTTokenCATModel

    label = args.variant
    protocol = strict.verified_protocol(Path(args.suite_dir) / "protocol.json")
    if strict.file_hash(__file__) != protocol["entry_sha256"][Path(__file__).name]:
        raise ValueError("Locked-test evaluator changed after the protocol was sealed.")
    evidence = protocol["source_completion_evidence"][label]
    source_suite = Path(evidence["source_suite"])
    source_variant = evidence["source_variant"]
    completion_path = (
        source_suite / f"seed_{SEED}" / source_variant / "completed.json"
    )
    if strict.file_hash(completion_path) != evidence["completion_sha256"]:
        raise ValueError(f"Source completion changed: {completion_path}")
    source = strict.read_json(completion_path)
    checkpoint_path = Path(source["run_dir"]) / "best_checkpoint.pt"
    if str(checkpoint_path.resolve()) != evidence["checkpoint"]:
        raise ValueError("Source completion points to a different checkpoint.")
    if strict.file_hash(checkpoint_path) != evidence["checkpoint_sha256"]:
        raise ValueError(f"Source checkpoint changed: {checkpoint_path}")
    output = Path(args.output).resolve()
    slot = Path(args.suite_dir) / label
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid evaluation output or slot already completed.")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    config = SimpleNamespace(**checkpoint["config"])
    if float(config.FAKE_THRESHOLD) != FAKE_THRESHOLD:
        raise ValueError("Checkpoint threshold differs from locked 0.55.")
    mask = checkpoint["model_state_dict"].get("channel_mask")
    actual_mask = [bool(value) for value in mask.cpu().tolist()]
    if actual_mask != EXPECTED_MASKS[label]:
        raise ValueError(
            f"Unexpected channel mask for {label}: {actual_mask}; "
            f"expected {EXPECTED_MASKS[label]}"
        )

    frame, split_path, cache_dir, _, signature = restore_saved_split(config, "test")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BERT_MODEL, local_files_only=True, use_fast=True
    )
    dataset = MVCNewsDatasetCAT(
        frame, tokenizer, str(cache_dir), config.MAX_LEN_TITLE,
        config.MAX_LEN_TEXT, config.MAX_LEN_DESC,
    )
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=(
            config.SEM_WEIGHT, config.FACT_WEIGHT,
            config.LOGIC_WEIGHT, config.LOCAL_WEIGHT,
        ),
        cat_layers=config.CAT_LAYERS,
        cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER,
        dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT,
        learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=getattr(config, "MASK_MISSING_EVIDENCE", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(
        f"Locked test: {label}; device={device}; samples={len(dataset)}; "
        f"threshold={FAKE_THRESHOLD}", flush=True,
    )
    metrics_path = output / "test_metrics.json"
    metrics = evaluate_cat(
        model, loader, device, output_path=metrics_path,
        fake_threshold=FAKE_THRESHOLD,
    )
    metadata = {
        "protocol_id": protocol["protocol_id"],
        "variant": label,
        "source_variant": VARIANT_SOURCES[label][1],
        "seed": SEED,
        "split": "test",
        "saved_split": str(split_path.resolve()),
        "cache_signature": signature,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": source["artifact_sha256"]["best_checkpoint.pt"],
        "checkpoint_epoch": source["checkpoint_epoch"],
        "channel_mask": actual_mask,
        "fake_threshold": FAKE_THRESHOLD,
        "ai_decision_rule": "argmax",
        "n": len(dataset),
        "test_based_selection": False,
    }
    strict.atomic_json(output / "evaluation_metadata.json", metadata)
    row = metric_row(label, source, metrics, output)
    row.update({
        "protocol_id": protocol["protocol_id"],
        "artifact_sha256": {
            name: strict.file_hash(output / name) for name in OUTPUT_ARTIFACTS
        },
    })
    strict.atomic_json(output / "worker_result.json", row)
    del model, loader, dataset, tokenizer, frame
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Locked test worker completed:", output, flush=True)


def launch_child(arguments, lock, log=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *arguments]
    child = subprocess.Popen(
        command,
        cwd=str(SOURCE_DIR),
        env=strict.strict_environment(),
        stdout=log,
        stderr=subprocess.STDOUT if log else None,
        **lock.subprocess_options(),
    )
    try:
        code = child.wait()
    except KeyboardInterrupt:
        print("Interrupted; waiting for the test worker to exit safely.", flush=True)
        code = child.wait()
    if code:
        raise RuntimeError(
            f"Test worker exited with code {code}; no completion recorded. "
            "Inspect its worker.log."
        )


def protect_output(root, sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Locked-test output must not overlap a source suite.")


def main(argv=None):
    from config import cfg

    replication_root = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-suite",
        default=str(replication_root / "missing_evidence_v1"),
    )
    parser.add_argument(
        "--channel-suite",
        default=str(replication_root / "strict_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--three-suite",
        default=str(replication_root / "strict_three_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--suite-dir",
        default=str(replication_root / "locked_three_channel_test_v1"),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-new-evals", type=int, default=len(VARIANTS))
    parser.add_argument(
        "--worker", choices=("preflight", "evaluate"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_new_evals < 1:
        parser.error("--max-new-evals must be positive")

    root = Path(args.suite_dir).resolve()
    baseline = Path(args.baseline_suite).resolve()
    channel_suite = Path(args.channel_suite).resolve()
    three_suite = Path(args.three_suite).resolve()
    args.suite_dir = str(root)
    args.baseline_suite = str(baseline)
    args.channel_suite = str(channel_suite)
    args.three_suite = str(three_suite)

    if args.worker == "preflight":
        protocol, sources = prepare(baseline, channel_suite, three_suite)
        strict.atomic_json(args.output, {"protocol": protocol, "sources": sources})
        return
    if args.worker == "evaluate":
        evaluate_worker(args)
        return

    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(prefix="mvct_locked_test_preflight_") as temporary:
            result_file = Path(temporary) / "result.json"
            launch_child(
                [
                    "--worker", "preflight",
                    "--baseline-suite", str(baseline),
                    "--channel-suite", str(channel_suite),
                    "--three-suite", str(three_suite),
                    "--output", str(result_file),
                ],
                lock,
            )
            result = strict.read_json(result_file)

        protocol = result["protocol"]
        protect_output(root, (baseline, channel_suite, three_suite))
        if (root / "protocol.json").exists():
            strict.same_protocol(strict.verified_protocol(root / "protocol.json"), protocol)
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty locked-test suite has no protocol; use a new directory.")

        rows, pending = [], []
        for label in VARIANTS:
            row = completed_record(root / label, protocol, label)
            if row is None:
                pending.append(label)
            else:
                rows.append(row)
        planned = pending[:args.max_new_evals]
        print(
            "Preflight passed: five frozen checkpoints, saved test split, fixed "
            f"threshold 0.55; completed={len(rows)}/5, pending={len(pending)}.",
            flush=True,
        )
        print("Next evaluations:", planned, "\nOutput:", root, flush=True)
        if args.check_only:
            print("Check only: no test predictions or suite files written.")
            return

        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)
        write_summary(root, rows)

        for label in planned:
            slot = root / label
            output = slot / (
                "evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            output.mkdir(parents=True, exist_ok=False)
            print(f"Evaluate {label}; log: {output / 'worker.log'}", flush=True)
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child(
                    [
                        "--worker", "evaluate",
                        "--baseline-suite", str(baseline),
                        "--channel-suite", str(channel_suite),
                        "--three-suite", str(three_suite),
                        "--suite-dir", str(root),
                        "--variant", label,
                        "--output", str(output),
                    ],
                    lock,
                    log,
                )
            row = strict.read_json(output / "worker_result.json")
            completed_record(slot, protocol, label, record=row)
            strict.atomic_json(slot / "completed.json", row)
            rows.append(row)
            write_summary(root, rows)
            print(f"Completed {label}; total={len(rows)}/5", flush=True)

        if len(rows) == len(VARIANTS):
            print("Locked three-channel test completed:", root, flush=True)
        else:
            print("Invocation limit reached; repeat the same command to continue.", flush=True)


if __name__ == "__main__":
    main()
