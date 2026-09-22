"""Strict seed-42 ablation suite for the three-active-channel MVCM.

The frozen three-channel reference is the completed ``without_factual`` run:
semantic, logical, and local consistency are active, while factual alignment
remains available only to construct logical candidates.  This suite reuses the
sealed reference and w/o-MVCM rows, trains exactly three new leave-one-active-
channel-out variants, selects checkpoints by validation loss, and never reads
the test split.
"""
import argparse
import csv
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import run_strict_channel_ablation as channel
import run_strict_replication as strict
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEED = 42
REFERENCE_VARIANT = "without_factual"
BASELINE_VARIANT = "without_mvcm"
VARIANT_WEIGHTS = {
    "three_without_semantic": (0.00, 0.00, 0.20, 0.15),
    "three_without_logical": (0.40, 0.00, 0.00, 0.15),
    "three_without_local": (0.40, 0.00, 0.20, 0.00),
}
VARIANTS = tuple(VARIANT_WEIGHTS)
MEASURES = strict.MEASURES
COMPARABLE_PROTOCOL_FIELDS = (
    "baseline_protocol_id",
    "baseline_completion_evidence",
    "reference_config",
    "inventory",
    "runtime",
    "selection",
    "threshold_scan",
    "test_evaluation",
    "initialization",
)


def verify_source_protocol(source_protocol, channel_protocol):
    changed = [
        key for key in COMPARABLE_PROTOCOL_FIELDS
        if source_protocol.get(key) != channel_protocol.get(key)
    ]
    source_entries = source_protocol.get("entry_sha256", {})
    channel_entries = channel_protocol.get("entry_sha256", {})
    changed_entries = [
        name for name, digest in source_entries.items()
        if channel_entries.get(name) != digest
    ]
    if changed or changed_entries:
        details = changed + [f"entry_sha256.{name}" for name in changed_entries]
        raise ValueError(
            "The seed-42 source suite is not comparable to the current strict "
            "runtime: " + ", ".join(details)
        )


def run_config(reference, variant, output):
    if variant not in VARIANTS:
        raise ValueError(f"Unplanned three-channel ablation: {variant}")
    config = SimpleNamespace(**vars(reference))
    config.OUTPUT_DIR = str(output)
    config.DEVICE = "cuda"
    config.REPLICATION_SEED = SEED
    config.EXPERIMENT_NAME = variant
    config.MASK_MISSING_EVIDENCE = False
    config.MVCM_ABLATION_WEIGHTS = list(VARIANT_WEIGHTS[variant])
    config.ACTIVE_MVCM_CHANNELS = [
        name for name, weight in zip(
            ("semantic", "factual", "logical", "local"),
            VARIANT_WEIGHTS[variant],
        )
        if weight > 0
    ]
    config.SAVE_BEST_CHECKPOINT = True
    config.SAVE_LAST_CHECKPOINT = False
    config.SAVE_EPOCH_CHECKPOINTS = False
    return config


def prepare(baseline, source_suite):
    """Validate the frozen cache/config plus the two reused source rows."""
    source_protocol, config, train, val, checkpoint_size = strict.prepare(
        baseline, (SEED,)
    )
    saved_channel_protocol = strict.verified_protocol(
        source_suite / "protocol.json"
    )
    verify_source_protocol(source_protocol, saved_channel_protocol)

    reference = strict.completed_record(
        source_suite / f"seed_{SEED}" / REFERENCE_VARIANT,
        saved_channel_protocol,
        REFERENCE_VARIANT,
        SEED,
    )
    baseline_row = strict.completed_record(
        source_suite / f"seed_{SEED}" / BASELINE_VARIANT,
        saved_channel_protocol,
        BASELINE_VARIANT,
        SEED,
    )
    if reference is None or baseline_row is None:
        raise ValueError(
            "The source suite must contain completed seed-42 without_factual "
            "and without_mvcm rows."
        )

    payload = {
        key: value for key, value in source_protocol.items()
        if key != "protocol_id"
    }
    entry_hashes = dict(payload["entry_sha256"])
    entry_hashes[Path(__file__).name] = strict.file_hash(__file__)
    entry_hashes[Path(channel.__file__).name] = strict.file_hash(channel.__file__)
    payload.update({
        "purpose": "Strict seed-42 ablation of the three-active-channel MVCM",
        "seeds": [SEED],
        "variants": list(VARIANTS),
        "entry_sha256": entry_hashes,
        "source_channel_suite": str(source_suite),
        "source_channel_protocol_id": saved_channel_protocol["protocol_id"],
        "source_completion_sha256": {
            REFERENCE_VARIANT: strict.file_hash(
                source_suite / f"seed_{SEED}" / REFERENCE_VARIANT
                / "completed.json"
            ),
            BASELINE_VARIANT: strict.file_hash(
                source_suite / f"seed_{SEED}" / BASELINE_VARIANT
                / "completed.json"
            ),
        },
        "three_channel_reference_weights": [0.40, 0.00, 0.20, 0.15],
        "ablation_weights": {
            name: list(weights) for name, weights in VARIANT_WEIGHTS.items()
        },
        "ablation_rule": (
            "Factual fusion remains disabled in every variant. Exactly one of "
            "semantic, logical, or local is additionally masked, and active "
            "learnable weights are renormalized by the model. Cached MVCM "
            "arrays are unchanged."
        ),
        "comparison_scope": (
            "Single pre-specified seed 42 on one runtime block. The sealed "
            "without_factual run is the three-channel reference; the sealed "
            "without_mvcm run is the whole-module baseline."
        ),
        "primary_metric": (
            "Validation Fake/Real macro-F1 at fixed strict > 0.55; "
            "both tasks and combined validation loss reported"
        ),
        "test_evaluation": False,
    })
    references = {
        "three_channel_full": reference,
        BASELINE_VARIANT: baseline_row,
    }
    return strict.seal(payload), config, train, val, checkpoint_size, references


def completed_record(slot, protocol, variant, record=None):
    return strict.completed_record(
        slot, protocol, variant, SEED, record=record
    )


def summarize(new_rows, references):
    keys = [(row["seed"], row["variant"]) for row in new_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate three-channel ablation results.")
    if any(seed != SEED or variant not in VARIANTS for seed, variant in keys):
        raise ValueError("Unplanned three-channel ablation result.")

    all_rows = {
        "three_channel_full": references["three_channel_full"],
        BASELINE_VARIANT: references[BASELINE_VARIANT],
        **{row["variant"]: row for row in new_rows},
    }
    full = all_rows["three_channel_full"]
    comparisons = {}
    for variant, row in all_rows.items():
        if variant == "three_channel_full":
            continue
        comparisons[variant] = {
            measure: full[measure] - row[measure] for measure in MEASURES
        }

    return {
        "planned_new_runs": len(VARIANTS),
        "completed_new_runs": len(new_rows),
        "seed": SEED,
        "primary_metric": "Validation Fake/Real macro-F1 at fixed 0.55",
        "by_variant": {
            name: {
                measure: row[measure] for measure in MEASURES
            }
            for name, row in all_rows.items()
        },
        "three_channel_full_minus_ablation": comparisons,
        "channel_order": ["semantic", "logical", "local"],
        "note": (
            "Single pre-specified seed 42 on one fixed validation split; "
            "descriptive ablation evidence, not a significance test or final "
            "test-set result. Positive metric deltas favor the three-channel "
            "full model; negative val_loss deltas favor it."
        ),
    }


def write_summary(root, new_rows, references):
    report = summarize(new_rows, references)
    strict.atomic_json(root / "three_channel_ablation_summary.json", report)
    columns = ("variant", *MEASURES)
    temporary = root / "three_channel_ablation_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant, metrics in report["by_variant"].items():
            writer.writerow({"variant": variant, **metrics})
    os.replace(temporary, root / "three_channel_ablation_results.csv")


def train_worker(args):
    import torch
    from model_cat import MVCTTokenCATModel
    from run_repeated_validation import make_loader, metrics_row, diagnose_validation
    from train_cat import train_model_cat
    from utils import set_seed

    protocol, reference, train, val, _, _ = prepare(
        Path(args.baseline_suite), Path(args.source_suite)
    )
    strict.same_protocol(
        strict.verified_protocol(Path(args.suite_dir) / "protocol.json"),
        protocol,
    )
    output = Path(args.output).resolve()
    slot = Path(args.suite_dir) / f"seed_{SEED}" / args.variant
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid worker output or slot already completed.")

    config = run_config(reference, args.variant, output)
    strict.atomic_json(output / "config.json", vars(config))
    strict.atomic_json(
        output / "run_identity.json",
        {"protocol_id": protocol["protocol_id"], "seed": SEED,
         "variant": args.variant},
    )
    set_seed(SEED)
    train_loader = make_loader(train, config, SEED, True)
    val_loader = make_loader(val, config, SEED, False)
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=VARIANT_WEIGHTS[args.variant],
        cat_layers=config.CAT_LAYERS,
        cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER,
        dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT,
        learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=False,
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    model, history = train_model_cat(
        model, train_loader, val_loader, config, optimizer,
        output_dir=str(output),
    )
    del model, optimizer, train_loader, val_loader, train, val
    gc.collect()
    torch.cuda.empty_cache()

    report, diagnostic = diagnose_validation(["--run-dir", str(output)])
    epoch, loss = strict.selected_loss_epoch(history["val_loss"])
    if (
        not report["loss_reproduced"]
        or report["checkpoint_epoch"] != epoch
        or not math.isclose(
            loss, report["checkpoint_val_loss"], abs_tol=1e-10, rel_tol=0
        )
    ):
        raise RuntimeError(
            "Loss-selected checkpoint failed verification; preserve this "
            "attempt and inspect its worker.log."
        )
    row = metrics_row(args.variant, SEED, report, diagnostic, output)
    names = [
        *strict.ARTIFACT_NAMES,
        str((diagnostic / "validation_metrics.json").relative_to(output)),
        str((diagnostic / "validation_predictions.json").relative_to(output)),
    ]
    row.update(
        protocol_id=protocol["protocol_id"],
        artifact_sha256={name: strict.file_hash(output / name) for name in names},
    )
    strict.atomic_json(output / "worker_result.json", row)
    print("Worker completed:", output, flush=True)


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
        print(
            "Interrupted; waiting for worker exit before releasing protection.",
            flush=True,
        )
        code = child.wait()
    if code:
        raise RuntimeError(
            f"Worker exited with code {code}; no completion recorded. "
            "Inspect its worker.log."
        )


def protect_output(root, sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Output suite must not overlap a source suite.")


def main(argv=None):
    from config import cfg

    replication_root = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-suite",
        default=str(replication_root / "missing_evidence_v1"),
    )
    parser.add_argument(
        "--source-suite",
        default=str(replication_root / "strict_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--suite-dir",
        default=str(replication_root / "strict_three_channel_ablation_seed42_v1"),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-new-runs", type=int, default=3)
    parser.add_argument(
        "--worker", choices=("preflight", "train"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")

    root = Path(args.suite_dir).resolve()
    baseline = Path(args.baseline_suite).resolve()
    source_suite = Path(args.source_suite).resolve()
    args.suite_dir = str(root)
    args.baseline_suite = str(baseline)
    args.source_suite = str(source_suite)

    if args.worker == "preflight":
        protocol, _, _, _, size, references = prepare(baseline, source_suite)
        strict.atomic_json(
            args.output,
            {"protocol": protocol, "checkpoint_size": size,
             "references": references},
        )
        return
    if args.worker == "train":
        train_worker(args)
        return

    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(
            prefix="mvct_strict_three_channel_preflight_"
        ) as temporary:
            result_file = Path(temporary) / "result.json"
            launch_child(
                ["--worker", "preflight", "--baseline-suite", str(baseline),
                 "--source-suite", str(source_suite), "--output",
                 str(result_file)],
                lock,
            )
            result = strict.read_json(result_file)

        protocol = result["protocol"]
        protect_output(root, (baseline, source_suite))
        if (root / "protocol.json").exists():
            strict.same_protocol(
                strict.verified_protocol(root / "protocol.json"), protocol
            )
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty suite has no protocol; use a new directory.")

        rows, pending = [], []
        for variant in VARIANTS:
            row = completed_record(
                root / f"seed_{SEED}" / variant, protocol, variant
            )
            if row is None:
                pending.append(variant)
            else:
                rows.append(row)
        planned = pending[:args.max_new_runs]

        disk_path = root
        while not disk_path.exists():
            disk_path = disk_path.parent
        required = (
            int(result["checkpoint_size"] * len(planned) * 1.2)
            + (2**30 if planned else 0)
        )
        free = shutil.disk_usage(disk_path).free
        if free < required:
            raise RuntimeError(
                f"Need about {required / 2**30:.1f} GiB free for "
                f"{len(planned)} runs; available {free / 2**30:.1f} GiB."
            )

        print(
            "Preflight passed: strict three-active-channel seed-42 ablation, "
            f"completed={len(rows)}/3, pending={len(pending)}.",
            flush=True,
        )
        print("Next variants:", planned, "\nOutput:", root, flush=True)
        if args.check_only:
            print("Check only: no training or suite files written.")
            return

        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)
        write_summary(root, rows, result["references"])

        for variant in planned:
            slot = root / f"seed_{SEED}" / variant
            output = slot / (
                "attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            output.mkdir(parents=True, exist_ok=False)
            print(
                f"Run {variant}, seed={SEED}; log: {output / 'worker.log'}",
                flush=True,
            )
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child(
                    ["--worker", "train", "--baseline-suite", str(baseline),
                     "--source-suite", str(source_suite), "--suite-dir",
                     str(root), "--variant", variant, "--output", str(output)],
                    lock,
                    log,
                )
            row = strict.read_json(output / "worker_result.json")
            completed_record(slot, protocol, variant, record=row)
            strict.atomic_json(slot / "completed.json", row)
            rows.append(row)
            write_summary(root, rows, result["references"])
            print(
                f"Completed {variant}, seed={SEED}; total={len(rows)}/3",
                flush=True,
            )

        if len(rows) == len(VARIANTS):
            print("Three-channel ablation completed:", root, flush=True)
        else:
            print(
                "Invocation limit reached; repeat the same command to continue.",
                flush=True,
            )


if __name__ == "__main__":
    main()
