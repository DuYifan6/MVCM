"""Strict, single-seed MVCM channel-ablation suite.

This entry is intentionally separate from ``run_strict_replication.py`` so the
completed whole-module confirmation protocol remains immutable.  It runs six
fresh seed-42 trainings on one runtime block, selects checkpoints by the
original combined validation loss, evaluates validation at the fixed 0.55
threshold, and never evaluates the test split.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import run_strict_replication as strict
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEED = 42
VARIANT_WEIGHTS = {
    "full": (0.40, 0.25, 0.20, 0.15),
    "without_mvcm": (0.00, 0.00, 0.00, 0.00),
    "without_semantic": (0.00, 0.25, 0.20, 0.15),
    "without_factual": (0.40, 0.00, 0.20, 0.15),
    "without_logical": (0.40, 0.25, 0.00, 0.15),
    "without_local": (0.40, 0.25, 0.20, 0.00),
}
VARIANTS = tuple(VARIANT_WEIGHTS)
MEASURES = strict.MEASURES


def run_config(reference, variant, output):
    if variant not in VARIANTS:
        raise ValueError(f"Unplanned channel-ablation variant: {variant}")
    config = SimpleNamespace(**vars(reference))
    config.OUTPUT_DIR = str(output)
    config.DEVICE = "cuda"
    config.REPLICATION_SEED = SEED
    config.EXPERIMENT_NAME = variant
    config.MASK_MISSING_EVIDENCE = False
    config.MVCM_ABLATION_WEIGHTS = list(VARIANT_WEIGHTS[variant])
    config.SAVE_BEST_CHECKPOINT = True
    config.SAVE_LAST_CHECKPOINT = False
    config.SAVE_EPOCH_CHECKPOINTS = False
    return config


def prepare(baseline):
    """Run the frozen-input preflight and seal a channel-ablation protocol."""
    source_protocol, config, train, val, checkpoint_size = strict.prepare(
        baseline, (SEED,)
    )
    payload = {
        key: value for key, value in source_protocol.items()
        if key != "protocol_id"
    }
    entry_hashes = dict(payload["entry_sha256"])
    entry_hashes[Path(__file__).name] = strict.file_hash(__file__)
    payload.update({
        "purpose": "Strict seed-42 MVCM whole-module and channel ablation",
        "seeds": [SEED],
        "variants": list(VARIANTS),
        "entry_sha256": entry_hashes,
        "source_strict_preflight_protocol_id": source_protocol["protocol_id"],
        "ablation_weights": {
            name: list(weights) for name, weights in VARIANT_WEIGHTS.items()
        },
        "ablation_rule": (
            "Set the named channel's initialization weight to zero; the model "
            "stores the active-channel mask and renormalizes learnable weights "
            "over the remaining channels. Cached MVCM arrays are unchanged."
        ),
        "comparison_scope": (
            "Six fresh trainings with fixed seed 42 on one runtime block; "
            "full-minus-ablation validation differences are descriptive."
        ),
        "primary_metric": (
            "Validation Fake/Real macro-F1 at fixed strict > 0.55; "
            "both tasks and combined validation loss reported"
        ),
        "test_evaluation": False,
    })
    return strict.seal(payload), config, train, val, checkpoint_size


def completed_record(slot, protocol, variant, record=None):
    return strict.completed_record(slot, protocol, variant, SEED, record=record)


def summarize(rows):
    keys = [(row["seed"], row["variant"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate channel-ablation results.")
    if any(seed != SEED or variant not in VARIANTS for seed, variant in keys):
        raise ValueError("Unplanned channel-ablation result.")
    by_name = {row["variant"]: row for row in rows}
    summary = {
        "planned_runs": len(VARIANTS),
        "completed_runs": len(rows),
        "seed": SEED,
        "primary_metric": "Validation Fake/Real macro-F1 at fixed 0.55",
        "by_variant": {
            variant: (
                {measure: by_name[variant][measure] for measure in MEASURES}
                if variant in by_name else None
            )
            for variant in VARIANTS
        },
        "full_minus_ablation": {},
        "note": (
            "Single pre-specified seed 42 on one fixed validation split; "
            "descriptive ablation evidence, not a significance test or final "
            "test-set result. Positive metric deltas favor full; negative "
            "val_loss deltas favor full because lower loss is better."
        ),
    }
    full = by_name.get("full")
    if full is not None:
        for variant in VARIANTS:
            if variant == "full" or variant not in by_name:
                continue
            summary["full_minus_ablation"][variant] = {
                measure: full[measure] - by_name[variant][measure]
                for measure in MEASURES
            }
    return summary


def write_summary(root, rows):
    strict.atomic_json(root / "ablation_summary.json", summarize(rows))
    columns = (
        "seed", "variant", "checkpoint_epoch", "reference_threshold",
        *MEASURES, "selected_threshold", "selected_validation_fake_macro_f1",
        "run_dir",
    )
    temporary = root / "ablation_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, root / "ablation_results.csv")


def train_worker(args):
    import torch
    from model_cat import MVCTTokenCATModel
    from run_repeated_validation import make_loader, metrics_row, diagnose_validation
    from train_cat import train_model_cat
    from utils import set_seed

    protocol, reference, train, val, _ = prepare(Path(args.baseline_suite))
    strict.same_protocol(strict.verified_protocol(Path(args.suite_dir) / "protocol.json"), protocol)
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
        or not __import__("math").isclose(
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


def main(argv=None):
    from config import cfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-suite",
        default=str(Path(cfg.OUTPUT_DIR) / "replications/missing_evidence_v1"),
    )
    parser.add_argument(
        "--suite-dir",
        default=str(
            Path(cfg.OUTPUT_DIR)
            / "replications/strict_channel_ablation_seed42_v1"
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--max-new-runs", type=int, default=2,
        help="Maximum fresh variants in this invocation (default: 2)",
    )
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
    args.suite_dir, args.baseline_suite = str(root), str(baseline)
    if args.worker == "preflight":
        protocol, _, _, _, size = prepare(baseline)
        strict.atomic_json(
            args.output, {"protocol": protocol, "checkpoint_size": size}
        )
        return
    if args.worker == "train":
        train_worker(args)
        return

    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(
            prefix="mvct_strict_channel_preflight_"
        ) as temporary:
            result_file = Path(temporary) / "result.json"
            launch_child(
                ["--worker", "preflight", "--baseline-suite", str(baseline),
                 "--output", str(result_file)],
                lock,
            )
            result = strict.read_json(result_file)
        protocol = result["protocol"]
        strict.check_root(root, baseline, protocol)

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
            "Preflight passed: strict seed 42, six variants, loss-selected, "
            f"fixed 0.55; completed={len(rows)}/6, pending={len(pending)}.",
            flush=True,
        )
        print("Next runs:", planned, "\nOutput:", root, flush=True)
        if args.check_only:
            print("Check only: no training or suite files written.")
            return

        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)
        write_summary(root, rows)
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
                     "--suite-dir", str(root), "--variant", variant,
                     "--output", str(output)],
                    lock,
                    log,
                )
            row = strict.read_json(output / "worker_result.json")
            completed_record(slot, protocol, variant, record=row)
            strict.atomic_json(slot / "completed.json", row)
            rows.append(row)
            write_summary(root, rows)
            print(
                f"Completed {variant}, seed={SEED}; total={len(rows)}/6",
                flush=True,
            )
        if len(rows) == len(VARIANTS):
            print("Channel ablation completed:", root, flush=True)
        else:
            print(
                "Invocation limit reached; repeat the same command to continue.",
                flush=True,
            )


if __name__ == "__main__":
    main()
