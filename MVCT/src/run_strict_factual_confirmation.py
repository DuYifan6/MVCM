"""Targeted strict confirmation of the factual MVCM channel.

The seed-42 channel-ablation suite produced an unexpected fixed-threshold
classification gain after removing the factual channel.  This separate suite
trains only ``without_factual`` for seeds 43 and 44, pairs those runs with the
already completed full runs on the same runtime block, and combines them with
the sealed seed-42 channel-ablation pair.  It never evaluates the test split.
"""
import argparse
import csv
import gc
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import run_strict_channel_ablation as channel
import run_strict_replication as strict
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEEDS = (43, 44)
VARIANT = "without_factual"
WEIGHTS = (0.40, 0.00, 0.20, 0.15)
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


def verify_compatible_channel_protocol(source_protocol, channel_protocol):
    changed = [
        key for key in COMPARABLE_PROTOCOL_FIELDS
        if source_protocol.get(key) != channel_protocol.get(key)
    ]
    core_entries = source_protocol.get("entry_sha256", {})
    channel_entries = channel_protocol.get("entry_sha256", {})
    changed_entries = [
        name for name, digest in core_entries.items()
        if channel_entries.get(name) != digest
    ]
    if changed or changed_entries:
        details = changed + [f"entry_sha256.{name}" for name in changed_entries]
        raise ValueError(
            "Seed-42 channel suite is not comparable to the current full "
            "reference suite: " + ", ".join(details)
        )


def run_config(reference, seed, output):
    if seed not in SEEDS:
        raise ValueError(f"Unplanned factual-confirmation seed: {seed}")
    config = SimpleNamespace(**vars(reference))
    config.OUTPUT_DIR = str(output)
    config.DEVICE = "cuda"
    config.REPLICATION_SEED = seed
    config.EXPERIMENT_NAME = VARIANT
    config.MASK_MISSING_EVIDENCE = False
    config.MVCM_ABLATION_WEIGHTS = list(WEIGHTS)
    config.SAVE_BEST_CHECKPOINT = True
    config.SAVE_LAST_CHECKPOINT = False
    config.SAVE_EPOCH_CHECKPOINTS = False
    return config


def prepare(baseline, full_suite, seed42_suite):
    """Validate frozen inputs plus all reused full/seed-42 evidence."""
    source_protocol, config, train, val, checkpoint_size = strict.prepare(
        baseline, SEEDS
    )
    saved_full_protocol = strict.verified_protocol(full_suite / "protocol.json")
    strict.same_protocol(saved_full_protocol, source_protocol)

    full_rows = {}
    for seed in SEEDS:
        row = strict.completed_record(
            full_suite / f"seed_{seed}" / "full",
            saved_full_protocol,
            "full",
            seed,
        )
        if row is None:
            raise ValueError(f"Missing completed full reference for seed {seed}.")
        full_rows[str(seed)] = row

    channel_protocol = strict.verified_protocol(seed42_suite / "protocol.json")
    verify_compatible_channel_protocol(source_protocol, channel_protocol)
    seed42_full = strict.completed_record(
        seed42_suite / "seed_42" / "full", channel_protocol, "full", 42
    )
    seed42_factual = strict.completed_record(
        seed42_suite / "seed_42" / VARIANT,
        channel_protocol,
        VARIANT,
        42,
    )
    if seed42_full is None or seed42_factual is None:
        raise ValueError("Seed-42 full/without_factual pair is incomplete.")

    payload = {
        key: value for key, value in source_protocol.items()
        if key != "protocol_id"
    }
    entry_hashes = dict(payload["entry_sha256"])
    entry_hashes[Path(__file__).name] = strict.file_hash(__file__)
    entry_hashes[Path(channel.__file__).name] = strict.file_hash(channel.__file__)
    payload.update({
        "purpose": "Targeted strict confirmation of the factual MVCM channel",
        "seeds": list(SEEDS),
        "variants": [VARIANT],
        "entry_sha256": entry_hashes,
        "full_reference_suite": str(full_suite),
        "full_reference_protocol_id": saved_full_protocol["protocol_id"],
        "full_reference_completion_sha256": {
            str(seed): strict.file_hash(
                full_suite / f"seed_{seed}" / "full" / "completed.json"
            )
            for seed in SEEDS
        },
        "seed42_channel_suite": str(seed42_suite),
        "seed42_channel_protocol_id": channel_protocol["protocol_id"],
        "seed42_completion_sha256": {
            "full": strict.file_hash(
                seed42_suite / "seed_42" / "full" / "completed.json"
            ),
            VARIANT: strict.file_hash(
                seed42_suite / "seed_42" / VARIANT / "completed.json"
            ),
        },
        "ablation_weights": list(WEIGHTS),
        "comparison_scope": (
            "Targeted post-ablation confirmation: seed 42 is reused from the "
            "sealed channel suite; without_factual is newly trained for seeds "
            "43 and 44 and paired with sealed full runs on the same runtime."
        ),
        "primary_metric": (
            "Validation Fake/Real macro-F1 at fixed strict > 0.55; "
            "full-minus-without_factual differences reported"
        ),
        "test_evaluation": False,
    })
    references = {
        "full": {"42": seed42_full, **full_rows},
        VARIANT: {"42": seed42_factual},
    }
    return strict.seal(payload), config, train, val, checkpoint_size, references


def completed_record(slot, protocol, seed, record=None):
    return strict.completed_record(
        slot, protocol, VARIANT, seed, record=record
    )


def _stats(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(new_rows, references):
    keys = [(row["seed"], row["variant"]) for row in new_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate factual-confirmation results.")
    if any(seed not in SEEDS or variant != VARIANT for seed, variant in keys):
        raise ValueError("Unplanned factual-confirmation result.")

    full = {int(seed): row for seed, row in references["full"].items()}
    factual = {
        int(seed): row for seed, row in references[VARIANT].items()
    }
    factual.update({row["seed"]: row for row in new_rows})
    paired_seeds = sorted(set(full) & set(factual))
    differences = [
        {
            "seed": seed,
            **{
                measure: full[seed][measure] - factual[seed][measure]
                for measure in MEASURES
            },
        }
        for seed in paired_seeds
    ]

    return {
        "planned_new_runs": len(SEEDS),
        "completed_new_runs": len(new_rows),
        "planned_pairs": 3,
        "completed_pairs": len(differences),
        "primary_metric": "Validation Fake/Real macro-F1 at fixed 0.55",
        "by_variant": {
            "full": {
                "n": len(paired_seeds),
                "seeds": paired_seeds,
                **{
                    measure: _stats([full[seed][measure] for seed in paired_seeds])
                    for measure in MEASURES
                },
            },
            VARIANT: {
                "n": len(paired_seeds),
                "seeds": paired_seeds,
                **{
                    measure: _stats([factual[seed][measure] for seed in paired_seeds])
                    for measure in MEASURES
                },
            },
        },
        "paired_full_minus_without_factual": {
            "n_pairs": len(differences),
            "per_seed": differences,
            **{
                measure: {
                    **_stats([row[measure] for row in differences]),
                    "positive_pairs": sum(
                        row[measure] > 0 for row in differences
                    ),
                }
                for measure in MEASURES
            },
        },
        "note": (
            "Targeted validation-only confirmation prompted by an unexpected "
            "seed-42 factual-channel result. It is descriptive, not a "
            "significance test or final test-set result. Positive metric "
            "deltas favor full; negative val_loss deltas favor full."
        ),
    }


def write_summary(root, new_rows, references):
    report = summarize(new_rows, references)
    strict.atomic_json(root / "factual_confirmation_summary.json", report)
    paired = report["paired_full_minus_without_factual"]["per_seed"]
    columns = ("seed", *MEASURES)
    temporary = root / "full_minus_without_factual.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired)
    os.replace(temporary, root / "full_minus_without_factual.csv")


def train_worker(args):
    import torch
    from model_cat import MVCTTokenCATModel
    from run_repeated_validation import make_loader, metrics_row, diagnose_validation
    from train_cat import train_model_cat
    from utils import set_seed

    protocol, reference, train, val, _, _ = prepare(
        Path(args.baseline_suite),
        Path(args.full_suite),
        Path(args.seed42_suite),
    )
    strict.same_protocol(
        strict.verified_protocol(Path(args.suite_dir) / "protocol.json"),
        protocol,
    )
    output = Path(args.output).resolve()
    slot = Path(args.suite_dir) / f"seed_{args.seed}" / VARIANT
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid worker output or slot already completed.")

    config = run_config(reference, args.seed, output)
    strict.atomic_json(output / "config.json", vars(config))
    strict.atomic_json(
        output / "run_identity.json",
        {"protocol_id": protocol["protocol_id"], "seed": args.seed,
         "variant": VARIANT},
    )
    set_seed(args.seed)
    train_loader = make_loader(train, config, args.seed, True)
    val_loader = make_loader(val, config, args.seed, False)
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=WEIGHTS,
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
    row = metrics_row(VARIANT, args.seed, report, diagnostic, output)
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


def _protect_output(root, sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Output suite must not overlap any source suite.")


def main(argv=None):
    from config import cfg

    replication_root = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-suite",
        default=str(replication_root / "missing_evidence_v1"),
    )
    parser.add_argument(
        "--full-suite",
        default=str(replication_root / "strict_loss_confirmation_host_b_v1"),
    )
    parser.add_argument(
        "--seed42-suite",
        default=str(replication_root / "strict_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--suite-dir",
        default=str(replication_root / "strict_factual_confirmation_v1"),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-new-runs", type=int, default=2)
    parser.add_argument(
        "--worker", choices=("preflight", "train"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")

    root = Path(args.suite_dir).resolve()
    baseline = Path(args.baseline_suite).resolve()
    full_suite = Path(args.full_suite).resolve()
    seed42_suite = Path(args.seed42_suite).resolve()
    args.suite_dir = str(root)
    args.baseline_suite = str(baseline)
    args.full_suite = str(full_suite)
    args.seed42_suite = str(seed42_suite)

    if args.worker == "preflight":
        protocol, _, _, _, size, references = prepare(
            baseline, full_suite, seed42_suite
        )
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
            prefix="mvct_strict_factual_preflight_"
        ) as temporary:
            result_file = Path(temporary) / "result.json"
            launch_child(
                ["--worker", "preflight", "--baseline-suite", str(baseline),
                 "--full-suite", str(full_suite), "--seed42-suite",
                 str(seed42_suite), "--output", str(result_file)],
                lock,
            )
            result = strict.read_json(result_file)

        protocol = result["protocol"]
        sources = (baseline, full_suite, seed42_suite)
        _protect_output(root, sources)
        if (root / "protocol.json").exists():
            strict.same_protocol(
                strict.verified_protocol(root / "protocol.json"), protocol
            )
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty suite has no protocol; use a new directory.")

        rows, pending = [], []
        for seed in SEEDS:
            row = completed_record(
                root / f"seed_{seed}" / VARIANT, protocol, seed
            )
            if row is None:
                pending.append(seed)
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
            "Preflight passed: targeted without_factual confirmation, "
            f"completed={len(rows)}/2, pending={len(pending)}.",
            flush=True,
        )
        print("Next seeds:", planned, "\nOutput:", root, flush=True)
        if args.check_only:
            print("Check only: no training or suite files written.")
            return

        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)
        write_summary(root, rows, result["references"])

        for seed in planned:
            slot = root / f"seed_{seed}" / VARIANT
            output = slot / (
                "attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            output.mkdir(parents=True, exist_ok=False)
            print(
                f"Run {VARIANT}, seed={seed}; log: {output / 'worker.log'}",
                flush=True,
            )
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child(
                    ["--worker", "train", "--baseline-suite", str(baseline),
                     "--full-suite", str(full_suite), "--seed42-suite",
                     str(seed42_suite), "--suite-dir", str(root), "--seed",
                     str(seed), "--output", str(output)],
                    lock,
                    log,
                )
            row = strict.read_json(output / "worker_result.json")
            completed_record(slot, protocol, seed, record=row)
            strict.atomic_json(slot / "completed.json", row)
            rows.append(row)
            write_summary(root, rows, result["references"])
            print(
                f"Completed {VARIANT}, seed={seed}; total={len(rows)}/2",
                flush=True,
            )

        if len(rows) == len(SEEDS):
            print("Factual confirmation completed:", root, flush=True)
        else:
            print(
                "Invocation limit reached; repeat the same command to continue.",
                flush=True,
            )


if __name__ == "__main__":
    main()
