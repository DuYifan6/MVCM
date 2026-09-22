"""Fresh-process, loss-selected, strict six-run MVCM confirmation suite.

Only this new entry writes the new suite. Frozen training/diagnostic code is reused.
The launcher is stdlib-only; strict environment is set before worker Python starts.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from importlib.metadata import version
from types import SimpleNamespace

from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs

SOURCE_DIR = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
VARIANTS = ("full", "without_mvcm")
MEASURES = ("fake_accuracy", "fake_macro_f1", "ai_accuracy", "ai_macro_f1", "val_loss")
STRICT_ENV = {
    "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(payload):
    return {**payload, "protocol_id": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def verified_protocol(path):
    saved = read_json(path)
    payload = {k: v for k, v in saved.items() if k != "protocol_id"}
    if seal(payload) != saved:
        raise ValueError(f"Protocol integrity failure: {path}; do not edit historical protocols.")
    return saved


def same_protocol(expected, actual):
    if expected != actual:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise ValueError("Protocol changed: " + ", ".join(changed) + ". Do not mix runs or edit protocol.json.")


def strict_environment():
    return {**os.environ, **STRICT_ENV}


def configure_strict():
    if any(os.environ.get(k) != v for k, v in STRICT_ENV.items()):
        raise RuntimeError("Worker must be launched through the strict entry, not directly.")
    from run_repro_check import configure_runtime
    configure_runtime("strict", fixture=False)


def validate_reference(config):
    if getattr(config, "MASK_MISSING_EVIDENCE", False):
        raise ValueError("Reference must be original full, not missing_evidence.")
    if config.FAKE_CLASS_WEIGHT is not None or config.FAKE_THRESHOLD != 0.55:
        raise ValueError("Expected original focal loss and fixed threshold 0.55.")


def run_config(reference, seed, variant, output):
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError("Only the six predeclared runs are allowed.")
    config = SimpleNamespace(**vars(reference))
    config.OUTPUT_DIR = str(output)
    config.DEVICE = "cuda"
    config.REPLICATION_SEED = seed
    config.EXPERIMENT_NAME = variant
    config.MASK_MISSING_EVIDENCE = False
    config.SAVE_BEST_CHECKPOINT = True
    config.SAVE_LAST_CHECKPOINT = False
    config.SAVE_EPOCH_CHECKPOINTS = False
    return config


def prepare(baseline, run_seeds=SEEDS):
    """Read-only full input preflight; never constructs caches or evaluates test."""
    import torch
    from run_repro_check import real_inputs
    from run_repeated_validation import validate_completion as validate_old

    configure_strict()
    old = verified_protocol(baseline / "protocol.json")
    if old["seeds"] != list(SEEDS) or old["variants"] != ["full", "without_mvcm", "missing_evidence"]:
        raise ValueError("Expected the original nine-run missing_evidence_v1 baseline.")
    for package, old_version in old["versions"].items():
        actual = str(torch.__version__) if package == "torch" else version(package)
        if actual != old_version:
            raise ValueError(f"Baseline dependency changed: {package}: {old_version} -> {actual}")
    print("Preflight: verifying original completions, files, splits, local model and caches...", flush=True)
    baseline_records = {}
    for seed in SEEDS:
        for variant in old["variants"]:
            done = baseline / f"seed_{seed}" / variant / "completed.json"
            record = validate_old(done, old["protocol_id"], variant, seed)
            baseline_records[f"{seed}/{variant}"] = {
                "completion": file_hash(done),
                "report": file_hash(Path(record["diagnostic_dir"]) / "validation_metrics.json"),
            }
    config, train, val, cache_paths, inventory = real_inputs(SimpleNamespace(suite_dir=str(baseline)))
    validate_reference(config)
    # Hash all training/validation cache bytes, not test inputs. Retain per-ID evidence.
    inventory.pop("cache_scope")
    inventory["train_val_cache_sha256"] = {str(k): file_hash(v) for k, v in sorted(cache_paths.items())}
    inventory["sample_counts"] = {"train": len(train), "validation": len(val)}
    if not len(train) or not len(val):
        raise ValueError("Empty training or validation split.")
    gpu = torch.cuda.get_device_properties(0)
    identity = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    payload = {
        "schema": 1, "purpose": "Strict confirmation of original loss-selected MVCM comparison",
        "seeds": list(run_seeds), "variants": list(VARIANTS), "split_seed": 42,
        "baseline_suite": str(baseline), "baseline_protocol_id": old["protocol_id"],
        "baseline_completion_evidence": baseline_records, "reference_config": old["reference_config"],
        "inventory": inventory,
        "entry_sha256": {name: file_hash(SOURCE_DIR / name) for name in
                         ("run_strict_replication.py", "run_repro_check.py", "runtime_guard.py", "config.py")},
        "runtime": {
            "python": sys.version, "executable": sys.executable,
            "packages": {name: version(name) for name in
                         ("torch", "numpy", "pandas", "transformers", "tokenizers", "scikit-learn", "sentence-transformers")},
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": gpu.name, "gpu_bytes": gpu.total_memory, "nvidia_smi": identity,
            "environment": {k: os.environ[k] for k in STRICT_ENV},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "selection": "Minimum combined validation task loss; improvement > 1e-6; original loss early stopping",
        "primary_metric": "Validation Fake/Real macro-F1 at fixed strict > 0.55; both tasks reported",
        "threshold_scan": "Existing diagnostic only; never used for selection or tuning in this suite",
        "test_evaluation": False, "initialization": "Fresh pretrained BERT, independent process per run",
    }
    checkpoint_size = max((Path(read_json(baseline / f"seed_{s}" / "full" / "completed.json")["run_dir"])
                           / "best_checkpoint.pt").stat().st_size for s in SEEDS)
    return seal(payload), config, train, val, checkpoint_size


def selected_loss_epoch(losses):
    best, epoch = math.inf, None
    for i, value in enumerate(losses, 1):
        if not math.isfinite(value):
            raise ValueError("Non-finite validation history.")
        if value < best - 1e-6:
            best, epoch = value, i
    if epoch is None:
        raise ValueError("No valid validation epoch.")
    return epoch, best


ARTIFACT_NAMES = ("best_checkpoint.pt", "history.json", "config.json", "run_identity.json")


def completed_record(slot, protocol, variant, seed, record=None):
    done = slot / "completed.json"
    if record is None and not done.exists():
        return None
    row = read_json(done) if record is None else record
    if (row.get("protocol_id"), row.get("variant"), row.get("seed")) != (protocol["protocol_id"], variant, seed):
        raise ValueError(f"Completed identity mismatch: {done}")
    output = Path(row["run_dir"]).resolve()
    if output.parent != slot.resolve() or not output.name.startswith("attempt_"):
        raise ValueError("Completion points outside its assigned slot.")
    if not row.get("loss_reproduced") or row.get("reference_threshold") != 0.55:
        raise ValueError("Completion has invalid loss reproduction or threshold.")
    diagnostic = Path(row["diagnostic_dir"]).resolve()
    if diagnostic.parent != output:
        raise ValueError("Diagnostic must belong to this attempt.")
    expected = set(ARTIFACT_NAMES) | {str((diagnostic / "validation_metrics.json").relative_to(output)),
                                    str((diagnostic / "validation_predictions.json").relative_to(output))}
    if set(row["artifact_sha256"]) != expected:
        raise ValueError("Incomplete artifact evidence.")
    for name, digest in row["artifact_sha256"].items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Completed artifact changed: {output / name}")
    report = read_json(diagnostic / "validation_metrics.json")
    fixed = report["at_reference_threshold"]
    original = {"checkpoint_epoch": report["checkpoint_epoch"],
                "reference_threshold": report["reference_threshold"],
                "fake_accuracy": fixed["fake_report"]["accuracy"],
                "fake_macro_f1": fixed["fake_report"]["macro avg"]["f1-score"],
                "ai_accuracy": fixed["ai_report"]["accuracy"],
                "ai_macro_f1": fixed["ai_report"]["macro avg"]["f1-score"],
                "val_loss": report["combined_task_loss"], "loss_reproduced": report["loss_reproduced"],
                "selected_threshold": report["selected_on_validation"]["threshold"],
                "selected_validation_fake_macro_f1": report["selected_on_validation"]["macro_f1"]}
    if any(row.get(k) != v for k, v in original.items()):
        raise ValueError("Completed metrics differ from diagnostic report.")
    epoch, loss = selected_loss_epoch(read_json(output / "history.json")["val_loss"])
    if epoch != row["checkpoint_epoch"] or not math.isclose(loss, row["val_loss"], abs_tol=1e-5, rel_tol=1e-4):
        raise ValueError("Completion did not use the original loss selector.")
    return row


def summarize(rows, planned_seeds=SEEDS):
    keys = [(r["seed"], r["variant"]) for r in rows]
    if len(set(keys)) != len(keys) or any(s not in planned_seeds or v not in VARIANTS for s, v in keys):
        raise ValueError("Duplicate or unplanned results.")
    def stats(values):
        return {"mean": statistics.mean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) > 1 else None}
    result = {"planned_runs": len(planned_seeds) * len(VARIANTS), "completed_runs": len(rows), "by_variant": {},
              "primary_metric": "Validation Fake/Real macro-F1 at fixed 0.55",
              "note": "Sample SD across three training seeds on one fixed validation split; not a confidence interval, significance test or final test result."}
    for variant in VARIANTS:
        group = [r for r in rows if r["variant"] == variant]
        result["by_variant"][variant] = {"n": len(group), "seeds": [r["seed"] for r in group],
                                         **{m: stats([r[m] for r in group]) for m in MEASURES}}
    baseline = {r["seed"]: r for r in rows if r["variant"] == "without_mvcm"}
    changes = [{"seed": r["seed"], **{m: r[m] - baseline[r["seed"]][m] for m in MEASURES}}
               for r in rows if r["variant"] == "full" and r["seed"] in baseline]
    result["paired_full_minus_without_mvcm"] = {
        "n_pairs": len(changes), "per_seed": changes,
        **{m: {**stats([r[m] for r in changes]), "positive_pairs": sum(r[m] > 0 for r in changes)} for m in MEASURES}}
    return result


def write_summary(root, rows, planned_seeds=SEEDS, extra_columns=()):
    atomic_json(root / "replication_summary.json", summarize(rows, planned_seeds))
    columns = ("seed", "variant", "checkpoint_epoch", "reference_threshold", *MEASURES, "run_dir", *extra_columns)
    temporary = root / "replication_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, root / "replication_results.csv")


def train_worker(args):
    import torch
    from model_cat import MVCTTokenCATModel
    from run_repeated_validation import make_loader, metrics_row, diagnose_validation
    from train_cat import train_model_cat
    from utils import set_seed

    protocol, reference, train, val, _ = prepare(Path(args.baseline_suite), args.seeds)
    same_protocol(verified_protocol(Path(args.suite_dir) / "protocol.json"), protocol)
    output = Path(args.output).resolve()
    slot = Path(args.suite_dir) / f"seed_{args.seed}" / args.variant
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid worker output or slot already completed.")
    config = run_config(reference, args.seed, args.variant, output)
    atomic_json(output / "config.json", vars(config))
    atomic_json(output / "run_identity.json", {"protocol_id": protocol["protocol_id"], "seed": args.seed, "variant": args.variant})
    set_seed(args.seed)
    train_loader = make_loader(train, config, args.seed, True)
    val_loader = make_loader(val, config, args.seed, False)
    weights = (0., 0., 0., 0.) if args.variant == "without_mvcm" else (
        config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT)
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL, mvc_weights=weights,
        cat_layers=config.CAT_LAYERS, cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER, dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT, learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=False).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    model, history = train_model_cat(model, train_loader, val_loader, config, optimizer, output_dir=str(output))
    del model, optimizer, train_loader, val_loader, train, val
    gc.collect()
    torch.cuda.empty_cache()
    report, diagnostic = diagnose_validation(["--run-dir", str(output)])
    epoch, loss = selected_loss_epoch(history["val_loss"])
    if not report["loss_reproduced"] or report["checkpoint_epoch"] != epoch or not math.isclose(
            loss, report["checkpoint_val_loss"], abs_tol=1e-10, rel_tol=0):
        raise RuntimeError("Loss-selected checkpoint failed verification; preserve this attempt and inspect logs.")
    row = metrics_row(args.variant, args.seed, report, diagnostic, output)
    names = [*ARTIFACT_NAMES, str((diagnostic / "validation_metrics.json").relative_to(output)),
             str((diagnostic / "validation_predictions.json").relative_to(output))]
    row.update(protocol_id=protocol["protocol_id"], artifact_sha256={name: file_hash(output / name) for name in names})
    # Supervisor commits completion only after successful worker exit and verification.
    atomic_json(output / "worker_result.json", row)
    print("Worker completed:", output, flush=True)


def check_root(root, baseline, protocol):
    if root == baseline or root in baseline.parents or baseline in root.parents:
        raise ValueError("New suite must not overlap the old baseline directory.")
    if (root / "protocol.json").exists():
        same_protocol(verified_protocol(root / "protocol.json"), protocol)
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty suite has no protocol. Use a new empty directory.")


def launch_child(arguments, lock, log=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *arguments]
    child = subprocess.Popen(command, cwd=str(SOURCE_DIR), env=strict_environment(),
                             stdout=log, stderr=subprocess.STDOUT if log else None, **lock.subprocess_options())
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # Linux child retains the lock if the supervisor is killed.
        print("Interrupted; waiting for worker exit before releasing protection.", flush=True)
        code = child.wait()
    if code:
        raise RuntimeError(f"Worker exited with code {code}; no completion recorded. Inspect its log.")


def main(argv=None):
    from config import cfg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-suite", default=str(Path(cfg.OUTPUT_DIR) / "replications/missing_evidence_v1"))
    parser.add_argument("--suite-dir", default=str(Path(cfg.OUTPUT_DIR) / "replications/strict_loss_confirmation_v1"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS),
                        help="Predeclared training seeds for a new, separate suite (default: 42 43 44)")
    parser.add_argument("--max-new-runs", type=int, default=2, help="Default 2: one paired seed per invocation")
    parser.add_argument("--worker", choices=("preflight", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.seeds = tuple(args.seeds)
    if tuple(sorted(set(args.seeds))) != args.seeds:
        parser.error("--seeds must be unique and in ascending order")
    if args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    # Resolve relative user paths against the calling directory before child cwd changes.
    root, baseline = Path(args.suite_dir).resolve(), Path(args.baseline_suite).resolve()
    args.suite_dir, args.baseline_suite = str(root), str(baseline)
    if args.worker == "preflight":
        protocol, _, _, _, size = prepare(baseline, args.seeds)
        atomic_json(args.output, {"protocol": protocol, "checkpoint_size": size})
        return
    if args.worker == "train":
        train_worker(args)
        return
    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(prefix="mvct_strict_preflight_") as temporary:
            result_file = Path(temporary) / "result.json"
            launch_child(["--worker", "preflight", "--baseline-suite", str(baseline),
                          "--seeds", *(str(seed) for seed in args.seeds), "--output", str(result_file)], lock)
            result = read_json(result_file)
        protocol = result["protocol"]
        check_root(root, baseline, protocol)
        rows = []
        pending = []
        for seed in args.seeds:
            for variant in VARIANTS:
                row = completed_record(root / f"seed_{seed}" / variant, protocol, variant, seed)
                if row is None:
                    pending.append((seed, variant))
                else:
                    rows.append(row)
        planned = pending[:args.max_new_runs]
        disk_path = root
        while not disk_path.exists():
            disk_path = disk_path.parent
        required = int(result["checkpoint_size"] * len(planned) * 1.2) + (2**30 if planned else 0)
        free = shutil.disk_usage(disk_path).free
        if free < required:
            raise RuntimeError(f"Need about {required / 2**30:.1f} GiB free for {len(planned)} runs; available {free / 2**30:.1f} GiB.")
        print(f"Preflight passed: strict, loss-selected, fixed 0.55; completed={len(rows)}/6, pending={len(pending)}.", flush=True)
        print("Next runs:", planned, "\nOutput:", root, flush=True)
        if args.check_only:
            print("Check only: no training or suite files written.")
            return
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            atomic_json(root / "protocol.json", protocol)
        write_summary(root, rows, args.seeds)
        for seed, variant in planned:
            slot = root / f"seed_{seed}" / variant
            output = slot / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            output.mkdir(parents=True, exist_ok=False)
            print(f"Run {variant}, seed={seed}; log: {output / 'worker.log'}", flush=True)
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child(["--worker", "train", "--baseline-suite", str(baseline), "--suite-dir", str(root),
                              "--seeds", *(str(value) for value in args.seeds), "--seed", str(seed),
                              "--variant", variant, "--output", str(output)], lock, log)
            row = read_json(output / "worker_result.json")
            completed_record(slot, protocol, variant, seed, record=row)
            atomic_json(slot / "completed.json", row)
            rows.append(row)
            write_summary(root, rows, args.seeds)
            print(f"Completed {variant}, seed={seed}; total={len(rows)}/6", flush=True)
        if len(rows) == len(args.seeds) * len(VARIANTS):
            print("Replication completed:", root, flush=True)
        else:
            print("Invocation limit reached; repeat the same command to continue.", flush=True)


if __name__ == "__main__":
    main()
