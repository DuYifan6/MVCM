"""Paired three-seed replication of MVCM variants, with validation-only reporting."""
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from dataset_cat import load_and_split_with_mvc_cat
from diagnose_validation import main as diagnose_validation, signature_from_config
from model_cat import MVCTTokenCATModel
from run_missing_evidence import check_saved_split
from train_cat import train_model_cat
from utils import set_seed

SEEDS = (42, 43, 44)
VARIANTS = ("full", "without_mvcm", "missing_evidence")
MEASURES = ("fake_accuracy", "fake_macro_f1", "ai_accuracy", "ai_macro_f1", "val_loss")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, config, seed, train):
    workers = config.TRAIN_NUM_WORKERS if train else config.VAL_NUM_WORKERS
    # Independent generators give paired variants identical sample ordering,
    # regardless of model initialization or previous runs' worker lifetimes.
    generator = torch.Generator().manual_seed(seed + (0 if train else 100000))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train,
                      num_workers=workers, pin_memory=config.PIN_MEMORY,
                      persistent_workers=workers > 0, generator=generator,
                      worker_init_fn=seed_worker)


def summarize(rows):
    summary = {"planned_runs": len(SEEDS) * len(VARIANTS), "completed_runs": len(rows),
               "primary_metric": "Validation Fake/Real macro-F1 at the fixed reference threshold",
               "note": "Sample SD across training seeds on one fixed split; not a confidence interval or independent test-set confirmation.",
               "by_variant": {}, "paired_vs_without_mvcm": {}}
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        entry = {"n": len(group), "seeds": [r["seed"] for r in group]}
        for name in MEASURES:
            values = [r[name] for r in group]
            entry[name] = {"mean": float(np.mean(values)) if values else None,
                           "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        summary["by_variant"][variant] = entry
    baseline = {r["seed"]: r for r in rows if r["variant"] == "without_mvcm"}
    for variant in ("full", "missing_evidence"):
        paired = [r for r in rows if r["variant"] == variant and r["seed"] in baseline]
        changes = [{"seed": r["seed"], **{name: r[name] - baseline[r["seed"]][name] for name in MEASURES}} for r in paired]
        summary["paired_vs_without_mvcm"][variant] = {
            "n_pairs": len(changes), "per_seed_difference": changes,
            "fake_macro_f1_mean_difference": float(np.mean([r["fake_macro_f1"] for r in changes])) if changes else None,
            "fake_macro_f1_positive_pairs": sum(r["fake_macro_f1"] > 0 for r in changes),
        }
    return summary


def metrics_row(variant, seed, report, diagnostic_dir, run_dir):
    fixed = report["at_reference_threshold"]
    return {
        "variant": variant, "seed": seed, "checkpoint_epoch": report["checkpoint_epoch"],
        "reference_threshold": report["reference_threshold"],
        "fake_accuracy": fixed["fake_report"]["accuracy"],
        "fake_macro_f1": fixed["fake_report"]["macro avg"]["f1-score"],
        "ai_accuracy": fixed["ai_report"]["accuracy"],
        "ai_macro_f1": fixed["ai_report"]["macro avg"]["f1-score"],
        "val_loss": report["combined_task_loss"],
        "selected_threshold": report["selected_on_validation"]["threshold"],
        "selected_validation_fake_macro_f1": report["selected_on_validation"]["macro_f1"],
        "loss_reproduced": report["loss_reproduced"],
        "diagnostic_dir": str(diagnostic_dir), "run_dir": str(run_dir),
    }


def validate_completion(path, expected_id, variant, seed):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("protocol_id") != expected_id or record.get("variant") != variant or record.get("seed") != seed:
        raise ValueError(f"Completed run has different protocol/identity: {path}")
    if not record.get("loss_reproduced"):
        raise ValueError(f"Completed run failed loss reproduction: {path}")
    if not (Path(record["run_dir"]) / "best_checkpoint.pt").is_file():
        raise FileNotFoundError(f"Completed run checkpoint missing: {path}")
    if not (Path(record["diagnostic_dir"]) / "validation_metrics.json").is_file():
        raise FileNotFoundError(f"Completed validation output missing: {path}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", default=str(Path(cfg.OUTPUT_DIR) / "ablation" / "full"))
    parser.add_argument("--suite-dir", default=str(Path(cfg.OUTPUT_DIR) / "replications" / "missing_evidence_v1"))
    parser.add_argument("--max-new-runs", type=int, default=None, help="Optional cap on additional complete runs in this invocation")
    args = parser.parse_args(argv)
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; use GPU mode before starting replication training.")
    checkpoint_path = Path(args.reference_run) / "best_checkpoint.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    config = SimpleNamespace(**checkpoint["config"])
    if not bool(checkpoint["model_state_dict"]["channel_mask"].all()):
        raise ValueError("Reference must have all four MVC channels enabled.")
    del checkpoint
    if getattr(config, "MASK_MISSING_EVIDENCE", False):
        raise ValueError("Reference must be the original full model.")
    if config.FAKE_CLASS_WEIGHT is not None:
        raise ValueError("This frozen replication protocol expects the original unweighted focal configuration.")
    signature = signature_from_config(config)
    manifest_path = Path(config.MVC_CACHE_DIR) / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if signature != manifest["builder_signature"]:
        raise ValueError("Reference configuration and existing caches do not match.")

    source_dir = Path(__file__).resolve().parent
    protocol = {
        "seeds": list(SEEDS), "variants": list(VARIANTS), "split_seed": 42,
        "reference_checkpoint": str(checkpoint_path.resolve()), "reference_config": vars(config),
        "cache_signature": signature,
        "dataset_sha256": file_hash(config.DATA_PATH),
        "split_sha256": {name: file_hash(Path(config.SPLIT_DIR) / f"{name}.csv") for name in ("train", "val", "test")},
        "source_sha256": {name: file_hash(source_dir / name) for name in (
            "run_repeated_validation.py", "model_cat.py", "train_cat.py", "dataset_cat.py",
            "diagnose_validation.py", "run_missing_evidence.py", "consistency_hybrid.py", "utils.py")},
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
        "selection": "Best combined validation task loss; existing early stopping retained.",
        "primary_metric": "Validation Fake/Real macro-F1 at the reference threshold; report both tasks.",
        "threshold_scan": "Secondary descriptive output only; not used to select a run or alter config.",
        "test_evaluation": False, "loader": "Fresh per-run seeded generators and workers; fixed split, paired training seeds.",
    }
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    root = Path(args.suite_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_id"] != protocol_id:
            raise ValueError("This suite directory belongs to a different code/data/config protocol. Use a new --suite-dir.")
    else:
        if any(root.iterdir()):
            raise ValueError("Suite directory is non-empty but has no protocol; use an empty directory.")
        atomic_json({"protocol_id": protocol_id, **protocol}, protocol_path)

    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            done = root / f"seed_{seed}" / variant / "completed.json"
            if done.is_file():
                rows.append(validate_completion(done, protocol_id, variant, seed))

    def write_summary():
        result = summarize(rows)
        atomic_json(result, root / "replication_summary.json")
        pd.DataFrame(rows).to_csv(root / "replication_results.csv", index=False)
        return result

    write_summary()
    pending = len(SEEDS) * len(VARIANTS) - len(rows)
    if not pending:
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        print("All planned runs already completed:", root.resolve())
        return
    planned_now = min(pending, args.max_new_runs or pending)
    required = int(checkpoint_path.stat().st_size * planned_now * 1.15)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f"Insufficient free data-disk space: reserve about {required / 2**30:.1f} GiB for {planned_now} checkpoints. Use --max-new-runs to limit this invocation.")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL, local_files_only=True, use_fast=True)
    train_ds, val_ds, test_ds = load_and_split_with_mvc_cat(
        config.DATA_PATH, tokenizer, SimpleNamespace(cache_signature=signature),
        mvc_cache_dir=config.MVC_CACHE_DIR, precompute=False, random_state=42,
        max_len_title=config.MAX_LEN_TITLE, max_len_text=config.MAX_LEN_TEXT, max_len_desc=config.MAX_LEN_DESC,
        drop_incomplete=config.DROP_INCOMPLETE, max_body_chars=config.MAX_BODY_CHARS,
        group_column=config.GROUP_COLUMN, split_dir=None,
        drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES,
    )
    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        check_saved_split(dataset, Path(config.SPLIT_DIR) / f"{name}.csv")
    del test_ds
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    completed_now = 0
    for seed in SEEDS:
        for variant in VARIANTS:
            slot = root / f"seed_{seed}" / variant
            done = slot / "completed.json"
            if done.exists():
                print(f"Skip completed: {variant}, seed={seed}", flush=True)
                continue
            if args.max_new_runs is not None and completed_now >= args.max_new_runs:
                print("Invocation limit reached; repeat the same command to continue.", flush=True)
                return
            # Preserve interrupted attempts; restarting does not overwrite them.
            output = slot / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            output.mkdir(parents=True, exist_ok=False)
            run_cfg = copy.deepcopy(config)
            run_cfg.OUTPUT_DIR = str(output)
            run_cfg.DEVICE = "cuda"
            run_cfg.REPLICATION_SEED = seed
            run_cfg.EXPERIMENT_NAME = variant
            run_cfg.MASK_MISSING_EVIDENCE = variant == "missing_evidence"
            run_cfg.SAVE_BEST_CHECKPOINT = True
            run_cfg.SAVE_LAST_CHECKPOINT = False
            run_cfg.SAVE_EPOCH_CHECKPOINTS = False
            atomic_json(vars(run_cfg), output / "config.json")
            atomic_json({"protocol_id": protocol_id, "seed": seed, "variant": variant}, output / "run_identity.json")
            print(f"\nRun {variant}, seed={seed}; output={output.resolve()}", flush=True)
            set_seed(seed)
            train_loader = make_loader(train_ds, run_cfg, seed, True)
            val_loader = make_loader(val_ds, run_cfg, seed, False)
            weights = (0., 0., 0., 0.) if variant == "without_mvcm" else (
                config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT)
            model = MVCTTokenCATModel(
                bert_model_name=config.BERT_MODEL, mvc_weights=weights,
                cat_layers=config.CAT_LAYERS, cat_heads=config.CAT_HEADS,
                cat_ff_multiplier=config.CAT_FF_MULTIPLIER, dropout_p=config.CAT_DROPOUT,
                lambda_init=config.CAT_LAMBDA_INIT, learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
                mask_missing_evidence=run_cfg.MASK_MISSING_EVIDENCE,
            ).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
            model, _ = train_model_cat(model, train_loader, val_loader, run_cfg, optimizer, output_dir=str(output))
            del model, optimizer, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            report, diagnostic_dir = diagnose_validation(["--run-dir", str(output)])
            if not report["loss_reproduced"]:
                raise RuntimeError(f"Checkpoint loss did not reproduce. Inspect {diagnostic_dir}; suite stopped before further runs.")
            row = metrics_row(variant, seed, report, diagnostic_dir, output)
            row["protocol_id"] = protocol_id
            atomic_json(row, done)
            rows.append(row)
            completed_now += 1
            write_summary()
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    print("Replication completed:", root.resolve())


if __name__ == "__main__":
    main()
