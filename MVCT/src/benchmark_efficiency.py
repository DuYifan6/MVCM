"""Benchmark frozen MVCT, w/o-MVCM, and RoBERTa artifacts.

The benchmark separates offline MVC cache construction/storage from warm-cache
end-to-end inference.  It never trains a model, selects a checkpoint, scans a
threshold, or changes a prediction.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_locked_three_channel_test as locked
import run_strict_baselines_seed42 as baselines
import run_strict_replication as strict


MAIN_VARIANTS = ("three_channel_full", "without_mvcm")


def percentile(values, quantile):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def validated_omp_threads():
    value = os.environ.get("OMP_NUM_THREADS")
    try:
        threads = int(value) if value is not None else None
    except ValueError as error:
        raise ValueError(
            "OMP_NUM_THREADS must be a positive integer before benchmarking"
        ) from error
    if threads is None or threads <= 0:
        raise ValueError(
            "Set OMP_NUM_THREADS to a positive integer before benchmarking, "
            "for example: export OMP_NUM_THREADS=4"
        )
    return threads


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


def cache_profile(cache_dir):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = strict.read_json(manifest_path) if manifest_path.is_file() else {}
    cache_files = list(cache_dir.glob("mvc_*.npy"))
    total_bytes = sum(path.stat().st_size for path in cache_files)
    built = int(manifest.get("built", 0))
    reused = int(manifest.get("reused", 0))
    unique = int(manifest.get("unique_samples", 0))
    elapsed = manifest.get("elapsed_seconds")
    cold_build_valid = bool(unique and built == unique and reused == 0)
    return {
        "cache_dir": str(cache_dir.resolve()),
        "cache_manifest": str(manifest_path.resolve()) if manifest_path.is_file() else None,
        "cache_files": len(cache_files),
        "cache_bytes": total_bytes,
        "cache_gib": total_bytes / (1024 ** 3),
        "manifest_unique_samples": unique,
        "manifest_built": built,
        "manifest_reused": reused,
        "manifest_failed": int(manifest.get("failed", 0)),
        "manifest_elapsed_seconds": elapsed,
        "manifest_seconds_per_built_sample": (
            float(elapsed) / built if elapsed is not None and built else None
        ),
        "manifest_cold_build_valid": cold_build_valid,
        "note": (
            "Elapsed time represents a clean offline build only when "
            "manifest_cold_build_valid is true."
        ),
    }


def load_main_variant(main_suite, variant):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from dataset_cat import MVCNewsDatasetCAT
    from model_cat import MVCTTokenCATModel

    protocol = strict.verified_protocol(Path(main_suite) / "protocol.json")
    evidence = protocol["source_completion_evidence"][variant]
    checkpoint_path = Path(evidence["checkpoint"])
    if strict.file_hash(checkpoint_path) != evidence["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint digest mismatch: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    config = SimpleNamespace(**checkpoint["config"])
    frame, _, cache_dir, _, _ = locked.restore_saved_split(config, "test")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BERT_MODEL, local_files_only=True, use_fast=True
    )
    dataset = MVCNewsDatasetCAT(
        frame, tokenizer, str(cache_dir), config.MAX_LEN_TITLE,
        config.MAX_LEN_TEXT, config.MAX_LEN_DESC,
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

    def loader_factory(batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return model, loader_factory, checkpoint_path, cache_profile(cache_dir), len(dataset)


def load_roberta(baseline_suite):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from baseline_models import BaselineNewsDataset

    baseline_suite = Path(baseline_suite)
    protocol = strict.verified_protocol(baseline_suite / "protocol.json")
    record = baselines.training_record(
        baseline_suite / "roberta", protocol, "roberta"
    )
    output_dir = Path(record["output_dir"])
    checkpoint_path = output_dir / "best_checkpoint.pt"
    config = SimpleNamespace(**strict.read_json(output_dir / "config.json"))
    spec, model_path = baselines.model_paths(protocol, "roberta")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    frame, _ = baselines.restore_frame(config, "test")
    dataset = BaselineNewsDataset(
        frame, tokenizer, config.MAX_LEN_TITLE,
        config.MAX_LEN_TEXT, config.MAX_LEN_DESC,
    )
    model = baselines.build_neural("roberta", protocol, tokenizer)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint

    def loader_factory(batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return model, loader_factory, checkpoint_path, None, len(dataset)


def move_and_forward(model, batch, kind, device):
    if kind == "main":
        sequence = {
            key: value.to(device, non_blocking=False)
            for key, value in batch["sequence"].items()
        }
        mvc = batch["mvc"].float().to(device, non_blocking=False)
        model(sequence, mvc)
    else:
        model(
            batch["input_ids"].to(device, non_blocking=False),
            batch["attention_mask"].to(device, non_blocking=False),
        )


def benchmark_one(
    model, loader_factory, kind, variant, checkpoint_path, device,
    batch_size, warmup_batches, measured_batches, repeats, test_samples,
):
    import torch

    model.to(device).eval()
    total_parameters, trainable_parameters = parameter_counts(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    measured_samples = 0
    with torch.inference_mode():
        for _ in range(repeats):
            iterator = iter(loader_factory(batch_size))
            for _ in range(warmup_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                move_and_forward(model, batch, kind, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(measured_batches):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                move_and_forward(model, batch, kind, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                current_batch = int(batch["sample_id"].shape[0])
                timings.append((elapsed, current_batch))
                measured_samples += current_batch

    if not timings:
        raise RuntimeError("No timed batches were produced")
    per_article_ms = [elapsed * 1000.0 / count for elapsed, count in timings]
    total_seconds = sum(elapsed for elapsed, _ in timings)
    allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    return {
        "variant": variant,
        "device": str(device),
        "batch_size": batch_size,
        "test_samples": test_samples,
        "warmup_batches_per_repeat": warmup_batches,
        "measured_batches_requested_per_repeat": measured_batches,
        "repeats": repeats,
        "timed_batches": len(timings),
        "timed_samples": measured_samples,
        "median_ms_per_article": statistics.median(per_article_ms),
        "p95_ms_per_article": percentile(per_article_ms, 95),
        "throughput_articles_per_second": measured_samples / total_seconds,
        "peak_allocated_gib": allocated / (1024 ** 3) if allocated is not None else None,
        "peak_reserved_gib": reserved / (1024 ** 3) if reserved is not None else None,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
        "checkpoint_mib": Path(checkpoint_path).stat().st_size / (1024 ** 2),
        "timing_scope": (
            "warm-cache end-to-end batch retrieval, tokenization/cache read, "
            "host-to-device transfer, and frozen forward pass; model loading excluded"
        ),
    }


def write_results(output_dir, rows, offline_cache, arguments, device, omp_threads):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "efficiency_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "analysis": "Frozen-model efficiency and resource profiling",
        "selection_or_training_performed": False,
        "arguments": vars(arguments),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_mib": (
                torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
                if device.type == "cuda" else None
            ),
            "gpu_driver": arguments.driver_version,
            "omp_num_threads": omp_threads,
        },
        "offline_mvc_cache": offline_cache,
        "online_inference": rows,
    }
    json_path = output_dir / "efficiency_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-suite", required=True)
    parser.add_argument("--baseline-suite", required=True)
    parser.add_argument("--output-dir", default="outputs/p0_efficiency")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measured-batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--driver-version", default=None)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.batch_sizes):
        raise ValueError("Batch sizes must be positive")
    if min(args.warmup_batches, args.measured_batches, args.repeats) < 1:
        raise ValueError("Warmup, measured batches, and repeats must be positive")

    omp_threads = validated_omp_threads()
    import torch

    torch.set_num_threads(omp_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    offline_cache = None
    loaders = [
        ("three_channel_full", "main", lambda: load_main_variant(args.main_suite, "three_channel_full")),
        ("without_mvcm", "main", lambda: load_main_variant(args.main_suite, "without_mvcm")),
        ("roberta", "baseline", lambda: load_roberta(args.baseline_suite)),
    ]
    for variant, kind, load in loaders:
        model, loader_factory, checkpoint, cache, test_samples = load()
        if cache is not None and offline_cache is None:
            offline_cache = cache
        for batch_size in args.batch_sizes:
            print(f"Benchmarking {variant}, batch_size={batch_size}", flush=True)
            rows.append(benchmark_one(
                model, loader_factory, kind, variant, checkpoint, device,
                batch_size, args.warmup_batches, args.measured_batches,
                args.repeats, test_samples,
            ))
        del model, loader_factory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path, json_path = write_results(
        args.output_dir, rows, offline_cache, args, device, omp_threads
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
