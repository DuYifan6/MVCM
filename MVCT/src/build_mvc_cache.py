import argparse
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from config import cfg
from consistency_hybrid import ConsistencyBuilderHybrid
from dataset_cat import (
    _cache_key,
    _clean_and_group_dataframe,
    _compute_single_mvc,
)
from utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the versioned MVCM cache without loading BERT or training data loaders."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only build N samples")
    parser.add_argument(
        "--use-nli", action=argparse.BooleanOptionalAction, default=cfg.USE_NLI
    )
    parser.add_argument("--openie-workers", type=int, default=cfg.OPENIE_WORKERS)
    parser.add_argument("--sbert-batch-size", type=int, default=cfg.SBERT_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    Path(cfg.MVC_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("NLI enabled:", args.use_nli)
    print("OpenIE workers:", args.openie_workers)
    print("SBERT batch size:", args.sbert_batch_size)

    builder = ConsistencyBuilderHybrid(
        sbert_model=cfg.SBERT_MODEL,
        stanford_url=cfg.STANFORD_URL,
        nli_model=cfg.NLI_MODEL,
        use_nli=args.use_nli,
        nli_batch_size=cfg.NLI_BATCH_SIZE,
        nli_max_length=cfg.NLI_MAX_LENGTH,
        nli_contradiction_id=cfg.NLI_CONTRADICTION_ID,
        sbert_batch_size=args.sbert_batch_size,
        openie_workers=args.openie_workers,
        gpu_batch_auto_reduce=cfg.GPU_BATCH_AUTO_REDUCE,
        sem_weight=cfg.SEM_WEIGHT,
        fact_weight=cfg.FACT_WEIGHT,
        logic_weight=cfg.LOGIC_WEIGHT,
        local_weight=cfg.LOCAL_WEIGHT,
        logic_weights=(
            cfg.LOGIC_NLI_WEIGHT,
            cfg.LOGIC_NEGATION_WEIGHT,
            cfg.LOGIC_TEMPORAL_WEIGHT,
            cfg.LOGIC_NUMERIC_WEIGHT,
        ),
        alignment_threshold=cfg.LOGIC_ALIGNMENT_THRESHOLD,
        coverage_target=cfg.LOGIC_COVERAGE_TARGET,
        max_logic_pairs=cfg.MAX_LOGIC_PAIRS_PER_VIEW_PAIR,
        device=device,
    )
    frame = _clean_and_group_dataframe(
        cfg.DATA_PATH,
        drop_incomplete=cfg.DROP_INCOMPLETE,
        max_body_chars=cfg.MAX_BODY_CHARS,
        group_column=cfg.GROUP_COLUMN,
        drop_conflicting_duplicates=cfg.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=cfg.DROP_EXACT_DUPLICATES,
    )
    frame = frame.reset_index(drop=True)
    if args.limit is not None:
        frame = frame.iloc[: max(0, args.limit)].copy()
    rows = list(frame.itertuples(index=False))
    started = time.time()
    reused = 0
    built = 0
    failures = []
    try:
        for row in tqdm(rows, desc="Build MVC cache", unit="sample"):
            expected = Path(cfg.MVC_CACHE_DIR) / f"mvc_{_cache_key(row, builder)}.npy"
            existed = expected.exists()
            try:
                _compute_single_mvc(row, builder, cfg.MVC_CACHE_DIR)
                if existed:
                    reused += 1
                else:
                    built += 1
            except Exception as error:
                failures.append({"sample_id": int(row.ID), "error": repr(error)})
                tqdm.write(f"Failed sample {row.ID}: {error}")
    finally:
        builder.close()

    elapsed = time.time() - started
    manifest = {
        "builder_signature": builder.cache_signature,
        "host": socket.gethostname(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "config": asdict(cfg),
        "unique_samples": len(rows),
        "built": built,
        "reused": reused,
        "failed": len(failures),
        "elapsed_seconds": elapsed,
        "samples_per_second": (built + reused) / elapsed if elapsed else None,
        "failures": failures,
    }
    save_json(manifest, Path(cfg.MVC_CACHE_DIR) / "cache_manifest.json")
    print(
        f"Finished: built={built}, reused={reused}, failed={len(failures)}, "
        f"elapsed={elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
