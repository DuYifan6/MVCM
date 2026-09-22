"""One-factor missing-evidence experiment; evaluate validation only."""
import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from dataset_cat import load_and_split_with_mvc_cat
from diagnose_validation import main as diagnose_validation
from diagnose_validation import signature_from_config
from model_cat import MVCTTokenCATModel
from train_cat import train_model_cat
from utils import save_json, set_seed


def check_saved_split(dataset, split_path):
    saved = pd.read_csv(split_path, dtype={"_split_group": str})
    current = dataset.df
    for column in ("ID", "label_fake", "label_ai", "_split_group"):
        if current[column].astype(str).tolist() != saved[column].astype(str).tolist():
            raise ValueError(f"Saved split differs from reconstructed data: {split_path}/{column}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", default=str(Path(cfg.OUTPUT_DIR) / "ablation" / "full"))
    args = parser.parse_args(argv)
    reference = Path(args.reference_run)
    checkpoint_path = reference / "best_checkpoint.pt"
    # Read only this project's trusted checkpoint. Train a fresh model below.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = SimpleNamespace(**checkpoint["config"])
    del checkpoint
    gc.collect()
    if getattr(config, "MASK_MISSING_EVIDENCE", False):
        raise ValueError("Reference must be the original full model without missing-evidence masking.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Activate the GPU instance before training.")
    signature = signature_from_config(config)
    manifest_path = Path(config.MVC_CACHE_DIR) / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if signature != manifest["builder_signature"]:
        raise ValueError("Reference model and cache signatures differ; no caches will be rebuilt.")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL, local_files_only=True, use_fast=True)
    train_ds, val_ds, test_ds = load_and_split_with_mvc_cat(
        config.DATA_PATH, tokenizer, SimpleNamespace(cache_signature=signature),
        mvc_cache_dir=config.MVC_CACHE_DIR, precompute=False, random_state=42,
        max_len_title=config.MAX_LEN_TITLE, max_len_text=config.MAX_LEN_TEXT,
        max_len_desc=config.MAX_LEN_DESC, drop_incomplete=config.DROP_INCOMPLETE,
        max_body_chars=config.MAX_BODY_CHARS, group_column=config.GROUP_COLUMN,
        split_dir=None, drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES,
    )
    for name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        check_saved_split(dataset, Path(config.SPLIT_DIR) / f"{name}.csv")
    # Test membership is checked, but no test predictions/metrics are generated.
    del test_ds

    def loader(dataset, shuffle, workers):
        return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                          num_workers=workers, pin_memory=config.PIN_MEMORY,
                          persistent_workers=workers > 0)

    train_loader = loader(train_ds, True, config.TRAIN_NUM_WORKERS)
    val_loader = loader(val_ds, False, config.VAL_NUM_WORKERS)
    output = reference.parent / ("missing_evidence_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    config.OUTPUT_DIR = str(output)
    config.DEVICE = "cuda"
    config.MASK_MISSING_EVIDENCE = True
    config.SAVE_BEST_CHECKPOINT = True
    config.SAVE_LAST_CHECKPOINT = False
    save_json(vars(config), output / "config.json")
    save_json({
        "reference_checkpoint": str(checkpoint_path.resolve()), "seed": 42,
        "cache_signature": signature, "only_method_change": "mask_missing_evidence=True",
        "policy": "For factual approximately 0 AND logical approximately 0.5 (atol=1e-6), omit those channels and renormalize remaining active weights per view pair.",
        "evaluation": "Validation only. Original threshold, losses, bias strength and diagonal behavior retained.",
        "initialization": "Fresh pretrained BERT and fresh CAT/heads, seed 42, as in the ablation runner; not fine-tuning the reference checkpoint.",
    }, output / "experiment_protocol.json")
    print("Output directory:", output.resolve(), flush=True)
    print("MASK_MISSING_EVIDENCE: True; seed: 42; test evaluation: disabled", flush=True)
    set_seed(42)
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=(config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT),
        cat_layers=config.CAT_LAYERS, cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER, dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT, learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=True,
    ).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    model, _ = train_model_cat(model, train_loader, val_loader, config, optimizer, output_dir=str(output))
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    diagnose_validation(["--run-dir", str(output)])
    print("Missing-evidence experiment finished. Original caches and run outputs were preserved.")


if __name__ == "__main__":
    main()
