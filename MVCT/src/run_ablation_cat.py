import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from dataset_cat import load_and_split_with_mvc_cat
from evaluate_cat import evaluate_cat
from model_cat import MVCTTokenCATModel
from train_cat import train_model_cat
from utils import set_seed


EXPERIMENTS = {
    "full": (0.40, 0.25, 0.20, 0.15),
    "without_mvcm": (0.00, 0.00, 0.00, 0.00),
    "without_semantic": (0.00, 0.25, 0.20, 0.15),
    "without_factual": (0.40, 0.00, 0.20, 0.15),
    "without_logical": (0.40, 0.25, 0.00, 0.15),
    "without_local": (0.40, 0.25, 0.20, 0.00),
}


def _load_cache_signature():
    manifest_path = Path(cfg.MVC_CACHE_DIR) / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cache manifest is missing: {manifest_path}. Run build_mvc_cache.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["builder_signature"]


def _loader(dataset, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=workers > 0,
    )


def _summary(name, metrics):
    fake = metrics["fake_report"]
    ai = metrics["ai_report"]
    return {
        "experiment": name,
        "fake_accuracy": fake["accuracy"],
        "fake_macro_f1": fake["macro avg"]["f1-score"],
        "ai_accuracy": ai["accuracy"],
        "ai_macro_f1": ai["macro avg"]["f1-score"],
        "spearman_r": metrics["consistency_attention_correlation"]["spearman_r"],
        "pearson_r": metrics["consistency_attention_correlation"]["pearson_r"],
    }


def main():
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.BERT_MODEL, local_files_only=True, use_fast=True
    )
    signature_holder = SimpleNamespace(cache_signature=_load_cache_signature())
    train_ds, val_ds, test_ds = load_and_split_with_mvc_cat(
        cfg.DATA_PATH,
        tokenizer,
        signature_holder,
        mvc_cache_dir=cfg.MVC_CACHE_DIR,
        precompute=False,
        random_state=42,
        max_len_title=cfg.MAX_LEN_TITLE,
        max_len_text=cfg.MAX_LEN_TEXT,
        max_len_desc=cfg.MAX_LEN_DESC,
        drop_incomplete=cfg.DROP_INCOMPLETE,
        max_body_chars=cfg.MAX_BODY_CHARS,
        group_column=cfg.GROUP_COLUMN,
        split_dir=cfg.SPLIT_DIR,
        drop_conflicting_duplicates=cfg.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=cfg.DROP_EXACT_DUPLICATES,
    )
    train_loader = _loader(train_ds, True, cfg.TRAIN_NUM_WORKERS)
    val_loader = _loader(val_ds, False, cfg.VAL_NUM_WORKERS)
    test_loader = _loader(test_ds, False, cfg.VAL_NUM_WORKERS)

    result_root = Path(cfg.OUTPUT_DIR) / "ablation"
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in EXPERIMENTS.items():
        output_dir = result_root / name
        metrics_path = output_dir / "test_metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                rows.append(_summary(name, json.load(handle)))
            continue

        print(f"\n===== Ablation: {name}; weights={weights} =====")
        set_seed(42)
        experiment_cfg = copy.deepcopy(cfg)
        experiment_cfg.OUTPUT_DIR = str(output_dir)
        experiment_cfg.SAVE_LAST_CHECKPOINT = False
        experiment_cfg.SAVE_BEST_CHECKPOINT = True
        model = MVCTTokenCATModel(
            bert_model_name=cfg.BERT_MODEL,
            mvc_weights=weights,
            cat_layers=cfg.CAT_LAYERS,
            cat_heads=cfg.CAT_HEADS,
            cat_ff_multiplier=cfg.CAT_FF_MULTIPLIER,
            dropout_p=cfg.CAT_DROPOUT,
            lambda_init=cfg.CAT_LAMBDA_INIT,
            learnable_mvc_weights=cfg.LEARNABLE_MVC_WEIGHTS,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )
        model, _ = train_model_cat(
            model,
            train_loader,
            val_loader,
            experiment_cfg,
            optimizer,
            output_dir=str(output_dir),
        )
        metrics = evaluate_cat(
            model,
            test_loader,
            device,
            output_path=str(metrics_path),
            fake_threshold=cfg.FAKE_THRESHOLD,
        )
        rows.append(_summary(name, metrics))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_path = result_root / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    print("Saved:", result_path)


if __name__ == "__main__":
    main()
