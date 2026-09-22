"""Evaluate a saved model on its saved validation split; never evaluate test data."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from consistency_hybrid import ConsistencyBuilderHybrid
from dataset_cat import MVCNewsDatasetCAT, _cache_key, _clean_and_group_dataframe
from model_cat import MVCTTokenCATModel
from train_cat import FocalLossCE, _move_sequence
from utils import compute_metrics_twoheads, save_json


def signature_from_config(config):
    # Reproduce the existing cache signature without loading SBERT/NLI/OpenIE.
    builder = ConsistencyBuilderHybrid.__new__(ConsistencyBuilderHybrid)
    builder.sbert_model_name = config.SBERT_MODEL
    builder.nli_model_name = config.NLI_MODEL
    builder.use_nli = config.USE_NLI
    builder.mvc_weights = np.asarray([
        config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT,
        config.LOCAL_WEIGHT,
    ], dtype=np.float32)
    builder.fact_component_weights = np.asarray([0.4, 0.3, 0.3], dtype=np.float32)
    builder.fact_component_weights /= builder.fact_component_weights.sum()
    builder.logic_weights = np.asarray([
        config.LOGIC_NLI_WEIGHT, config.LOGIC_NEGATION_WEIGHT,
        config.LOGIC_TEMPORAL_WEIGHT, config.LOGIC_NUMERIC_WEIGHT,
    ], dtype=np.float32)
    builder.alignment_threshold = config.LOGIC_ALIGNMENT_THRESHOLD
    builder.coverage_target = config.LOGIC_COVERAGE_TARGET
    builder.max_logic_pairs = config.MAX_LOGIC_PAIRS_PER_VIEW_PAIR
    builder.openie_workers = config.OPENIE_WORKERS
    return builder._make_cache_signature(config.STANFORD_URL)


def threshold_scan(labels, probabilities, reference):
    """Fixed grid, macro-F1 objective, ties prefer the existing threshold."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    rows = []
    for threshold in sorted(set([round(i / 100, 2) for i in range(5, 96)] + [reference])):
        predicted = (probabilities > threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "fake_precision": float(precision_score(labels, predicted, zero_division=0)),
            "fake_recall": float(recall_score(labels, predicted, zero_division=0)),
        })
    selected = max(rows, key=lambda r: (r["macro_f1"], -abs(r["threshold"] - reference), -r["threshold"]))
    return rows, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=cfg.OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    # Only load a trusted checkpoint created by this project.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_val_loss = float(checkpoint["val_loss"])
    config = SimpleNamespace(**checkpoint["config"])
    signature = signature_from_config(config)
    cache_dir = Path(config.MVC_CACHE_DIR)
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    if manifest["builder_signature"] != signature:
        raise ValueError("Checkpoint and cache signatures differ. Do not rebuild or replace caches automatically.")

    split_path = Path(config.SPLIT_DIR) / "val.csv"
    saved = pd.read_csv(split_path, dtype={"_split_group": str})
    frame = _clean_and_group_dataframe(
        config.DATA_PATH, drop_incomplete=config.DROP_INCOMPLETE,
        max_body_chars=config.MAX_BODY_CHARS, group_column=config.GROUP_COLUMN,
        drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES,
    )
    if not frame.ID.is_unique or not saved.ID.is_unique:
        raise ValueError("Sample IDs must be unique to restore the saved validation split.")
    frame = frame.set_index("ID").loc[saved.ID].reset_index()
    for column in ("label_fake", "label_ai", "_split_group"):
        if frame[column].astype(str).tolist() != saved[column].astype(str).tolist():
            raise ValueError(f"Saved validation split mismatch: {column}")
    builder = SimpleNamespace(cache_signature=signature)
    frame["mvc_path"] = [str(cache_dir / f"mvc_{_cache_key(row, builder)}.npy")
                         for row in frame.itertuples(index=False)]
    missing = [path for path in frame.mvc_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} validation caches; no features were recomputed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; checkpoint epoch: {checkpoint['epoch']}; validation samples: {len(frame)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL, local_files_only=True, use_fast=True)
    dataset = MVCNewsDatasetCAT(frame, tokenizer, str(cache_dir), config.MAX_LEN_TITLE,
                               config.MAX_LEN_TEXT, config.MAX_LEN_DESC)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=(config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT),
        cat_layers=config.CAT_LAYERS, cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER, dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT, learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=getattr(config, "MASK_MISSING_EVIDENCE", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device).eval()
    focal = FocalLossCE(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                       weight=config.FAKE_CLASS_WEIGHT)
    predictions, fake_loss, ai_loss = [], 0.0, 0.0
    with torch.inference_mode():
        for batch in loader:
            fake, ai, _ = model(_move_sequence(batch, device), batch["mvc"].float().to(device))
            labels_fake, labels_ai = batch["label_fake"].to(device), batch["label_ai"].to(device)
            fake_loss += focal(fake, labels_fake).item() * len(labels_fake)
            ai_loss += torch.nn.functional.cross_entropy(ai, labels_ai).item() * len(labels_ai)
            pf, pa = fake.softmax(-1)[:, 1].cpu().tolist(), ai.softmax(-1)[:, 1].cpu().tolist()
            for i, sample_id in enumerate(batch["sample_id"].tolist()):
                predictions.append({"sample_id": sample_id, "true_fake": int(labels_fake[i]),
                                    "true_ai": int(labels_ai[i]), "prob_fake": pf[i], "prob_ai": pa[i]})
    true_fake = [r["true_fake"] for r in predictions]
    true_ai = [r["true_ai"] for r in predictions]
    probs = np.asarray([r["prob_fake"] for r in predictions])
    pred_ai = [int(r["prob_ai"] > 0.5) for r in predictions]
    scan, selected = threshold_scan(true_fake, probs, config.FAKE_THRESHOLD)
    report = {
        "split": "validation", "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_val_loss": checkpoint_val_loss,
        "loss_reproduction_abs_difference": abs((fake_loss + 0.5 * ai_loss) / len(predictions) - checkpoint_val_loss),
        "loss_reproduced": bool(np.isclose((fake_loss + 0.5 * ai_loss) / len(predictions), checkpoint_val_loss, atol=1e-5, rtol=1e-4)),
        "saved_split": str(split_path.resolve()), "cache_signature": signature,
        "n": len(predictions), "reference_threshold": config.FAKE_THRESHOLD,
        "mask_missing_evidence": getattr(config, "MASK_MISSING_EVIDENCE", False),
        "fake_focal_loss": fake_loss / len(predictions),
        "ai_cross_entropy": ai_loss / len(predictions),
        "combined_task_loss": (fake_loss + 0.5 * ai_loss) / len(predictions),
        "at_reference_threshold": compute_metrics_twoheads((probs > config.FAKE_THRESHOLD).astype(int), true_fake, pred_ai, true_ai),
        "at_0_50_threshold": compute_metrics_twoheads((probs > 0.5).astype(int), true_fake, pred_ai, true_ai),
        "selected_on_validation": selected,
        "selection_rule": "Max validation fake macro-F1, grid 0.05..0.95 step 0.01; ties nearest reference, then lower threshold.",
        "note": "Validation selection scores are optimistic; this script does not evaluate the test split or modify config/checkpoints.",
    }
    output = run_dir / ("validation_diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    save_json(report, output / "validation_metrics.json")
    save_json(predictions, output / "validation_predictions.json")
    save_json(scan, output / "threshold_scan.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved to:", output.resolve())
    return report, output


if __name__ == "__main__":
    main()
