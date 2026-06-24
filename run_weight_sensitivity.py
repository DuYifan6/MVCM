import os
import json
import copy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from config import cfg
from utils import set_seed
from dataset import load_and_split_with_mvc
from model import MVCTModel
from train import train_model
from evaluate import evaluate


WEIGHT_EXPERIMENTS = {
    "equal": (0.25, 0.25, 0.25, 0.25),
    "default": (0.40, 0.25, 0.20, 0.15),
    "semantic_high": (0.50, 0.20, 0.15, 0.15),
    "factual_high": (0.30, 0.35, 0.20, 0.15),
    "logical_high": (0.30, 0.20, 0.35, 0.15),
    "local_high": (0.30, 0.20, 0.15, 0.35),
}


def load_metrics_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def summarize_metrics(exp_name, weights, metrics):
    fake_report = metrics["fake_report"]
    ai_report = metrics["ai_report"]

    fake_macro_f1 = fake_report.get("macro avg", {}).get("f1-score", None)
    ai_macro_f1 = ai_report.get("macro avg", {}).get("f1-score", None)

    row = {
        "experiment": exp_name,
        "semantic_weight": weights[0],
        "factual_weight": weights[1],
        "logical_weight": weights[2],
        "local_weight": weights[3],
        "fake_acc": fake_report.get("accuracy", None),
        "fake_macro_f1": fake_macro_f1,
        "fake_weighted_f1": fake_report.get("weighted avg", {}).get("f1-score", None),
        "ai_acc": ai_report.get("accuracy", None),
        "ai_macro_f1": ai_macro_f1,
        "ai_weighted_f1": ai_report.get("weighted avg", {}).get("f1-score", None),
        "avg_macro_f1": (
            (fake_macro_f1 + ai_macro_f1) / 2
            if fake_macro_f1 is not None and ai_macro_f1 is not None
            else None
        ),
    }

    corr = metrics.get("consistency_attention_correlation", {})
    row["spearman_r"] = corr.get("spearman_r", None)
    row["pearson_r"] = corr.get("pearson_r", None)

    return row


def main():
    set_seed(42)

    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    tokenizer = BertTokenizerFast.from_pretrained(
        cfg.BERT_MODEL,
        local_files_only=True,
    )

    print("Loading dataset with existing MVC cache...")

    train_ds, val_ds, test_ds = load_and_split_with_mvc(
        cfg.DATA_PATH,
        tokenizer,
        builder=None,
        test_size=0.1,
        val_size=0.1,
        mvc_cache_dir=cfg.MVC_CACHE_DIR,
        precompute=False,
        random_state=42,
        max_len_title=cfg.MAX_LEN_TITLE,
        max_len_text=cfg.MAX_LEN_TEXT,
        max_len_desc=cfg.MAX_LEN_DESC,
        use_parallel=False,
        max_workers=cfg.MAX_WORKERS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(cfg, "TRAIN_NUM_WORKERS", 0),
        pin_memory=getattr(cfg, "PIN_MEMORY", False),
        persistent_workers=(getattr(cfg, "TRAIN_NUM_WORKERS", 0) > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "VAL_NUM_WORKERS", 0),
        pin_memory=getattr(cfg, "PIN_MEMORY", False),
        persistent_workers=(getattr(cfg, "VAL_NUM_WORKERS", 0) > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "VAL_NUM_WORKERS", 0),
        pin_memory=getattr(cfg, "PIN_MEMORY", False),
        persistent_workers=(getattr(cfg, "VAL_NUM_WORKERS", 0) > 0),
    )

    rows = []

    base_out_dir = os.path.join("outputs", "sensitivity", "mvc_weights")
    os.makedirs(base_out_dir, exist_ok=True)

    for exp_name, weights in WEIGHT_EXPERIMENTS.items():
        print(f"\n========== Running MVC weight sensitivity: {exp_name} ==========")
        print("MVC weights:", weights)

        out_dir = os.path.join(base_out_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        metric_path = os.path.join(out_dir, "test_metrics.json")

        old_metrics = load_metrics_if_exists(metric_path)
        if old_metrics is not None:
            print(f"Found existing result for {exp_name}, skip training.")
            rows.append(summarize_metrics(exp_name, weights, old_metrics))
            continue

        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.OUTPUT_DIR = out_dir

        cfg_exp.SEM_WEIGHT = weights[0]
        cfg_exp.FACT_WEIGHT = weights[1]
        cfg_exp.LOGIC_WEIGHT = weights[2]
        cfg_exp.LOCAL_WEIGHT = weights[3]

        # 只做论文结果，不保存模型 checkpoint
        cfg_exp.SAVE_EPOCH_CHECKPOINTS = False
        cfg_exp.SAVE_BEST_CHECKPOINT = False
        cfg_exp.SAVE_LAST_CHECKPOINT = False

        set_seed(42)

        model = MVCTModel(
            bert_model_name=cfg_exp.BERT_MODEL,
            mvc_weights=weights,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg_exp.LR,
            weight_decay=cfg_exp.WEIGHT_DECAY,
        )

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            cfg_exp,
            optimizer,
            output_dir=out_dir,
        )

        metrics = evaluate(
            model,
            test_loader,
            device,
            output_path=metric_path,
            fake_threshold=getattr(cfg_exp, "FAKE_THRESHOLD", 0.55),
        )

        rows.append(summarize_metrics(exp_name, weights, metrics))

        # 删除可能生成的模型权重，防止占空间
        for root, _, files in os.walk(out_dir):
            for file in files:
                if file.endswith(".pt"):
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception:
                        pass

        del model
        torch.cuda.empty_cache()

    result_path = os.path.join(base_out_dir, "mvc_weight_results.csv")
    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    print("\nMVC weight sensitivity results saved to:", result_path)


if __name__ == "__main__":
    main()