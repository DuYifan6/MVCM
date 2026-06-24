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


ABLATION_EXPERIMENTS = {
    "wo_mvcm": (0.00, 0.00, 0.00, 0.00),
    "wo_semantic": (0.00, 0.25, 0.20, 0.15),
    "wo_factual": (0.40, 0.00, 0.20, 0.15),
    "wo_logical": (0.40, 0.25, 0.00, 0.15),
    "wo_local": (0.40, 0.25, 0.20, 0.00),
}


def summarize_metrics(exp_name, metrics):
    fake_report = metrics["fake_report"]
    ai_report = metrics["ai_report"]

    row = {
        "experiment": exp_name,
        "fake_acc": fake_report.get("accuracy", None),
        "fake_macro_f1": fake_report.get("macro avg", {}).get("f1-score", None),
        "fake_weighted_f1": fake_report.get("weighted avg", {}).get("f1-score", None),
        "ai_acc": ai_report.get("accuracy", None),
        "ai_macro_f1": ai_report.get("macro avg", {}).get("f1-score", None),
        "ai_weighted_f1": ai_report.get("weighted avg", {}).get("f1-score", None),
    }

    if row["fake_macro_f1"] is not None and row["ai_macro_f1"] is not None:
        row["avg_macro_f1"] = (row["fake_macro_f1"] + row["ai_macro_f1"]) / 2
    else:
        row["avg_macro_f1"] = None

    corr = metrics.get("consistency_attention_correlation", {})
    row["spearman_r"] = corr.get("spearman_r", None)
    row["pearson_r"] = corr.get("pearson_r", None)

    return row


def load_metrics_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    set_seed(42)

    device = cfg.DEVICE if cfg.DEVICE != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizerFast.from_pretrained(
        cfg.BERT_MODEL,
        local_files_only=True,
    )

    print("Loading dataset with existing MVC cache...")

    # 关键：precompute=False，表示不重新计算 MVC，只读取已有 mvc_cache
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

    # 先把主实验结果加入总表
    main_metrics_path = os.path.join("outputs", "test_metrics.json")
    main_metrics = load_metrics_if_exists(main_metrics_path)
    if main_metrics is not None:
        rows.append(summarize_metrics("full_mvct", main_metrics))

    for exp_name, weights in ABLATION_EXPERIMENTS.items():
        print(f"\n========== Running ablation: {exp_name} ==========")
        print("MVC weights:", weights)

        out_dir = os.path.join("outputs", "ablation", exp_name)
        os.makedirs(out_dir, exist_ok=True)

        metric_path = os.path.join(out_dir, "test_metrics.json")

        # 如果已经跑过，就跳过，方便断点续跑
        old_metrics = load_metrics_if_exists(metric_path)
        if old_metrics is not None:
            print(f"Found existing result for {exp_name}, skip training.")
            rows.append(summarize_metrics(exp_name, old_metrics))
            continue

        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.OUTPUT_DIR = out_dir
        cfg_exp.SEM_WEIGHT = weights[0]
        cfg_exp.FACT_WEIGHT = weights[1]
        cfg_exp.LOGIC_WEIGHT = weights[2]
        cfg_exp.LOCAL_WEIGHT = weights[3]

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

        rows.append(summarize_metrics(exp_name, metrics))

        del model
        torch.cuda.empty_cache()

    result_path = os.path.join("outputs", "ablation", "ablation_results.csv")
    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    print("\nAblation results saved to:", result_path)


if __name__ == "__main__":
    main()