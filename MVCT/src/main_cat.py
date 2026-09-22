import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from consistency_hybrid import ConsistencyBuilderHybrid
from dataset_cat import load_and_split_with_mvc_cat
from evaluate_cat import evaluate_cat
from explain import (
    save_attention_heatmap,
    save_explanation_json,
    save_mvc_heatmaps,
    save_mvc_overview_figure,
)
from model_cat import MVCTTokenCATModel
from train_cat import train_model_cat
from utils import save_json, set_seed


def _dataloader(dataset, batch_size, shuffle, workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def main():
    set_seed(42)
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    save_json(asdict(cfg), os.path.join(cfg.OUTPUT_DIR, "config.json"))
    print("Using device:", device)
    if not cfg.USE_NLI:
        print("WARNING: USE_NLI=False; logical consistency uses explicit rules only.")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.BERT_MODEL, local_files_only=True, use_fast=True
    )
    builder = ConsistencyBuilderHybrid(
        sbert_model=cfg.SBERT_MODEL,
        stanford_url=cfg.STANFORD_URL,
        nli_model=cfg.NLI_MODEL,
        use_nli=cfg.USE_NLI,
        nli_batch_size=cfg.NLI_BATCH_SIZE,
        nli_max_length=cfg.NLI_MAX_LENGTH,
        nli_contradiction_id=cfg.NLI_CONTRADICTION_ID,
        sbert_batch_size=cfg.SBERT_BATCH_SIZE,
        openie_workers=cfg.OPENIE_WORKERS,
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
    train_ds, val_ds, test_ds = load_and_split_with_mvc_cat(
        cfg.DATA_PATH,
        tokenizer,
        builder,
        test_size=0.1,
        val_size=0.1,
        mvc_cache_dir=cfg.MVC_CACHE_DIR,
        precompute=True,
        random_state=42,
        max_len_title=cfg.MAX_LEN_TITLE,
        max_len_text=cfg.MAX_LEN_TEXT,
        max_len_desc=cfg.MAX_LEN_DESC,
        use_parallel=cfg.USE_PARALLEL,
        max_workers=cfg.MAX_WORKERS,
        drop_incomplete=cfg.DROP_INCOMPLETE,
        max_body_chars=cfg.MAX_BODY_CHARS,
        group_column=cfg.GROUP_COLUMN,
        split_dir=cfg.SPLIT_DIR,
        drop_conflicting_duplicates=cfg.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=cfg.DROP_EXACT_DUPLICATES,
    )
    train_loader = _dataloader(
        train_ds,
        cfg.BATCH_SIZE,
        True,
        cfg.TRAIN_NUM_WORKERS,
        cfg.PIN_MEMORY,
    )
    val_loader = _dataloader(
        val_ds,
        cfg.BATCH_SIZE,
        False,
        cfg.VAL_NUM_WORKERS,
        cfg.PIN_MEMORY,
    )
    test_loader = _dataloader(
        test_ds,
        cfg.BATCH_SIZE,
        False,
        cfg.VAL_NUM_WORKERS,
        cfg.PIN_MEMORY,
    )

    model = MVCTTokenCATModel(
        bert_model_name=cfg.BERT_MODEL,
        mvc_weights=(
            cfg.SEM_WEIGHT,
            cfg.FACT_WEIGHT,
            cfg.LOGIC_WEIGHT,
            cfg.LOCAL_WEIGHT,
        ),
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
        cfg,
        optimizer,
        output_dir=cfg.OUTPUT_DIR,
    )
    metrics = evaluate_cat(
        model,
        test_loader,
        device,
        output_path=os.path.join(cfg.OUTPUT_DIR, "test_metrics.json"),
        fake_threshold=cfg.FAKE_THRESHOLD,
    )
    print("Test metrics saved to", os.path.join(cfg.OUTPUT_DIR, "test_metrics.json"))

    batch = next(iter(test_loader))
    sequence = {
        key: value[0:1].to(device) for key, value in batch["sequence"].items()
    }
    mvc = batch["mvc"][0:1].float().to(device)
    model.eval()
    with torch.no_grad():
        _, _, attention = model(sequence, mvc)
        view_attention = model.aggregate_attention_by_view(
            attention, sequence["view_ids"], sequence["attention_mask"]
        )[0]
    mvc_sample = mvc[0].cpu()
    save_attention_heatmap(
        view_attention.cpu(), os.path.join(cfg.OUTPUT_DIR, "attn_sample.png")
    )
    save_mvc_heatmaps(mvc_sample, os.path.join(cfg.OUTPUT_DIR, "mvc_explain"))
    save_mvc_overview_figure(
        mvc_sample, os.path.join(cfg.OUTPUT_DIR, "mvc_overview.png")
    )
    explanation = {
        "sample_id": int(batch["sample_id"][0].item()),
        "view_attention": view_attention.cpu().tolist(),
        "mvc_semantic": mvc_sample[:, :, 0].tolist(),
        "mvc_factual": mvc_sample[:, :, 1].tolist(),
        "mvc_logical": mvc_sample[:, :, 2].tolist(),
        "mvc_local": mvc_sample[:, :, 3].tolist(),
        "learned_channel_weights": metrics["learned_parameters"][
            "mvc_channel_weights"
        ],
        "cat_lambda": metrics["learned_parameters"]["cat_lambda"],
    }
    save_explanation_json(
        explanation, os.path.join(cfg.OUTPUT_DIR, "explanation.json")
    )


if __name__ == "__main__":
    main()
