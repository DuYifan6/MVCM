# main.py
import os
import torch
from transformers import BertTokenizerFast

from config import cfg
from utils import set_seed
from dataset import load_and_split_with_mvc
from consistency_openie import ConsistencyBuilderOpenIE
from model import MVCTModel
from train import train_model
from evaluate import evaluate
from explain import (
    explain_example,
    save_attention_heatmap,
    save_mvc_heatmaps,
    save_mvc_overview_figure,
    save_explanation_json,
)


def main():
    # 固定随机种子
    set_seed(42)

    # 设备选择
    device = (
        cfg.DEVICE
        if cfg.DEVICE != "cuda"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # 1. 加载 tokenizer
    # ---------------------------
    tokenizer = BertTokenizerFast.from_pretrained(
        cfg.BERT_MODEL,
        local_files_only=True,  # 使用本地 BERT
    )

    # ---------------------------
    # 2. 初始化 MVC 构建器（含 Local 通道）
    # ---------------------------
    builder = ConsistencyBuilderOpenIE(
        sbert_model=cfg.SBERT_MODEL,
        stanford_url=cfg.STANFORD_URL,
        sem_weight=cfg.SEM_WEIGHT,
        fact_weight=cfg.FACT_WEIGHT,
        logic_weight=cfg.LOGIC_WEIGHT,
        local_weight=cfg.LOCAL_WEIGHT,
    )

    print("Loading data and precomputing MVCM (parallel if enabled)...")

    # ---------------------------
    # 3. 预计算 MVCM 并划分数据集
    # ---------------------------
    train_ds, val_ds, test_ds = load_and_split_with_mvc(
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
        use_parallel=cfg.USE_PARALLEL,  # 方案A：False
        max_workers=cfg.MAX_WORKERS,
    )

    from torch.utils.data import DataLoader

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

    # ---------------------------
    # 4. 构建模型与优化器
    # ---------------------------
    model = MVCTModel(
        bert_model_name=cfg.BERT_MODEL,
        mvc_weights=(
            cfg.SEM_WEIGHT,
            cfg.FACT_WEIGHT,
            cfg.LOGIC_WEIGHT,
            cfg.LOCAL_WEIGHT,
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ---------------------------
    # 5. 训练与早停
    # ---------------------------
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        optimizer,
        output_dir=cfg.OUTPUT_DIR,
    )

    # ---------------------------
    # 6. 测试评估
    # ---------------------------
    metrics = evaluate(
        model,
        test_loader,
        device,
        output_path=f"{cfg.OUTPUT_DIR}/test_metrics.json",
        fake_threshold=getattr(cfg, "FAKE_THRESHOLD", 0.5),
    )
    print("Test metrics saved to", f"{cfg.OUTPUT_DIR}/test_metrics.json")

    # ---------------------------
    # 7. 解释第一条测试样本
    # ---------------------------
    batch = next(iter(test_loader))

    title_field = {k: v[0:1].to(device) for k, v in batch["title"].items()}
    text_field = {k: v[0:1].to(device) for k, v in batch["text"].items()}
    desc_field = {k: v[0:1].to(device) for k, v in batch["desc"].items()}

    # (3,3,4) MVCM tensor: [Semantic, Factual, Logical, Local]
    mvc_tensor = batch["mvc"][0]

    expl = explain_example(
        model,
        title_field,
        text_field,
        desc_field,
        mvc_tensor,
        device=device,
    )

    # 注意力热力图
    save_attention_heatmap(
        expl["attention"],
        os.path.join(cfg.OUTPUT_DIR, "attn_sample.png"),
    )

    # MVCM 四通道热力图 + 总览图
    save_mvc_heatmaps(
        mvc_tensor,
        os.path.join(cfg.OUTPUT_DIR, "mvc_explain"),
    )
    save_mvc_overview_figure(
        mvc_tensor,
        os.path.join(cfg.OUTPUT_DIR, "mvc_overview.png"),
    )

    # 解释 JSON
    save_explanation_json(
        expl,
        os.path.join(cfg.OUTPUT_DIR, "explanation.json"),
    )


if __name__ == "__main__":
    main()
