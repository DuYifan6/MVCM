# explain.py
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


# ======================================================
# 1. Attention 可视化
# ======================================================
def save_attention_heatmap(attn_matrix, path, view_names=("title", "text", "desc")):
    """
    保存 3x3 attention 热力图。
    attn_matrix: (3,3) or (1,3,3)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy(attn_matrix)

    if arr.ndim == 3:
        arr = arr[0]
    assert arr.shape == (3, 3), f"attention shape should be (3,3), got {arr.shape}"

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        xticklabels=view_names,
        yticklabels=view_names,
    )
    plt.title("Consistency-Attention")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================
# 2. MVCM 热力图 (S/F/L/Local 四通道)
# ======================================================
def save_mvc_heatmaps(
    mvc_tensor,
    out_dir,
    view_names=("title", "text", "desc"),
    channel_names=("semantic", "fact", "logic", "local"),
):
    """
    对 MVCM 的每一个通道保存一张 3x3 热力图。
    mvc_tensor: (3,3,C) 或 (1,3,3,C)，C 可以是 3 或 4。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = _to_numpy(mvc_tensor)
    if tensor.ndim == 4:
        tensor = tensor[0]  # (3,3,C)

    if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.shape[1] != 3:
        raise ValueError(f"Unexpected MVC tensor shape: {tensor.shape}")

    C = tensor.shape[-1]
    # channel_names 最多用前 C 个
    names = channel_names[:C]

    for k in range(C):
        mat = tensor[:, :, k]
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            mat,
            xticklabels=view_names,
            yticklabels=view_names,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=0.0,
            vmax=1.0,
        )
        plt.title(f"MVC {names[k]}")
        plt.tight_layout()
        plt.savefig(out_dir / f"mvc_{names[k]}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_mvc_overview_figure(
    mvc_tensor,
    path,
    view_names=("title", "text", "desc"),
    channel_names=("Semantic", "Factual", "Logical", "Local"),
    cmap="vlag",
):
    """
    生成一张 2x2 子图的论文风格 Figure：四通道 MVCM。
    mvc_tensor: (3,3,4) 或 (1,3,3,4)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    mvc = _to_numpy(mvc_tensor)
    if mvc.ndim == 4:
        mvc = mvc[0]  # (3,3,C)

    if mvc.ndim != 3 or mvc.shape[0] != 3 or mvc.shape[1] != 3:
        raise ValueError(f"Unexpected mvc shape for overview: {mvc.shape}")

    C = mvc.shape[-1]
    if C < 4:
        raise ValueError(f"Need at least 4 channels for overview, got {C}")

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    axes = axes.flatten()

    for k in range(4):  # 只画前 4 个通道：S/F/L/Local
        ax = axes[k]
        mat = mvc[:, :, k]
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            xticklabels=view_names,
            yticklabels=view_names,
            cbar=(k == 3),
        )
        ax.set_title(channel_names[k], fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ======================================================
# 3. 单样本解释：attention + MVCM 四通道 + 简单 local 解释
# ======================================================
def _summarize_local_matrix(local_mat):
    """
    对 3x3 的 local 一致性矩阵做一个简单摘要，方便写入 JSON 用于解释。
    返回:
      {
        "title_text": float,
        "title_desc": float,
        "text_desc": float,
        "diag_mean": float,
        "offdiag_mean": float
      }
    """
    local = np.array(local_mat, dtype=float)
    if local.shape != (3, 3):
        return None

    tt = float(local[0, 1])  # title-text
    td = float(local[0, 2])  # title-desc
    xd = float(local[1, 2])  # text-desc

    diag_mean = float(np.mean(np.diag(local)))
    offdiag = local.copy()
    np.fill_diagonal(offdiag, np.nan)
    offdiag_mean = float(np.nanmean(offdiag))

    return {
        "title_text": tt,
        "title_desc": td,
        "text_desc": xd,
        "diag_mean": diag_mean,
        "offdiag_mean": offdiag_mean,
    }


def explain_example(model, title_field, text_field, desc_field, mvc_tensor, device="cpu"):
    """
    对单条样本进行前向推理，并返回:
      - attention 矩阵
      - MVC 各通道 (semantic / fact / logic / local)
      - 融合矩阵 mvc_comb
      - 简单的 local 一致性摘要 (local_summary)
    """
    model.eval()
    title = {k: v.to(device) for k, v in title_field.items()}
    text  = {k: v.to(device) for k, v in text_field.items()}
    desc  = {k: v.to(device) for k, v in desc_field.items()}

    mvc = mvc_tensor
    if not isinstance(mvc, torch.Tensor):
        mvc = torch.tensor(mvc, device=device)
    else:
        mvc = mvc.to(device)

    with torch.no_grad():
        # 如果 mvc 形状是 (3,3,*)，在模型内部会被扩展成 batch 维
        y_fake, y_ai, attn = model(
            title, text, desc, mvc.unsqueeze(0) if mvc.dim() >= 2 and mvc.size(0) == 3 else mvc
        )
        attn_np = attn[0].detach().cpu().numpy()

    mvc_np = mvc.detach().cpu().numpy()

    expl = {
        "attention": attn_np.tolist(),
        "mvc_comb": None,
        "mvc_semantic": None,
        "mvc_fact": None,
        "mvc_logic": None,
        "mvc_local": None,
        "local_summary": None,
    }

    # mvc_np 可能是 (3,3,C) 或 (B,3,3,C)
    if mvc_np.ndim == 4:
        mvc_np = mvc_np[0]  # (3,3,C)

    if mvc_np.ndim == 3 and mvc_np.shape[0] == 3 and mvc_np.shape[1] == 3:
        C = mvc_np.shape[2]
        if C >= 1:
            expl["mvc_semantic"] = mvc_np[:, :, 0].tolist()
        if C >= 2:
            expl["mvc_fact"] = mvc_np[:, :, 1].tolist()
        if C >= 3:
            expl["mvc_logic"] = mvc_np[:, :, 2].tolist()
        if C >= 4:
            expl["mvc_local"] = mvc_np[:, :, 3].tolist()
            expl["local_summary"] = _summarize_local_matrix(mvc_np[:, :, 3])

        # 用简单平均得到一个组合矩阵做可视化（训练时已经有加权）
        expl["mvc_comb"] = mvc_np.mean(axis=-1).tolist()
    elif mvc_np.ndim == 2 and mvc_np.shape == (3, 3):
        expl["mvc_comb"] = mvc_np.tolist()
    else:
        expl["mvc_comb"] = mvc_np.tolist()

    return expl


# ======================================================
# 4. 保存 JSON
# ======================================================
def save_explanation_json(expl, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expl, f, ensure_ascii=False, indent=2)
