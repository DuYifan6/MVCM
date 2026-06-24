# evaluate.py
import torch
from tqdm import tqdm
from utils import compute_metrics_twoheads, save_json,offdiag_mean
import numpy as np
from scipy.stats import spearmanr, pearsonr

def evaluate_epoch(model, dataloader, device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid"):
            title = {k: v.to(device) for k,v in batch["title"].items()}
            text  = {k: v.to(device) for k,v in batch["text"].items()}
            desc  = {k: v.to(device) for k,v in batch["desc"].items()}
            mvc   = batch["mvc"].float().to(device)
            labels_fake = batch["label_fake"].to(device)
            labels_ai   = batch["label_ai"].to(device)
            y_fake, y_ai, _ = model(title, text, desc, mvc)
            loss = ce(y_fake, labels_fake) + 0.5 * ce(y_ai, labels_ai)
            b = labels_fake.size(0)
            total_loss += loss.item() * b
            total_samples += b
    return total_loss / total_samples



def evaluate(model, dataloader, device, output_path=None, fake_threshold=0.5):
    model.eval()

    preds_fake, trues_fake = [], []
    preds_ai, trues_ai = [], []

    attn_off_list = []
    mvc_off_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            title = {k: v.to(device) for k,v in batch["title"].items()}
            text  = {k: v.to(device) for k,v in batch["text"].items()}
            desc  = {k: v.to(device) for k,v in batch["desc"].items()}
            mvc   = batch["mvc"].float().to(device)

            labels_fake = batch["label_fake"].to(device)
            labels_ai   = batch["label_ai"].to(device)

            y_fake, y_ai, attn = model(title, text, desc, mvc)

            # ========= 新增部分 =========
            # attention off-diagonal
            A_off = offdiag_mean(attn)  # (B,)
            attn_off_list.append(A_off.cpu())

            # MVC off-diagonal（用 comb 或指定通道）
            if mvc.dim() == 4:  # (B,3,3,C)
                # 例1：用 comb
                w = model.mvc_weights[:mvc.size(-1)].to(device)
                comb = (mvc * w.view(1,1,1,-1)).sum(dim=-1)  # (B,3,3)
            else:
                comb = mvc

            M_off = offdiag_mean(comb)
            mvc_off_list.append(M_off.cpu())
            # ============================

            probs_fake = torch.softmax(y_fake, dim=1)
            pred_fake = (probs_fake[:, 1] > fake_threshold).long()

            preds_fake.extend(pred_fake.cpu().tolist())
            preds_ai.extend(y_ai.argmax(dim=1).cpu().tolist())
            trues_fake.extend(labels_fake.cpu().tolist())
            trues_ai.extend(labels_ai.cpu().tolist())

    # ========= 统计相关性 =========
    A_all = torch.cat(attn_off_list).numpy()
    M_all = torch.cat(mvc_off_list).numpy()

    spearman_corr, spearman_p = spearmanr(A_all, M_all)
    pearson_corr, pearson_p   = pearsonr(A_all, M_all)

    metrics = compute_metrics_twoheads(preds_fake, trues_fake, preds_ai, trues_ai)

    metrics["consistency_attention_correlation"] = {
        "spearman_r": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_corr),
        "pearson_p": float(pearson_p),
    }


    if output_path:
        save_json(metrics, output_path)

    return metrics

