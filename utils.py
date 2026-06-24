# utils.py
import random
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def compute_metrics_twoheads(pred_fake, true_fake, pred_ai, true_ai):
    r = {}
    r["fake_report"] = classification_report(true_fake, pred_fake, output_dict=True, zero_division=0)
    r["ai_report"] = classification_report(true_ai, pred_ai, output_dict=True, zero_division=0)
    r["fake_confusion"] = confusion_matrix(true_fake, pred_fake).tolist()
    r["ai_confusion"] = confusion_matrix(true_ai, pred_ai).tolist()
    return r

def combine_twohead_to_four(pred_fake, pred_ai, true_label_four=None):
    pred_four = []
    for pf, pa in zip(pred_fake, pred_ai):
        if pf == 0 and pa == 0:
            pred_four.append(0)
        elif pf == 1 and pa == 0:
            pred_four.append(1)
        elif pf == 0 and pa == 1:
            pred_four.append(2)
        elif pf == 1 and pa == 1:
            pred_four.append(3)
    if true_label_four is None:
        return pred_four, None
    else:
        from sklearn.metrics import classification_report, confusion_matrix
        four_report = classification_report(true_label_four, pred_four, output_dict=True, zero_division=0)
        four_confusion = confusion_matrix(true_label_four, pred_four).tolist()
        return pred_four, {"four_report": four_report, "four_confusion": four_confusion}

def offdiag_mean(mat: torch.Tensor):
    """
    mat: (B,3,3) or (3,3)
    return:
        if input is (B,3,3) -> (B,)
        if input is (3,3)   -> scalar tensor
    """
    B = mat.size(0)
    diag = torch.eye(3, device=mat.device).bool().unsqueeze(0)  # (1,3,3)
    off = mat.masked_fill(diag, torch.nan)
    return torch.nanmean(off.view(B, -1), dim=1)