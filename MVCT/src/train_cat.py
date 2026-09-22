import os
from dataclasses import asdict, is_dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model_cat import attention_kl_regularizer
from utils import save_json


class FocalLossCE(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean", weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, reduction="none", weight=weight)
        probability = torch.exp(-ce)
        loss = self.alpha * (1.0 - probability) ** self.gamma * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _move_sequence(batch, device):
    return {key: value.to(device) for key, value in batch["sequence"].items()}


def _task_loss(y_fake, y_ai, labels_fake, labels_ai, focal_loss_fn):
    return focal_loss_fn(y_fake, labels_fake) + 0.5 * F.cross_entropy(y_ai, labels_ai)


def evaluate_epoch(model, dataloader, device, focal_loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid"):
            sequence = _move_sequence(batch, device)
            mvc = batch["mvc"].float().to(device)
            labels_fake = batch["label_fake"].to(device)
            labels_ai = batch["label_ai"].to(device)
            y_fake, y_ai, _ = model(sequence, mvc)
            loss = _task_loss(y_fake, y_ai, labels_fake, labels_ai, focal_loss_fn)
            batch_size = labels_fake.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples if total_samples else 0.0


def train_one_epoch(model, dataloader, optimizer, device, cfg, focal_loss_fn):
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    accumulation_steps = max(1, int(getattr(cfg, "ACCUM_STEPS", 1)))
    optimizer.zero_grad(set_to_none=True)
    last_step = -1

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        last_step = step
        sequence = _move_sequence(batch, device)
        mvc = batch["mvc"].float().to(device)
        labels_fake = batch["label_fake"].to(device)
        labels_ai = batch["label_ai"].to(device)
        y_fake, y_ai, attention = model(sequence, mvc)

        task_loss = _task_loss(y_fake, y_ai, labels_fake, labels_ai, focal_loss_fn)
        regularizer = attention_kl_regularizer(
            attention,
            sequence["attention_mask"],
            sequence.get("view_ids"),
        )
        loss = task_loss + float(getattr(cfg, "ATTN_REG_WEIGHT", 0.0)) * regularizer
        (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels_fake.size(0)
        total_loss += loss.item() * batch_size
        total_task_loss += task_loss.item() * batch_size
        total_reg_loss += regularizer.item() * batch_size
        total_samples += batch_size

    if last_step >= 0 and (last_step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    denominator = max(total_samples, 1)
    return {
        "loss": total_loss / denominator,
        "task_loss": total_task_loss / denominator,
        "attention_kl": total_reg_loss / denominator,
    }


def _config_dict(cfg):
    if is_dataclass(cfg):
        return asdict(cfg)
    return {
        key: value
        for key, value in vars(cfg).items()
        if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))
    }


def train_model_cat(model, train_loader, val_loader, cfg, optimizer, output_dir=None):
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_dir = output_dir or cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    focal_loss_fn = FocalLossCE(
        alpha=getattr(cfg, "FOCAL_ALPHA", 0.8),
        gamma=getattr(cfg, "FOCAL_GAMMA", 2.0),
        weight=getattr(cfg, "FAKE_CLASS_WEIGHT", None),
    )
    history = {
        "train_loss": [],
        "train_task_loss": [],
        "attention_kl": [],
        "val_loss": [],
    }
    best_val = float("inf")
    patience = int(getattr(cfg, "EARLYSTOP_PATIENCE", 3))
    patience_counter = 0
    best_path = os.path.join(output_dir, "best_checkpoint.pt")
    last_path = os.path.join(output_dir, "last_checkpoint.pt")
    best_saved_this_run = False

    for epoch in range(int(getattr(cfg, "EPOCHS", 10))):
        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} =====")
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, cfg, focal_loss_fn
        )
        val_loss = evaluate_epoch(model, val_loader, device, focal_loss_fn)
        history["train_loss"].append(train_stats["loss"])
        history["train_task_loss"].append(train_stats["task_loss"])
        history["attention_kl"].append(train_stats["attention_kl"])
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} "
            f"train={train_stats['loss']:.4f} val={val_loss:.4f} "
            f"attn_kl={train_stats['attention_kl']:.4f}"
        )

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "history": history,
            "config": _config_dict(cfg),
        }
        if getattr(cfg, "SAVE_LAST_CHECKPOINT", True):
            torch.save(checkpoint, last_path)
        if improved and getattr(cfg, "SAVE_BEST_CHECKPOINT", True):
            torch.save(checkpoint, best_path)
            best_saved_this_run = True
        if getattr(cfg, "SAVE_EPOCH_CHECKPOINTS", False):
            torch.save(checkpoint, os.path.join(output_dir, f"epoch_{epoch + 1}.pt"))
        save_json(history, os.path.join(output_dir, "history.json"))

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_saved_this_run:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint generated by this run: {best_path}")
    return model, history
