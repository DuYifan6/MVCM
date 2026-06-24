# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_json


# FocalLoss based on CrossEntropy (for logits with shape (B, C) and target int labels)
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
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, reduction="none", weight=w)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss



def evaluate_epoch(model, dataloader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid"):
            title = {k: v.to(device) for k, v in batch["title"].items()}
            text  = {k: v.to(device) for k, v in batch["text"].items()}
            desc  = {k: v.to(device) for k, v in batch["desc"].items()}
            mvc   = batch["mvc"].float().to(device)

            labels_fake = batch["label_fake"].to(device)
            labels_ai   = batch["label_ai"].to(device)

            y_fake, y_ai, _ = model(title, text, desc, mvc)

            loss = ce(y_fake, labels_fake) + 0.5 * ce(y_ai, labels_ai)
            b = labels_fake.size(0)
            total_loss += loss.item() * b
            total_samples += b

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


def train_one_epoch(model, dataloader, optimizer, device, cfg, focal_loss_fn):
    model.train()
    total_loss = 0.0
    total_samples = 0

    accumulation_steps = max(1, getattr(cfg, "ACCUM_STEPS", 1))
    optimizer.zero_grad()

    last_step = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Train")):
        last_step = step

        title = {k: v.to(device) for k, v in batch["title"].items()}
        text  = {k: v.to(device) for k, v in batch["text"].items()}
        desc  = {k: v.to(device) for k, v in batch["desc"].items()}
        mvc   = batch["mvc"].float().to(device)

        labels_fake = batch["label_fake"].to(device)
        labels_ai   = batch["label_ai"].to(device)

        y_fake, y_ai, _ = model(title, text, desc, mvc)

        # focal loss on fake head
        focal_loss = focal_loss_fn(y_fake, labels_fake)

        # normal CE for ai head
        ce_ai = F.cross_entropy(y_ai, labels_ai)

        loss = focal_loss + 0.5 * ce_ai

        # attention regularization (encourage attention distribution to be not overly peaked)
        if hasattr(model, "last_attn") and model.last_attn is not None:
            attn = model.last_attn  # (b,3,3)
            # 对每个样本、每个 entry 的平均值，鼓励其接近 1/3
            mean_per_entry = attn.mean(dim=(1, 2))  # (b,)
            attn_reg = ((mean_per_entry - (1.0 / 3.0)) ** 2).mean()
            loss = loss + cfg.ATTN_REG_WEIGHT * attn_reg

        # backward with gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels_fake.size(0) * accumulation_steps
        total_samples += labels_fake.size(0)

    # 处理尾巴：最后一小段不足 accumulation_steps 的情况
    if (last_step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg = total_loss / total_samples if total_samples > 0 else 0.0
    return avg


def train_model(model, train_loader, val_loader, cfg, optimizer, output_dir="outputs"):
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")

    focal_loss_fn = FocalLossCE(
        alpha=getattr(cfg, "FOCAL_ALPHA", 0.8),
        gamma=getattr(cfg, "FOCAL_GAMMA", 2.0),
        weight=getattr(cfg, "FAKE_CLASS_WEIGHT", None),
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val = float("inf")
    patience = getattr(cfg, "EARLYSTOP_PATIENCE", 3)
    patience_counter = 0
    best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pt")

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(getattr(cfg, "EPOCHS", 6)):
        print(f"\n===== Epoch {epoch + 1}/{cfg.EPOCHS} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            focal_loss_fn
        )

        val_loss = evaluate_epoch(
            model,
            val_loader,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        # 保存每一轮模型权重
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(output_dir, f"mvct_epoch{epoch + 1}.pt")
        # )

        # 保存完整 checkpoint，便于断点恢复
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "history": history,
            "config": cfg.__dict__ if hasattr(cfg, "__dict__") else {}
        }

        torch.save(
            checkpoint,
            os.path.join(output_dir, "last_checkpoint.pt")
        )

        save_json(history, os.path.join(output_dir, "history.json"))

        # optional: save epoch checkpoint
        if getattr(cfg, "SAVE_EPOCH_CHECKPOINTS", False):
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"mvct_epoch{epoch + 1}.pt")
            )

        # Early stopping + best checkpointing
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0

            # optional: save best checkpoint
            if getattr(cfg, "SAVE_BEST_CHECKPOINT", False):
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "best_checkpoint.pt")
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    # 加载验证集最优模型，用于后续测试
    if os.path.exists(best_ckpt_path):
        best_checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    return model, history
