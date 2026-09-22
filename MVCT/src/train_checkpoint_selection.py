"""F1 checkpoint selection; retain the original loss-driven training/early stop.

Do not edit train_cat.py: it is part of the completed replication's code hash.
Both selectors observe the same epochs and validation forward passes.
"""
import math
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from train_cat import FocalLossCE, _config_dict, _move_sequence, _task_loss, train_one_epoch
from utils import save_json


def evaluate_selection_epoch(model, dataloader, device, focal_loss_fn, threshold):
    model.eval()
    total_loss, total_samples = 0.0, 0
    true_fake, pred_fake, true_ai, pred_ai = [], [], [], []
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
            # Same strict > threshold and class-0 tie handling as diagnostics.
            pred_fake.extend((y_fake.softmax(-1)[:, 1] > threshold).long().cpu().tolist())
            pred_ai.extend(y_ai.argmax(-1).cpu().tolist())
            true_fake.extend(labels_fake.cpu().tolist())
            true_ai.extend(labels_ai.cpu().tolist())
    if not total_samples:
        raise ValueError("Validation set is empty; cannot select a checkpoint.")
    result = {
        "val_loss": total_loss / total_samples,
        "fake_macro_f1": float(f1_score(true_fake, pred_fake, labels=[0, 1], average="macro", zero_division=0)),
        "ai_macro_f1": float(f1_score(true_ai, pred_ai, labels=[0, 1], average="macro", zero_division=0)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite validation metrics; no checkpoint will be selected.")
    return result


class SelectionTracker:
    """F1 selects the output, while loss alone controls early stopping."""

    def __init__(self, patience):
        if patience < 1:
            raise ValueError("Early stopping patience must be positive.")
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_f1 = float("-inf")
        self.loss_epoch = self.f1_epoch = None

    def update(self, epoch, loss, f1):
        if not math.isfinite(loss) or not math.isfinite(f1):
            raise ValueError("Non-finite selection metrics.")
        loss_improved = loss < self.best_loss - 1e-6
        # Numerical ties retain the earlier epoch; no secondary task tuning.
        f1_improved = f1 > self.best_f1 + 1e-12
        if loss_improved:
            self.best_loss, self.loss_epoch, self.counter = loss, epoch, 0
        else:
            self.counter += 1
        if f1_improved:
            self.best_f1, self.f1_epoch = f1, epoch
        return loss_improved, f1_improved, self.counter >= self.patience


def train_model_cat(model, train_loader, val_loader, cfg, optimizer, output_dir=None):
    device = cfg.DEVICE if cfg.DEVICE != "cuda" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    threshold = float(cfg.FAKE_THRESHOLD)
    if threshold != 0.55:
        raise ValueError("This experiment freezes FAKE_THRESHOLD at 0.55.")
    root = Path(output_dir or cfg.OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best_checkpoint.pt"
    loss_root = root / "loss_selected"
    loss_root.mkdir(exist_ok=True)
    loss_path = loss_root / "best_checkpoint.pt"
    if best_path.exists() or loss_path.exists():
        raise FileExistsError("Use a fresh attempt directory; checkpoint overwrite refused.")
    if int(cfg.EPOCHS) < 1:
        raise ValueError("At least one training epoch is required.")
    focal = FocalLossCE(alpha=getattr(cfg, "FOCAL_ALPHA", 0.8),
                       gamma=getattr(cfg, "FOCAL_GAMMA", 2.0),
                       weight=getattr(cfg, "FAKE_CLASS_WEIGHT", None))
    history = {name: [] for name in (
        "train_loss", "train_task_loss", "attention_kl", "val_loss",
        "val_fake_macro_f1", "val_ai_macro_f1")}
    tracker = SelectionTracker(int(getattr(cfg, "EARLYSTOP_PATIENCE", 3)))
    for epoch in range(1, int(cfg.EPOCHS) + 1):
        print(f"\n===== Epoch {epoch}/{cfg.EPOCHS} =====", flush=True)
        stats = train_one_epoch(model, train_loader, optimizer, device, cfg, focal)
        valid = evaluate_selection_epoch(model, val_loader, device, focal, threshold)
        for name, value in (("train_loss", stats["loss"]),
                            ("train_task_loss", stats["task_loss"]),
                            ("attention_kl", stats["attention_kl"]),
                            ("val_loss", valid["val_loss"]),
                            ("val_fake_macro_f1", valid["fake_macro_f1"]),
                            ("val_ai_macro_f1", valid["ai_macro_f1"])):
            history[name].append(value)
        loss_better, f1_better, stop = tracker.update(epoch, valid["val_loss"], valid["fake_macro_f1"])
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "train_stats": stats,
            "val_loss": valid["val_loss"], "best_val_loss": tracker.best_loss,
            "val_fake_macro_f1": valid["fake_macro_f1"],
            "val_ai_macro_f1": valid["ai_macro_f1"],
            "best_val_fake_macro_f1": tracker.best_f1,
            "history": history, "config": _config_dict(cfg),
            "early_stopping_metric": "val_loss", "selection_threshold": threshold,
        }
        if loss_better:
            torch.save({**checkpoint, "selection_metric": "val_loss",
                        "config": {**checkpoint["config"], "CHECKPOINT_SELECTION": "val_loss"},
                        "selection_score": valid["val_loss"]}, str(loss_path))
        if f1_better:
            torch.save({**checkpoint, "selection_metric": "val_fake_macro_f1",
                        "selection_score": valid["fake_macro_f1"]}, str(best_path))
        save_json(history, root / "history.json")
        save_json({"f1_selected_epoch": tracker.f1_epoch,
                   "loss_selected_epoch": tracker.loss_epoch,
                   "best_val_fake_macro_f1": tracker.best_f1,
                   "best_val_loss": tracker.best_loss,
                   "epochs_run": epoch, "reference_threshold": threshold,
                   "early_stopping_metric": "val_loss",
                   "early_stopping_patience": tracker.patience,
                   "f1_tie_rule": "Keep the earlier epoch within 1e-12.",
                   "same_training_trajectory": True}, root / "selection_summary.json")
        print(f"Epoch {epoch}: val_loss={valid['val_loss']:.6f}, "
              f"fake_macro_f1={valid['fake_macro_f1']:.6f}, "
              f"ai_macro_f1={valid['ai_macro_f1']:.6f}; "
              f"best F1 epoch={tracker.f1_epoch}, best loss epoch={tracker.loss_epoch}", flush=True)
        if stop:
            print("Early stopping triggered (unchanged validation-loss rule).", flush=True)
            break
    del checkpoint
    selected = torch.load(str(best_path), map_location="cpu", mmap=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    print(f"Loaded F1-selected checkpoint generated by this run: {best_path}", flush=True)
    return model, history
