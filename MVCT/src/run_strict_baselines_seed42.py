"""Strict seed-42 controlled baselines for the frozen MVCT protocol.

Stage 1 trains four predeclared baselines and selects neural checkpoints only by
combined validation loss.  Stage 2 (``--evaluate-test``) is blocked until every
baseline is frozen, then evaluates the saved test split once at Fake/Real
threshold 0.55.  The baseline set was fixed after the main-model test had been
observed; that timing is recorded rather than presented as preregistration.
"""
import argparse
import csv
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import run_locked_three_channel_test as locked
import run_strict_replication as strict
import run_strict_three_channel_ablation as three
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEED = 42
FAKE_THRESHOLD = 0.55
VARIANTS = ("tfidf_logreg", "textcnn", "bert", "roberta")
NEURAL_VARIANTS = ("textcnn", "bert", "roberta")
MEASURES = ("fake_accuracy", "fake_macro_f1", "ai_accuracy", "ai_macro_f1")
MODEL_SPECS = {
    "tfidf_logreg": {
        "kind": "classical", "max_features": 50000, "ngram_range": [1, 2],
        "min_df": 2, "sublinear_tf": True, "C": 1.0, "max_iter": 1000,
    },
    "textcnn": {
        "kind": "neural", "tokenizer": "bert", "embedding_dim": 128,
        "filters": 128, "kernels": [3, 4, 5], "dropout": 0.1,
        "lr": 1e-3, "weight_decay": 1e-4,
    },
    "bert": {
        "kind": "neural", "tokenizer": "bert", "dropout": 0.1,
        "lr": 2e-5, "weight_decay": 0.01,
    },
    "roberta": {
        "kind": "neural", "tokenizer": "roberta", "dropout": 0.1,
        "lr": 2e-5, "weight_decay": 0.01,
    },
}
TRAIN_ARTIFACTS = {
    "tfidf_logreg": (
        "baseline_model.joblib", "config.json", "validation_metrics.json",
        "validation_predictions.json",
    ),
    "neural": (
        "best_checkpoint.pt", "history.json", "config.json",
        "validation_metrics.json", "validation_predictions.json",
    ),
}
TEST_ARTIFACTS = ("test_metrics.json", "test_predictions.json", "metadata.json")


def metrics_row(variant, metrics, output, checkpoint_epoch=None, val_loss=None):
    return {
        "variant": variant,
        "seed": SEED,
        "checkpoint_epoch": checkpoint_epoch,
        "val_loss": val_loss,
        "fake_threshold": FAKE_THRESHOLD,
        "fake_accuracy": metrics["fake_report"]["accuracy"],
        "fake_macro_f1": metrics["fake_report"]["macro avg"]["f1-score"],
        "ai_accuracy": metrics["ai_report"]["accuracy"],
        "ai_macro_f1": metrics["ai_report"]["macro avg"]["f1-score"],
        "output_dir": str(Path(output).resolve()),
    }


def predictions_and_metrics(ids, true_fake, true_ai, prob_fake, prob_ai):
    from utils import compute_metrics_twoheads

    pred_fake = (np.asarray(prob_fake) > FAKE_THRESHOLD).astype(int)
    pred_ai = (np.asarray(prob_ai) > 0.5).astype(int)
    metrics = compute_metrics_twoheads(
        pred_fake, true_fake, pred_ai, true_ai
    )
    rows = [
        {
            "sample_id": int(sample_id),
            "true_fake": int(yf), "pred_fake": int(pf),
            "prob_fake": float(qf),
            "true_ai": int(ya), "pred_ai": int(pa),
            "prob_ai": float(qa),
        }
        for sample_id, yf, pf, qf, ya, pa, qa in zip(
            ids, true_fake, pred_fake, prob_fake,
            true_ai, pred_ai, prob_ai,
        )
    ]
    return metrics, rows


def save_evaluation(output, prefix, ids, yf, ya, pf, pa):
    metrics, rows = predictions_and_metrics(ids, yf, ya, pf, pa)
    strict.atomic_json(output / f"{prefix}_metrics.json", metrics)
    strict.atomic_json(output / f"{prefix}_predictions.json", rows)
    return metrics


def combined_probability_loss(yf, ya, pf, pa, alpha=.8, gamma=2.0):
    yf, ya = np.asarray(yf, dtype=int), np.asarray(ya, dtype=int)
    pf = np.clip(np.asarray(pf), 1e-8, 1 - 1e-8)
    pa = np.clip(np.asarray(pa), 1e-8, 1 - 1e-8)
    true_pf = np.where(yf == 1, pf, 1 - pf)
    true_pa = np.where(ya == 1, pa, 1 - pa)
    fake = np.mean(alpha * (1 - true_pf) ** gamma * (-np.log(true_pf)))
    ai = np.mean(-np.log(true_pa))
    return float(fake + 0.5 * ai)


def prepare(baseline, channel_suite, three_suite, locked_suite, roberta_model):
    """Verify frozen inputs/main results and seal baseline choices."""
    source_protocol, reference, train, val, _ = strict.prepare(baseline, (SEED,))
    channel_protocol = strict.verified_protocol(channel_suite / "protocol.json")
    three_protocol = strict.verified_protocol(three_suite / "protocol.json")
    three.verify_source_protocol(source_protocol, channel_protocol)
    three.verify_source_protocol(source_protocol, three_protocol)
    locked_protocol = strict.verified_protocol(locked_suite / "protocol.json")
    if (
        locked_protocol.get("channel_protocol_id") != channel_protocol["protocol_id"]
        or locked_protocol.get("three_channel_protocol_id") != three_protocol["protocol_id"]
    ):
        raise ValueError("Locked main-model test does not match the verified sources.")
    summary_path = locked_suite / "locked_test_summary.json"
    main_summary = strict.read_json(summary_path)
    if main_summary.get("completed_evaluations") != 5:
        raise ValueError("The five locked main-model evaluations must be complete.")

    bert_path = Path(reference.BERT_MODEL)
    roberta_path = Path(roberta_model)
    for name, path in (("BERT", bert_path), ("RoBERTa", roberta_path)):
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"Local {name} model is incomplete: {path.resolve()}"
            )

    payload = {
        key: value for key, value in source_protocol.items()
        if key != "protocol_id"
    }
    entries = dict(payload["entry_sha256"])
    for name in (
        Path(__file__).name, "baseline_models.py", "dataset_cat.py",
        "train_cat.py", "utils.py",
    ):
        entries[name] = strict.file_hash(SOURCE_DIR / name)
    specs = {name: dict(value) for name, value in MODEL_SPECS.items()}
    specs["bert"]["model_path"] = str(bert_path.resolve())
    specs["textcnn"]["model_path"] = str(bert_path.resolve())
    specs["roberta"]["model_path"] = str(roberta_path.resolve())
    payload.update({
        "purpose": "Controlled seed-42 external and architecture baselines",
        "seeds": [SEED], "variants": list(VARIANTS),
        "entry_sha256": entries,
        "channel_protocol_id": channel_protocol["protocol_id"],
        "three_channel_protocol_id": three_protocol["protocol_id"],
        "locked_main_test_protocol_id": locked_protocol["protocol_id"],
        "locked_main_summary_sha256": strict.file_hash(summary_path),
        "model_specs": specs,
        "training_protocol": {
            "input": "title + body + description with the frozen token budgets",
            "seed": SEED, "batch_size": int(reference.BATCH_SIZE),
            "epochs": int(reference.EPOCHS),
            "early_stopping_patience": int(reference.EARLYSTOP_PATIENCE),
            "fake_loss": {"name": "focal", "alpha": float(reference.FOCAL_ALPHA),
                          "gamma": float(reference.FOCAL_GAMMA)},
            "ai_loss": "0.5 * cross_entropy",
            "checkpoint_selection": "minimum combined validation task loss",
            "fake_threshold": FAKE_THRESHOLD,
            "test_gate": "test evaluation forbidden until all four validation-selected baselines are frozen",
        },
        "selection": "Fixed model specifications; no hyperparameter grid or test-based selection",
        "primary_metric": "Test Fake/Real macro-F1 at fixed strict > 0.55",
        "threshold_scan": "Forbidden",
        "test_evaluation": "Second stage only after four completed training records",
        "timing_disclosure": (
            "The baseline set was fixed after the main-model seed-42 test was observed; "
            "baseline specifications were not chosen from baseline test outcomes."
        ),
    })
    return strict.seal(payload), reference, len(train), len(val)


def model_paths(protocol, variant):
    spec = protocol["model_specs"][variant]
    return spec, spec.get("model_path")


def restore_frame(config, split):
    frame, split_path, _, _, _ = locked.restore_saved_split(config, split)
    return frame, split_path


def make_loader(frame, tokenizer, config, shuffle):
    import torch
    from torch.utils.data import DataLoader
    from baseline_models import BaselineNewsDataset

    dataset = BaselineNewsDataset(
        frame, tokenizer, config.MAX_LEN_TITLE,
        config.MAX_LEN_TEXT, config.MAX_LEN_DESC,
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle,
        num_workers=0, generator=generator if shuffle else None,
    )


def build_neural(variant, protocol, tokenizer):
    from baseline_models import DualHeadTextCNN, DualHeadTransformer

    spec, model_path = model_paths(protocol, variant)
    if variant == "textcnn":
        return DualHeadTextCNN(
            len(tokenizer), tokenizer.pad_token_id,
            embedding_dim=spec["embedding_dim"], filters=spec["filters"],
            kernels=tuple(spec["kernels"]), dropout=spec["dropout"],
        )
    return DualHeadTransformer(model_path, dropout=spec["dropout"])


def neural_probabilities(model, loader, device):
    import torch

    ids, yf, ya, pf, pa = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            fake, ai = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            ids.extend(batch["sample_id"].tolist())
            yf.extend(batch["label_fake"].tolist())
            ya.extend(batch["label_ai"].tolist())
            pf.extend(fake.softmax(-1)[:, 1].cpu().tolist())
            pa.extend(ai.softmax(-1)[:, 1].cpu().tolist())
    return ids, yf, ya, pf, pa


def train_neural(variant, protocol, config, output):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer
    from train_cat import FocalLossCE
    from utils import set_seed

    spec, model_path = model_paths(protocol, variant)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    train_frame, _ = restore_frame(config, "train")
    val_frame, _ = restore_frame(config, "val")
    train_loader = make_loader(train_frame, tokenizer, config, True)
    val_loader = make_loader(val_frame, tokenizer, config, False)
    set_seed(SEED)
    model = build_neural(variant, protocol, tokenizer).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"]
    )
    focal = FocalLossCE(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    history = {"train_loss": [], "val_loss": []}
    best, best_epoch, stale = math.inf, None, 0
    checkpoint_path = output / "best_checkpoint.pt"

    for epoch in range(1, int(config.EPOCHS) + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            fake, ai = model(
                batch["input_ids"].to("cuda"),
                batch["attention_mask"].to("cuda"),
            )
            yf = batch["label_fake"].to("cuda")
            ya = batch["label_ai"].to("cuda")
            loss = focal(fake, yf) + 0.5 * functional.cross_entropy(ai, ya)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yf)
            count += len(yf)
        train_loss = total / count
        values = neural_probabilities(model, val_loader, "cuda")
        val_loss = combined_probability_loss(
            values[1], values[2], values[3], values[4],
            config.FOCAL_ALPHA, config.FOCAL_GAMMA,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"{variant} epoch {epoch}/{config.EPOCHS}: "
            f"train={train_loss:.6f}, val={val_loss:.6f}", flush=True,
        )
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save({
                "variant": variant, "seed": SEED, "epoch": epoch,
                "val_loss": val_loss, "model_state_dict": model.state_dict(),
                "model_spec": spec,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= int(config.EARLYSTOP_PATIENCE):
                break
        strict.atomic_json(output / "history.json", history)

    checkpoint = torch.load(str(checkpoint_path), map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    values = neural_probabilities(model, val_loader, "cuda")
    metrics = save_evaluation(output, "validation", *values)
    strict.atomic_json(output / "history.json", history)
    del model, optimizer, train_loader, val_loader, train_frame, val_frame
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, best_epoch, best


def joined_text(frame):
    return (
        frame.title.astype(str) + " [SEP] " + frame.text.astype(str)
        + " [SEP] " + frame.desc.astype(str)
    ).tolist()


def train_logistic(protocol, config, output):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    spec = protocol["model_specs"]["tfidf_logreg"]
    train, _ = restore_frame(config, "train")
    val, _ = restore_frame(config, "val")
    vectorizer = TfidfVectorizer(
        lowercase=True, max_features=spec["max_features"],
        ngram_range=tuple(spec["ngram_range"]), min_df=spec["min_df"],
        sublinear_tf=spec["sublinear_tf"], dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(joined_text(train))
    x_val = vectorizer.transform(joined_text(val))
    classifiers = {}
    probabilities = []
    for target in ("label_fake", "label_ai"):
        classifier = LogisticRegression(
            C=spec["C"], max_iter=spec["max_iter"], solver="liblinear",
            random_state=SEED,
        )
        classifier.fit(x_train, train[target].astype(int))
        classifiers[target] = classifier
        probabilities.append(classifier.predict_proba(x_val)[:, 1])
    joblib.dump(
        {"vectorizer": vectorizer, "classifiers": classifiers, "spec": spec},
        output / "baseline_model.joblib",
    )
    metrics = save_evaluation(
        output, "validation", val.ID.tolist(),
        val.label_fake.astype(int).tolist(), val.label_ai.astype(int).tolist(),
        probabilities[0].tolist(), probabilities[1].tolist(),
    )
    val_loss = combined_probability_loss(
        val.label_fake, val.label_ai, probabilities[0], probabilities[1],
        config.FOCAL_ALPHA, config.FOCAL_GAMMA,
    )
    return metrics, None, val_loss


def train_worker(args):
    import torch

    protocol = strict.verified_protocol(Path(args.suite_dir) / "protocol.json")
    if strict.file_hash(__file__) != protocol["entry_sha256"][Path(__file__).name]:
        raise ValueError("Baseline runner changed after protocol sealing.")
    variant = args.variant
    slot = Path(args.suite_dir) / variant
    output = Path(args.output).resolve()
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid baseline output or completed slot.")
    config = SimpleNamespace(**protocol["reference_config"])
    config.BERT_MODEL = protocol["model_specs"]["bert"]["model_path"]
    strict.atomic_json(output / "config.json", vars(config))
    if variant == "tfidf_logreg":
        metrics, epoch, loss = train_logistic(protocol, config, output)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Neural baselines require CUDA in this protocol.")
        metrics, epoch, loss = train_neural(variant, protocol, config, output)
    row = metrics_row(variant, metrics, output, epoch, loss)
    names = TRAIN_ARTIFACTS["tfidf_logreg" if variant == "tfidf_logreg" else "neural"]
    row.update({
        "protocol_id": protocol["protocol_id"], "stage": "validation_frozen",
        "artifact_sha256": {name: strict.file_hash(output / name) for name in names},
    })
    strict.atomic_json(output / "worker_result.json", row)
    print("Baseline validation checkpoint frozen:", variant, flush=True)


def training_record(slot, protocol, variant, record=None):
    done = slot / "completed.json"
    if record is None and not done.exists():
        return None
    row = strict.read_json(done) if record is None else record
    if (row.get("protocol_id"), row.get("variant"), row.get("seed")) != (
        protocol["protocol_id"], variant, SEED
    ) or row.get("stage") != "validation_frozen":
        raise ValueError(f"Invalid frozen baseline record: {done}")
    output = Path(row["output_dir"]).resolve()
    if output.parent != slot.resolve() or not output.name.startswith("attempt_"):
        raise ValueError("Frozen baseline record points outside its slot.")
    expected = TRAIN_ARTIFACTS[
        "tfidf_logreg" if variant == "tfidf_logreg" else "neural"
    ]
    if set(row["artifact_sha256"]) != set(expected):
        raise ValueError("Frozen baseline artifact set is incomplete.")
    for name, digest in row["artifact_sha256"].items():
        if strict.file_hash(output / name) != digest:
            raise ValueError(f"Frozen baseline artifact changed: {output / name}")
    return row


def evaluate_test_worker(args):
    import joblib
    import torch
    from transformers import AutoTokenizer

    protocol = strict.verified_protocol(Path(args.suite_dir) / "protocol.json")
    variant = args.variant
    train_row = training_record(
        Path(args.suite_dir) / variant, protocol, variant
    )
    output = Path(args.output).resolve()
    test_slot = Path(args.suite_dir) / "locked_test" / variant
    if output.parent != test_slot.resolve() or (test_slot / "completed.json").exists():
        raise ValueError("Invalid or completed baseline test slot.")
    config = SimpleNamespace(**strict.read_json(Path(train_row["output_dir"]) / "config.json"))
    frame, split_path = restore_frame(config, "test")
    if variant == "tfidf_logreg":
        bundle = joblib.load(Path(train_row["output_dir"]) / "baseline_model.joblib")
        matrix = bundle["vectorizer"].transform(joined_text(frame))
        pf = bundle["classifiers"]["label_fake"].predict_proba(matrix)[:, 1]
        pa = bundle["classifiers"]["label_ai"].predict_proba(matrix)[:, 1]
        values = (
            frame.ID.tolist(), frame.label_fake.astype(int).tolist(),
            frame.label_ai.astype(int).tolist(), pf.tolist(), pa.tolist(),
        )
    else:
        _, model_path = model_paths(protocol, variant)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        loader = make_loader(frame, tokenizer, config, False)
        model = build_neural(variant, protocol, tokenizer)
        checkpoint = torch.load(
            str(Path(train_row["output_dir"]) / "best_checkpoint.pt"),
            map_location="cpu", mmap=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        values = neural_probabilities(model, loader, device)
    metrics = save_evaluation(output, "test", *values)
    metadata = {
        "protocol_id": protocol["protocol_id"], "variant": variant,
        "seed": SEED, "split": "test", "saved_split": str(split_path.resolve()),
        "fake_threshold": FAKE_THRESHOLD, "test_based_selection": False,
        "training_completion_sha256": strict.file_hash(
            Path(args.suite_dir) / variant / "completed.json"
        ),
    }
    strict.atomic_json(output / "metadata.json", metadata)
    row = metrics_row(
        variant, metrics, output, train_row["checkpoint_epoch"], train_row["val_loss"]
    )
    row.update({
        "protocol_id": protocol["protocol_id"], "stage": "locked_test",
        "artifact_sha256": {
            name: strict.file_hash(output / name) for name in TEST_ARTIFACTS
        },
    })
    strict.atomic_json(output / "worker_result.json", row)
    print("Locked baseline test completed:", variant, flush=True)


def test_record(slot, protocol, variant, record=None):
    done = slot / "completed.json"
    if record is None and not done.exists():
        return None
    row = strict.read_json(done) if record is None else record
    if (row.get("protocol_id"), row.get("variant"), row.get("seed")) != (
        protocol["protocol_id"], variant, SEED
    ) or row.get("stage") != "locked_test":
        raise ValueError(f"Invalid locked baseline test record: {done}")
    output = Path(row["output_dir"]).resolve()
    if output.parent != slot.resolve() or not output.name.startswith("evaluation_"):
        raise ValueError("Locked baseline test record points outside its slot.")
    if set(row["artifact_sha256"]) != set(TEST_ARTIFACTS):
        raise ValueError("Locked baseline test artifact set is incomplete.")
    for name, digest in row["artifact_sha256"].items():
        if strict.file_hash(output / name) != digest:
            raise ValueError(f"Locked baseline test artifact changed: {output / name}")
    return row


def summarize(rows, stage):
    by_name = {row["variant"]: row for row in rows}
    if len(by_name) != len(rows) or any(name not in VARIANTS for name in by_name):
        raise ValueError("Duplicate or unplanned baseline result.")
    return {
        "stage": stage, "seed": SEED, "planned": len(VARIANTS),
        "completed": len(rows), "fake_threshold": FAKE_THRESHOLD,
        "by_variant": {
            name: (
                {key: by_name[name][key] for key in (*MEASURES, "val_loss")}
                if name in by_name else None
            ) for name in VARIANTS
        },
        "note": (
            "Fixed seed-42 controlled baselines. Validation loss selects neural "
            "checkpoints; no baseline test result changes a model or threshold."
        ),
    }


def write_summary(root, rows, stage):
    prefix = "baseline_validation" if stage == "validation_frozen" else "baseline_test"
    strict.atomic_json(root / f"{prefix}_summary.json", summarize(rows, stage))
    temporary = root / f"{prefix}_results.csv.tmp"
    columns = (
        "variant", "seed", "checkpoint_epoch", "val_loss", "fake_threshold",
        *MEASURES, "output_dir",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, root / f"{prefix}_results.csv")


def launch_child(arguments, lock, log=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *arguments]
    child = subprocess.Popen(
        command, cwd=str(SOURCE_DIR), env=strict.strict_environment(),
        stdout=log, stderr=subprocess.STDOUT if log else None,
        **lock.subprocess_options(),
    )
    code = child.wait()
    if code:
        raise RuntimeError(
            f"Baseline worker exited with code {code}; inspect worker.log."
        )


def main(argv=None):
    from config import cfg

    replications = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-suite", default=str(replications / "missing_evidence_v1"))
    parser.add_argument("--channel-suite", default=str(replications / "strict_channel_ablation_seed42_v1"))
    parser.add_argument("--three-suite", default=str(replications / "strict_three_channel_ablation_seed42_v1"))
    parser.add_argument("--locked-main-suite", default=str(replications / "locked_three_channel_test_v1"))
    parser.add_argument("--suite-dir", default=str(replications / "strict_baselines_seed42_v1"))
    parser.add_argument("--roberta-model", default="models/roberta-base")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-new-runs", type=int, default=len(VARIANTS))
    parser.add_argument("--worker", choices=("preflight", "train", "test"), help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    paths = {
        name: Path(getattr(args, name)).resolve() for name in
        ("baseline_suite", "channel_suite", "three_suite", "locked_main_suite", "suite_dir")
    }
    for name, value in paths.items():
        setattr(args, name, str(value))
    args.roberta_model = str(Path(args.roberta_model).resolve())

    if args.worker == "preflight":
        protocol, _, train_n, val_n = prepare(
            paths["baseline_suite"], paths["channel_suite"], paths["three_suite"],
            paths["locked_main_suite"], Path(args.roberta_model),
        )
        strict.atomic_json(args.output, {"protocol": protocol, "train_n": train_n, "val_n": val_n})
        return
    if args.worker == "train":
        train_worker(args); return
    if args.worker == "test":
        evaluate_test_worker(args); return

    root = paths["suite_dir"]
    sources = [paths[key] for key in ("baseline_suite", "channel_suite", "three_suite", "locked_main_suite")]
    locked.protect_output(root, sources)
    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(prefix="mvct_baseline_preflight_") as temporary:
            result_path = Path(temporary) / "result.json"
            launch_child([
                "--worker", "preflight", "--baseline-suite", args.baseline_suite,
                "--channel-suite", args.channel_suite, "--three-suite", args.three_suite,
                "--locked-main-suite", args.locked_main_suite,
                "--roberta-model", args.roberta_model, "--output", str(result_path),
            ], lock)
            result = strict.read_json(result_path)
        protocol = result["protocol"]
        if (root / "protocol.json").exists():
            strict.same_protocol(strict.verified_protocol(root / "protocol.json"), protocol)
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty baseline suite has no protocol; use a new directory.")

        train_rows, train_pending = [], []
        for variant in VARIANTS:
            row = training_record(root / variant, protocol, variant)
            (train_pending if row is None else train_rows).append(variant if row is None else row)
        test_rows, test_pending = [], []
        for variant in VARIANTS:
            row = test_record(root / "locked_test" / variant, protocol, variant)
            (test_pending if row is None else test_rows).append(variant if row is None else row)

        print(
            f"Preflight passed: seed 42; train/val={result['train_n']}/{result['val_n']}; "
            f"frozen baselines={len(train_rows)}/4; test evaluations={len(test_rows)}/4.",
            flush=True,
        )
        if args.check_only:
            print("Check only: no baseline suite files written."); return
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)

        if not args.evaluate_test:
            planned = train_pending[:args.max_new_runs]
            write_summary(root, train_rows, "validation_frozen")
            for variant in planned:
                slot = root / variant
                output = slot / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                output.mkdir(parents=True, exist_ok=False)
                print(f"Train {variant}; log: {output / 'worker.log'}", flush=True)
                with (output / "worker.log").open("w", encoding="utf-8") as log:
                    launch_child([
                        "--worker", "train", "--suite-dir", str(root),
                        "--variant", variant, "--output", str(output),
                    ], lock, log)
                row = strict.read_json(output / "worker_result.json")
                training_record(slot, protocol, variant, row)
                strict.atomic_json(slot / "completed.json", row)
                train_rows.append(row); write_summary(root, train_rows, "validation_frozen")
                print(f"Frozen {variant}; total={len(train_rows)}/4", flush=True)
            if len(train_rows) == len(VARIANTS):
                print("All baselines frozen on validation. Run again with --evaluate-test.")
            else:
                print("Invocation limit reached; repeat the same training command.")
            return

        if len(train_rows) != len(VARIANTS):
            raise RuntimeError("Test gate closed: freeze all four baselines first.")
        planned = test_pending[:args.max_new_runs]
        write_summary(root, test_rows, "locked_test")
        for variant in planned:
            slot = root / "locked_test" / variant
            output = slot / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            output.mkdir(parents=True, exist_ok=False)
            print(f"Evaluate {variant}; log: {output / 'worker.log'}", flush=True)
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child([
                    "--worker", "test", "--suite-dir", str(root),
                    "--variant", variant, "--output", str(output),
                ], lock, log)
            row = strict.read_json(output / "worker_result.json")
            test_record(slot, protocol, variant, row)
            strict.atomic_json(slot / "completed.json", row)
            test_rows.append(row); write_summary(root, test_rows, "locked_test")
            print(f"Tested {variant}; total={len(test_rows)}/4", flush=True)
        if len(test_rows) == len(VARIANTS):
            print("Controlled baseline suite completed:", root)
        else:
            print("Invocation limit reached; repeat --evaluate-test.")


if __name__ == "__main__":
    main()
