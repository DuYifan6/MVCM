"""Extract final-model diagnostic tensors for a preselected representative case.

This script reads one ID from `select_representative_case.py`, restores the
sealed test split and existing MVC cache, and performs read-only forward passes
through the frozen full MVCT and w/o-MVCM checkpoints. It does not retrain,
change thresholds, rebuild caches, or write into locked experiment suites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def completion_metadata(suite, variant):
    completion_path = Path(suite) / variant / "completed.json"
    completion = read_json(completion_path)
    evaluation_dir = Path(completion["evaluation_dir"])
    metadata_path = evaluation_dir / "evaluation_metadata.json"
    metadata = read_json(metadata_path)
    return completion_path, completion, evaluation_dir, metadata_path, metadata


def build_model(config, checkpoint, device):
    from model_cat import MVCTTokenCATModel

    model = MVCTTokenCATModel(
        bert_model_name=config.BERT_MODEL,
        mvc_weights=(
            config.SEM_WEIGHT,
            config.FACT_WEIGHT,
            config.LOGIC_WEIGHT,
            config.LOCAL_WEIGHT,
        ),
        cat_layers=config.CAT_LAYERS,
        cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER,
        dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT,
        learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=getattr(config, "MASK_MISSING_EVIDENCE", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def extract_model_state(model, item, device, fake_threshold):
    import torch

    sequence = {
        key: value.unsqueeze(0).to(device) for key, value in item["sequence"].items()
    }
    mvc = item["mvc"].unsqueeze(0).float().to(device)
    with torch.no_grad():
        fake_logits, ai_logits, attention = model(sequence, mvc)
        fake_probability = torch.softmax(fake_logits, dim=1)[0, 1]
        ai_probability = torch.softmax(ai_logits, dim=1)[0, 1]
        view_attention = model.aggregate_attention_by_view(
            attention, sequence["view_ids"], sequence["attention_mask"]
        )[0]
        fused_bias = model.combine_mvc(mvc)[0]
    return {
        "pred_fake": int((fake_probability > fake_threshold).item()),
        "prob_fake": float(fake_probability.item()),
        "pred_ai": int(torch.argmax(ai_logits, dim=1)[0].item()),
        "prob_ai": float(ai_probability.item()),
        "channel_mask": [bool(value) for value in model.channel_mask.cpu().tolist()],
        "masked_softmax_channel_weights": [
            float(value) for value in model.mvc_weights.detach().cpu().tolist()
        ],
        "cat_lambda": [
            float(layer.attention.lambda_bias.detach().cpu().item())
            for layer in model.cat_layers
        ],
        "fused_signed_view_bias": fused_bias.detach().cpu().tolist(),
        "final_cat_view_attention": view_attention.detach().cpu().tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--main-suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=("primary", "boundary"), default="primary")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    import run_strict_replication as strict
    from dataset_cat import MVCNewsDatasetCAT
    from run_locked_three_channel_test import FAKE_THRESHOLD, restore_saved_split

    selection = read_json(args.selection)
    if args.case == "primary":
        chosen = selection["primary_recommendation"]["selected"]
    else:
        chosen = selection["balancing_failure_case"]["selected"]
    if not chosen:
        raise ValueError(f"Selection report contains no {args.case} case")
    sample_id = int(chosen["sample_id"])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    variants = {}
    reference_frame = None
    reference_item = None
    reference_config = None
    cache_path = None
    for variant in ("three_channel_full", "without_mvcm"):
        completion_path, completion, evaluation_dir, metadata_path, metadata = completion_metadata(
            args.main_suite, variant
        )
        checkpoint_path = Path(metadata["checkpoint"])
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
        config = SimpleNamespace(**checkpoint["config"])
        if float(config.FAKE_THRESHOLD) != float(FAKE_THRESHOLD):
            raise ValueError(f"Unexpected threshold in {variant} checkpoint")
        frame, split_path, _, _, cache_signature = restore_saved_split(config, "test")
        selected_frame = frame.loc[frame["ID"].astype(int) == sample_id].copy()
        if len(selected_frame) != 1:
            raise ValueError(f"Expected one saved-test row for ID {sample_id}, found {len(selected_frame)}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.BERT_MODEL, local_files_only=True, use_fast=True
        )
        dataset = MVCNewsDatasetCAT(
            selected_frame,
            tokenizer,
            config.MVC_CACHE_DIR,
            config.MAX_LEN_TITLE,
            config.MAX_LEN_TEXT,
            config.MAX_LEN_DESC,
        )
        item = dataset[0]
        model = build_model(config, checkpoint, device)
        state = extract_model_state(model, item, device, float(FAKE_THRESHOLD))
        state.update(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": strict.file_hash(checkpoint_path),
                "completion_sha256": strict.file_hash(completion_path),
                "evaluation_metadata_sha256": strict.file_hash(metadata_path),
                "saved_split": str(Path(split_path).resolve()),
                "saved_split_sha256": strict.file_hash(split_path),
                "cache_signature": cache_signature,
            }
        )
        variants[variant] = state
        if reference_frame is None:
            reference_frame = selected_frame.iloc[0]
            reference_item = item
            reference_config = config
            cache_path = Path(selected_frame.iloc[0]["mvc_path"])
        del model, dataset, tokenizer, checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()

    mvc = reference_item["mvc"].cpu()
    row = reference_frame
    report = {
        "purpose": "Read-only diagnostic extraction for an illustrative case",
        "selection_is_post_hoc": True,
        "case_type": args.case,
        "sample_id": sample_id,
        "group": chosen["group"],
        "true_fake": int(reference_item["label_fake"].item()),
        "true_ai": int(reference_item["label_ai"].item()),
        "text": {
            "title": str(row["title"]),
            "description": str(row["desc"]),
            "body_excerpt": str(row["text"])[:1000],
            "body_characters": len(str(row["text"])),
        },
        "mvc_channels": {
            "semantic": mvc[:, :, 0].tolist(),
            "factual_cached_only": mvc[:, :, 1].tolist(),
            "logical": mvc[:, :, 2].tolist(),
            "local": mvc[:, :, 3].tolist(),
        },
        "active_channels_for_final_fusion": ["semantic", "logical", "local"],
        "variants": variants,
        "provenance": {
            "selection_report": str(Path(args.selection).resolve()),
            "selection_report_sha256": strict.file_hash(args.selection),
            "mvc_cache": str(cache_path.resolve()),
            "mvc_cache_sha256": strict.file_hash(cache_path),
            "bert_model": str(reference_config.BERT_MODEL),
            "device": device,
        },
        "interpretation_boundary": (
            "The matrices and final-layer view attention are descriptive internal states. "
            "They do not establish attention faithfulness or a uniquely identified causal mechanism."
        ),
    }
    write_json(args.output, report)
    print(f"Extracted {args.case} case sample_id={sample_id} on {device}")
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
