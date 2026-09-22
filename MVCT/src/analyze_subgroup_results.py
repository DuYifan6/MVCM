"""Extract HF/MF/HR/MR results from frozen per-sample predictions.

This analysis never changes predictions or thresholds.  It joins each model's
``test_predictions.json`` to ``data_all.csv`` by sample ID, verifies that the
stored labels agree, and reports conditional correctness for the two tasks and
joint exact-match accuracy with Wilson 95% confidence intervals.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


GROUP_ORDER = ("HR", "MR", "HF", "MF")
EXPECTED_LABELS = {
    "HR": (0, 0),
    "MR": (0, 1),
    "HF": (1, 0),
    "MF": (1, 1),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def parse_prediction_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    return name.strip(), Path(path.strip())


def read_predictions(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "sample_id", "true_fake", "pred_fake", "true_ai", "pred_ai"
    }
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Prediction file is empty or not a JSON list: {path}")
    frame = pd.DataFrame(rows)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file lacks columns {missing}: {path}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs in {path}: {duplicates[:5]}")
    for column in required:
        frame[column] = frame[column].astype(int)
    for column in ("true_fake", "pred_fake", "true_ai", "pred_ai"):
        invalid = sorted(set(frame[column]) - {0, 1})
        if invalid:
            raise ValueError(f"Non-binary values in {column} of {path}: {invalid}")
    return frame


def load_labels(data_path):
    data = pd.read_csv(data_path, usecols=["ID", "type", "label_fake", "label_ai"])
    data = data.rename(columns={"ID": "sample_id"})
    if data["sample_id"].duplicated().any():
        raise ValueError("data_all.csv contains duplicate sample IDs")
    data["type"] = data["type"].astype(str).str.upper().str.strip()
    unknown = sorted(set(data["type"]) - set(GROUP_ORDER))
    if unknown:
        raise ValueError(f"Unknown type labels in dataset: {unknown}")
    for group, (fake_label, ai_label) in EXPECTED_LABELS.items():
        subset = data[data["type"] == group]
        if not (
            subset["label_fake"].astype(int).eq(fake_label).all()
            and subset["label_ai"].astype(int).eq(ai_label).all()
        ):
            raise ValueError(f"Dataset labels do not match the meaning of {group}")
    return data


def correctness_row(model, group, subset):
    total = len(subset)
    fake_successes = int(subset["pred_fake"].eq(subset["true_fake"]).sum())
    ai_successes = int(subset["pred_ai"].eq(subset["true_ai"]).sum())
    joint_successes = int((
        subset["pred_fake"].eq(subset["true_fake"])
        & subset["pred_ai"].eq(subset["true_ai"])
    ).sum())
    row = {"model": model, "group": group, "n": total}
    for prefix, successes in (
        ("veracity", fake_successes),
        ("provenance", ai_successes),
        ("joint", joint_successes),
    ):
        low, high = wilson_interval(successes, total)
        row[f"{prefix}_correct"] = successes
        row[f"{prefix}_accuracy"] = successes / total
        row[f"{prefix}_ci95_low"] = low
        row[f"{prefix}_ci95_high"] = high
    return row


def analyze(data_path, prediction_specs):
    labels = load_labels(data_path)
    results = []
    sources = {}
    reference_ids = None
    for model, prediction_path in prediction_specs:
        predictions = read_predictions(prediction_path)
        ids = tuple(sorted(predictions["sample_id"].tolist()))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Prediction sample IDs are not aligned for model {model}")
        joined = predictions.merge(labels, on="sample_id", how="left", validate="one_to_one")
        if joined["type"].isna().any():
            missing = joined.loc[joined["type"].isna(), "sample_id"].tolist()
            raise ValueError(f"Prediction IDs absent from dataset for {model}: {missing[:5]}")
        if not (
            joined["true_fake"].eq(joined["label_fake"].astype(int)).all()
            and joined["true_ai"].eq(joined["label_ai"].astype(int)).all()
        ):
            raise ValueError(f"Stored prediction labels disagree with dataset for {model}")
        for group in GROUP_ORDER:
            results.append(correctness_row(model, group, joined[joined["type"] == group]))
        sources[model] = {
            "path": str(Path(prediction_path).resolve()),
            "sha256": file_hash(prediction_path),
            "n": len(predictions),
        }
    return results, sources


def write_outputs(output_dir, results, sources, data_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "subgroup_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "analysis": "HF/MF/HR/MR conditional correctness",
        "confidence_interval": "Wilson score interval, 95%",
        "metric_note": (
            "Within-group labels are constant; conditional accuracy is reported "
            "instead of within-group macro-F1."
        ),
        "data": {
            "path": str(Path(data_path).resolve()),
            "sha256": file_hash(data_path),
        },
        "prediction_sources": sources,
        "results": results,
    }
    json_path = output_dir / "subgroup_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data_all.csv")
    parser.add_argument(
        "--prediction", action="append", type=parse_prediction_spec, required=True,
        help="Repeatable NAME=PATH specification for a test_predictions.json file.",
    )
    parser.add_argument("--output-dir", default="outputs/p0_subgroups")
    args = parser.parse_args(argv)
    names = [name for name, _ in args.prediction]
    if len(names) != len(set(names)):
        raise ValueError("Prediction model names must be unique")
    for _, path in args.prediction:
        if not path.is_file():
            raise FileNotFoundError(path)
    results, sources = analyze(args.data, args.prediction)
    csv_path, json_path = write_outputs(args.output_dir, results, sources, args.data)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
