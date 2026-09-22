"""Deterministically select illustrative cases from frozen locked-test predictions.

The selector never changes a model, prediction, threshold, or reported metric.
It chooses a median, non-extreme case under a predeclared rule and records the
complete candidate count and fallback tier so the example is auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prediction_path_from_completion(completion_path, directory_key):
    row = read_json(completion_path)
    directory = Path(row[directory_key])
    path = directory / "test_predictions.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file referenced by {completion_path}: {path}")
    return path


def main_prediction_path(suite, variant):
    completion = Path(suite) / variant / "completed.json"
    if completion.is_file():
        return prediction_path_from_completion(completion, "evaluation_dir")
    table = Path(suite) / "locked_test_results.csv"
    if table.is_file():
        with table.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["variant"] == variant:
                    path = Path(row["evaluation_dir"]) / "test_predictions.json"
                    if path.is_file():
                        return path
    raise FileNotFoundError(f"Could not resolve {variant} predictions from {suite}")


def baseline_prediction_path(suite, variant="roberta"):
    completion = Path(suite) / "locked_test" / variant / "completed.json"
    if completion.is_file():
        return prediction_path_from_completion(completion, "output_dir")
    table = Path(suite) / "locked_test_results.csv"
    if table.is_file():
        with table.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["variant"] == variant:
                    path = Path(row["output_dir"]) / "test_predictions.json"
                    if path.is_file():
                        return path
    raise FileNotFoundError(f"Could not resolve {variant} predictions from {suite}")


def prediction_table(path):
    rows = read_json(path)
    table = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id in table:
            raise ValueError(f"Duplicate sample_id {sample_id} in {path}")
        table[sample_id] = row
    return table


def load_text_rows(path):
    if path is None:
        return {}
    table = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(float(row["ID"]))
            if sample_id not in table:
                table[sample_id] = row
    return table


def true_probability(row, task):
    truth = int(row[f"true_{task}"])
    positive = float(row[f"prob_{task}"])
    return positive if truth == 1 else 1.0 - positive


def group_name(true_fake, true_ai):
    return {0: "H", 1: "M"}[int(true_ai)] + {0: "R", 1: "F"}[int(true_fake)]


def normalized_distance(value, center, values):
    deviations = [abs(item - center) for item in values]
    scale = median(deviations) if deviations else 0.0
    if not math.isfinite(scale) or scale < 1e-12:
        scale = max(max(values, default=center) - min(values, default=center), 1.0)
    return abs(value - center) / scale


def text_metadata(row):
    if not row:
        return {"word_count": None, "title": None, "description": None, "body_excerpt": None}
    title = str(row.get("title", "")).strip()
    body = str(row.get("text", "")).strip()
    description = str(row.get("desc", "")).strip()
    return {
        "word_count": len((title + " " + body + " " + description).split()),
        "title": title,
        "description": description,
        "body_excerpt": body[:500],
    }


def aligned_rows(mvct, without, roberta, texts):
    ids = set(mvct)
    if ids != set(without) or ids != set(roberta):
        raise ValueError("Prediction files do not contain identical sample IDs")
    rows = []
    for sample_id in sorted(ids):
        truth = (int(mvct[sample_id]["true_fake"]), int(mvct[sample_id]["true_ai"]))
        for name, table in (("without_mvcm", without), ("roberta", roberta)):
            other = table[sample_id]
            if truth != (int(other["true_fake"]), int(other["true_ai"])):
                raise ValueError(f"Ground-truth mismatch for sample {sample_id} in {name}")
        metadata = text_metadata(texts.get(sample_id))
        rows.append(
            {
                "sample_id": sample_id,
                "true_fake": truth[0],
                "true_ai": truth[1],
                "group": group_name(*truth),
                "mvct": mvct[sample_id],
                "without_mvcm": without[sample_id],
                "roberta": roberta[sample_id],
                **metadata,
            }
        )
    return rows


def is_correct(row, model, task):
    return int(row[model][f"pred_{task}"]) == int(row[f"true_{task}"])


def gain_score(row):
    full = true_probability(row["mvct"], "fake")
    controls = [
        true_probability(row["without_mvcm"], "fake"),
        true_probability(row["roberta"], "fake"),
    ]
    return full - max(controls)


def choose_median_case(candidates, score_function):
    if not candidates:
        return None, []
    gains = [score_function(row) for row in candidates]
    gain_center = median(gains)
    word_values = [row["word_count"] for row in candidates if row["word_count"] is not None]
    word_center = median(word_values) if word_values else None
    log_words = [math.log1p(value) for value in word_values]
    log_center = math.log1p(word_center) if word_center is not None else None
    scored = []
    for row, gain in zip(candidates, gains):
        score = normalized_distance(gain, gain_center, gains)
        if row["word_count"] is not None and log_words:
            score += 0.20 * normalized_distance(math.log1p(row["word_count"]), log_center, log_words)
        scored.append((score, row["sample_id"], gain, row))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][3], scored


def compact_case(row, gain=None):
    if row is None:
        return None
    result = {
        "sample_id": row["sample_id"],
        "group": row["group"],
        "true_fake": row["true_fake"],
        "true_ai": row["true_ai"],
        "word_count": row["word_count"],
        "title": row["title"],
        "description": row["description"],
        "body_excerpt": row["body_excerpt"],
        "predictions": {
            name: {
                key: value
                for key, value in row[name].items()
                if key in {"pred_fake", "prob_fake", "pred_ai", "prob_ai"}
            }
            for name in ("mvct", "without_mvcm", "roberta")
        },
    }
    if gain is not None:
        result["mvct_true_fake_probability_advantage_over_best_control"] = gain
    return result


def select_hf_gain(rows):
    tiers = [
        (
            "HF; MVCT correct on both tasks; both controls miss veracity; all models correct provenance",
            lambda row: row["group"] == "HF"
            and is_correct(row, "mvct", "fake")
            and is_correct(row, "mvct", "ai")
            and not is_correct(row, "without_mvcm", "fake")
            and not is_correct(row, "roberta", "fake")
            and is_correct(row, "without_mvcm", "ai")
            and is_correct(row, "roberta", "ai"),
        ),
        (
            "HF; MVCT correct on both tasks; w/o MVCM misses veracity",
            lambda row: row["group"] == "HF"
            and is_correct(row, "mvct", "fake")
            and is_correct(row, "mvct", "ai")
            and not is_correct(row, "without_mvcm", "fake"),
        ),
        (
            "HF; MVCT correct on both tasks; at least one control misses veracity",
            lambda row: row["group"] == "HF"
            and is_correct(row, "mvct", "fake")
            and is_correct(row, "mvct", "ai")
            and (not is_correct(row, "without_mvcm", "fake") or not is_correct(row, "roberta", "fake")),
        ),
    ]
    for index, (description, predicate) in enumerate(tiers, start=1):
        candidates = [row for row in rows if predicate(row)]
        if candidates:
            selected, ranked = choose_median_case(candidates, gain_score)
            return {
                "selection_tier": index,
                "criterion": description,
                "candidate_count": len(candidates),
                "selected": compact_case(selected, gain_score(selected)),
                "nearest_alternatives": [
                    compact_case(item[3], item[2]) for item in ranked[1:6]
                ],
            }
    return {"selection_tier": None, "criterion": None, "candidate_count": 0, "selected": None}


def select_hr_boundary(rows):
    candidates = [
        row
        for row in rows
        if row["group"] == "HR"
        and not is_correct(row, "mvct", "fake")
        and is_correct(row, "roberta", "fake")
        and is_correct(row, "mvct", "ai")
    ]
    selected, ranked = choose_median_case(
        candidates,
        lambda row: float(row["mvct"]["prob_fake"]),
    )
    return {
        "criterion": "HR; MVCT false positive on veracity; RoBERTa correct; MVCT correct provenance",
        "candidate_count": len(candidates),
        "selected": compact_case(selected),
        "nearest_alternatives": [compact_case(item[3]) for item in ranked[1:6]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-suite", required=True, help="Locked three-channel test suite")
    parser.add_argument("--baseline-suite", required=True, help="Strict baseline suite containing RoBERTa")
    parser.add_argument("--data-csv", help="Original article CSV; used only for text and length metadata")
    parser.add_argument("--output", required=True, help="Selection report JSON")
    args = parser.parse_args(argv)

    sources = {
        "mvct": main_prediction_path(args.main_suite, "three_channel_full"),
        "without_mvcm": main_prediction_path(args.main_suite, "without_mvcm"),
        "roberta": baseline_prediction_path(args.baseline_suite, "roberta"),
    }
    tables = {name: prediction_table(path) for name, path in sources.items()}
    texts = load_text_rows(args.data_csv)
    rows = aligned_rows(tables["mvct"], tables["without_mvcm"], tables["roberta"], texts)
    report = {
        "purpose": "Post-hoc selection of auditable illustrative cases; not model or threshold selection",
        "test_samples": len(rows),
        "source_prediction_files": {name: str(path.resolve()) for name, path in sources.items()},
        "primary_recommendation": select_hf_gain(rows),
        "balancing_failure_case": select_hr_boundary(rows),
        "selection_policy": (
            "Use the strictest nonempty tier. Within that tier choose the candidate closest to the median "
            "MVCT true-label probability advantage, with a small deterministic penalty for atypical article length. "
            "No consistency-matrix value, attention pattern, or visual attractiveness is used for selection."
        ),
    }
    write_json(args.output, report)
    primary = report["primary_recommendation"]
    selected = primary.get("selected")
    print(f"Aligned frozen predictions: {len(rows)}")
    print(f"HF selection tier: {primary['selection_tier']}; candidates: {primary['candidate_count']}")
    print(f"Selected HF sample_id: {None if selected is None else selected['sample_id']}")
    boundary = report["balancing_failure_case"]
    print(f"HR boundary candidates: {boundary['candidate_count']}")
    print(f"Selected HR boundary sample_id: {None if boundary['selected'] is None else boundary['selected']['sample_id']}")
    print(f"Wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
