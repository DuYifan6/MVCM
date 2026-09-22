import json

import pandas as pd

import analyze_subgroup_results as subgroup


def test_subgroup_analysis_and_wilson_intervals(tmp_path):
    data = pd.DataFrame([
        {"ID": 1, "type": "HR", "label_fake": 0, "label_ai": 0},
        {"ID": 2, "type": "MR", "label_fake": 0, "label_ai": 1},
        {"ID": 3, "type": "HF", "label_fake": 1, "label_ai": 0},
        {"ID": 4, "type": "MF", "label_fake": 1, "label_ai": 1},
    ])
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    predictions = [
        {"sample_id": 1, "true_fake": 0, "pred_fake": 0, "true_ai": 0, "pred_ai": 0},
        {"sample_id": 2, "true_fake": 0, "pred_fake": 1, "true_ai": 1, "pred_ai": 1},
        {"sample_id": 3, "true_fake": 1, "pred_fake": 1, "true_ai": 0, "pred_ai": 1},
        {"sample_id": 4, "true_fake": 1, "pred_fake": 1, "true_ai": 1, "pred_ai": 1},
    ]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    rows, sources = subgroup.analyze(data_path, [("mvct", prediction_path)])

    assert [row["group"] for row in rows] == list(subgroup.GROUP_ORDER)
    assert rows[0]["joint_accuracy"] == 1.0
    assert rows[1]["veracity_accuracy"] == 0.0
    assert rows[2]["provenance_accuracy"] == 0.0
    assert rows[3]["joint_accuracy"] == 1.0
    assert sources["mvct"]["n"] == 4
    assert 0.0 <= rows[0]["joint_ci95_low"] <= rows[0]["joint_ci95_high"] <= 1.0


def test_rejects_label_drift(tmp_path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame([
        {"ID": 1, "type": "HR", "label_fake": 0, "label_ai": 0},
    ]).to_csv(data_path, index=False)
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps([
        {"sample_id": 1, "true_fake": 1, "pred_fake": 0, "true_ai": 0, "pred_ai": 0},
    ]), encoding="utf-8")

    try:
        subgroup.analyze(data_path, [("mvct", prediction_path)])
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Expected label drift to be rejected")
