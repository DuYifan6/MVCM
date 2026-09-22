import numpy as np
import pytest
from sklearn.metrics import f1_score

import analyze_locked_predictions as analysis


def prediction_rows(ids, truth_fake, truth_ai, pred_fake, pred_ai):
    return [
        {
            "sample_id": sample_id,
            "true_fake": yf,
            "true_ai": ya,
            "pred_fake": pf,
            "pred_ai": pa,
        }
        for sample_id, yf, ya, pf, pa in zip(
            ids, truth_fake, truth_ai, pred_fake, pred_ai
        )
    ]


def synthetic_arrays():
    ids = list(range(1, 13))
    yf = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    ya = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    full_fake = yf
    full_ai = ya
    weaker_fake = [1 - value if index < 4 else value for index, value in enumerate(yf)]
    weaker_ai = [1 - value if index < 3 else value for index, value in enumerate(ya)]
    rows = {
        "three_channel_full": prediction_rows(ids, yf, ya, full_fake, full_ai),
        "roberta": prediction_rows(ids, yf, ya, weaker_fake, weaker_ai),
        "without_mvcm": prediction_rows(ids, yf, ya, weaker_fake[::-1], weaker_ai[::-1]),
    }
    return analysis.aligned_predictions(rows)


def test_binary_macro_f1_matches_sklearn():
    labels = np.asarray([0, 0, 1, 1, 1])
    predictions = np.asarray([0, 1, 1, 0, 1])
    expected = f1_score(
        labels, predictions, labels=[0, 1], average="macro", zero_division=0
    )
    assert analysis.binary_macro_f1(labels, predictions) == pytest.approx(expected)


def test_holm_adjustment_is_step_down_and_order_preserving():
    adjusted = analysis.holm_adjust([.01, .04, .03])
    assert adjusted == pytest.approx([.03, .06, .06])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        analysis.holm_adjust([1.2])


def test_alignment_sorts_ids_and_rejects_truth_drift():
    rows = {
        "three_channel_full": prediction_rows([2, 1], [1, 0], [0, 1], [1, 0], [0, 1]),
        "roberta": prediction_rows([1, 2], [0, 1], [1, 0], [0, 1], [1, 0]),
        "without_mvcm": prediction_rows([1, 2], [0, 1], [1, 0], [0, 0], [1, 1]),
    }
    arrays = analysis.aligned_predictions(rows)
    assert arrays["sample_id"].tolist() == [1, 2]
    rows["roberta"][0]["true_fake"] = 1
    with pytest.raises(ValueError, match="Ground-truth mismatch"):
        analysis.aligned_predictions(rows)


def test_stratified_bootstrap_preserves_joint_class_counts():
    arrays = synthetic_arrays()
    rng = np.random.default_rng(10)
    index = analysis.stratified_bootstrap_indices(
        arrays["true_fake"], arrays["true_ai"], rng
    )
    original = np.bincount(
        arrays["true_fake"] * 2 + arrays["true_ai"], minlength=4
    )
    sampled = np.bincount(
        arrays["true_fake"][index] * 2 + arrays["true_ai"][index], minlength=4
    )
    assert np.array_equal(original, sampled)


def test_small_analysis_reports_all_eight_rows_and_holm_families():
    report = analysis.analyze(
        synthetic_arrays(), bootstrap_reps=100, permutation_reps=100, seed=7
    )
    assert report["n"] == 12
    assert len(analysis.flat_rows(report)) == 8
    for comparison in analysis.COMPARISONS:
        for task in analysis.TASKS:
            result = report["comparisons"][comparison]["tasks"][task]
            assert result["macro_f1"]["test"]["holm_family_size"] == 4
            assert result["accuracy"]["test"]["holm_family_size"] == 4
            assert result["macro_f1"]["bootstrap_ci"]["repetitions"] == 100


def test_output_cannot_overlap_sources(tmp_path):
    main = tmp_path / "main"
    baseline = tmp_path / "baseline"
    main.mkdir(); baseline.mkdir()
    analysis.protect_output(tmp_path / "analysis", (main, baseline))
    with pytest.raises(ValueError, match="overlap"):
        analysis.protect_output(main / "nested", (main, baseline))
