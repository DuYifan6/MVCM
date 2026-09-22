import json

import run_p0_analysis


def test_resolves_frozen_prediction_paths(tmp_path):
    main = tmp_path / "main"
    baseline = tmp_path / "baseline"
    full_output = tmp_path / "full_output"
    without_output = tmp_path / "without_output"
    roberta_output = tmp_path / "roberta_output"
    for path in (full_output, without_output, roberta_output):
        path.mkdir(parents=True)
        (path / "test_predictions.json").write_text("[]", encoding="utf-8")
    (main / "three_channel_full").mkdir(parents=True)
    (main / "without_mvcm").mkdir(parents=True)
    (baseline / "locked_test" / "roberta").mkdir(parents=True)
    (main / "three_channel_full" / "completed.json").write_text(
        json.dumps({"evaluation_dir": str(full_output)}), encoding="utf-8"
    )
    (main / "without_mvcm" / "completed.json").write_text(
        json.dumps({"evaluation_dir": str(without_output)}), encoding="utf-8"
    )
    (baseline / "locked_test" / "roberta" / "completed.json").write_text(
        json.dumps({"output_dir": str(roberta_output)}), encoding="utf-8"
    )

    paths = run_p0_analysis.frozen_prediction_paths(main, baseline)

    assert paths["mvct"] == full_output / "test_predictions.json"
    assert paths["without_mvcm"] == without_output / "test_predictions.json"
    assert paths["roberta"] == roberta_output / "test_predictions.json"
