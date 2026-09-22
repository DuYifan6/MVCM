from pathlib import Path

import pytest

import build_final_evidence_package as package


def metric_row(fake_accuracy, fake_f1, ai_accuracy, ai_f1):
    return {
        "fake_accuracy": fake_accuracy,
        "fake_macro_f1": fake_f1,
        "ai_accuracy": ai_accuracy,
        "ai_macro_f1": ai_f1,
    }


def test_main_table_has_fixed_order_and_full_minus_comparator_deltas():
    main = {
        "by_variant": {
            "three_channel_full": metric_row(.91, .90, .96, .95),
            "without_mvcm": metric_row(.87, .85, .94, .93),
        }
    }
    baseline = {
        "by_variant": {
            name: metric_row(.80 + index / 100, .79 + index / 100, .90, .89)
            for index, name in enumerate(package.BASELINE_ORDER)
        }
    }
    rows = package.build_main_table(main, baseline)
    assert [row["variant"] for row in rows] == list(package.MAIN_ORDER)
    by_name = {row["variant"]: row for row in rows}
    assert by_name["without_mvcm"]["full_minus_fake_macro_f1"] == pytest.approx(.05)
    assert by_name["three_channel_full"]["full_minus_ai_accuracy"] == 0


def test_ablation_table_uses_only_predeclared_three_channel_variants():
    rows_by_name = {
        name: metric_row(.90, .89 - index / 100, .95, .94)
        for index, name in enumerate(package.ABLATION_ORDER)
    }
    rows = package.build_ablation_table({"by_variant": rows_by_name})
    assert [row["variant"] for row in rows] == list(package.ABLATION_ORDER)
    logical = next(row for row in rows if row["variant"] == "without_logical")
    assert logical["full_minus_fake_macro_f1"] == pytest.approx(.02)


def test_architecture_table_renames_four_and_three_channel_models():
    def aggregate(value):
        return {
            "n": 3,
            **{
                metric: {"mean": value, "sd": .01}
                for metric in (*package.METRICS, "val_loss")
            },
        }

    rows = package.build_architecture_table({
        "by_variant": {
            "full": aggregate(.90),
            "without_factual": aggregate(.91),
        }
    })
    assert [row["variant"] for row in rows] == [
        "four_channel_full", "three_channel_full"
    ]
    assert rows[1]["n_seeds"] == 3
    assert rows[1]["fake_macro_f1_mean"] == .91
    assert rows[1]["val_loss_sd"] == .01


def test_output_cannot_overlap_sources(tmp_path):
    sources = [tmp_path / name for name in ("main", "baseline", "statistics")]
    for source in sources:
        source.mkdir()
    package.protect_output(tmp_path / "evidence", sources)
    for bad in (sources[0], sources[1] / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            package.protect_output(bad, sources)


def test_write_csv_replaces_atomically_and_rejects_empty_rows(tmp_path):
    output = tmp_path / "table.csv"
    package.write_csv(output, [{"variant": "full", "score": .9}])
    assert output.read_text(encoding="utf-8").splitlines() == [
        "variant,score", "full,0.9"
    ]
    assert not Path(str(output) + ".tmp").exists()
    with pytest.raises(ValueError, match="empty"):
        package.write_csv(output, [])
