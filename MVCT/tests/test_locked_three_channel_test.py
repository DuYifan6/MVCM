from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_locked_three_channel_test as locked
import run_strict_replication as strict


def row(label, fake_f1):
    return {
        "variant": label,
        "seed": 42,
        "source_variant": locked.VARIANT_SOURCES[label][1],
        "checkpoint_epoch": 2,
        "fake_threshold": 0.55,
        "fake_accuracy": fake_f1 + 0.01,
        "fake_macro_f1": fake_f1,
        "ai_accuracy": fake_f1 + 0.03,
        "ai_macro_f1": fake_f1 + 0.02,
        "evaluation_dir": f"eval/{label}",
    }


def test_five_predeclared_masks_are_three_channel_and_ablation_consistent():
    assert locked.VARIANTS == (
        "three_channel_full",
        "without_mvcm",
        "without_semantic",
        "without_logical",
        "without_local",
    )
    assert locked.EXPECTED_MASKS["three_channel_full"] == [True, False, True, True]
    assert locked.EXPECTED_MASKS["without_mvcm"] == [False, False, False, False]
    for name in ("without_semantic", "without_logical", "without_local"):
        mask = locked.EXPECTED_MASKS[name]
        assert mask[1] is False
        assert sum(mask) == 2


def test_summary_reports_only_locked_metrics_and_full_deltas():
    rows = [
        row("three_channel_full", .90),
        row("without_mvcm", .86),
        row("without_semantic", .88),
    ]
    report = locked.summarize(rows)
    assert report["completed_evaluations"] == 3
    assert report["fake_threshold"] == .55
    assert report["by_variant"]["without_logical"] is None
    assert report["three_channel_full_minus_ablation"]["without_mvcm"]["fake_macro_f1"] == pytest.approx(.04)
    with pytest.raises(ValueError, match="Duplicate"):
        locked.summarize(rows + [rows[0]])


def test_output_cannot_overlap_any_source(tmp_path):
    sources = [tmp_path / name for name in ("baseline", "channel", "three")]
    for source in sources:
        source.mkdir()
    locked.protect_output(tmp_path / "new", sources)
    for bad in (sources[0], sources[1] / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            locked.protect_output(bad, sources)


def test_check_only_writes_nothing(tmp_path):
    baseline, channel, three, root = (
        tmp_path / "baseline", tmp_path / "channel",
        tmp_path / "three", tmp_path / "locked",
    )
    for path in (baseline, channel, three):
        path.mkdir()
    protocol = strict.seal({"fixture": True})

    def fake_child(arguments, lock, log=None):
        result_path = arguments[arguments.index("--output") + 1]
        strict.atomic_json(result_path, {"protocol": protocol, "sources": {}})

    with patch.object(locked, "launch_child", side_effect=fake_child), \
         patch.object(locked, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(locked, "refuse_unmanaged_jobs"):
        locked.main([
            "--baseline-suite", str(baseline),
            "--channel-suite", str(channel),
            "--three-suite", str(three),
            "--suite-dir", str(root),
            "--check-only",
        ])
    assert not root.exists()


def test_metric_row_reads_reports_without_threshold_search(tmp_path):
    metrics = {
        "fake_report": {"accuracy": .91, "macro avg": {"f1-score": .90}},
        "ai_report": {"accuracy": .96, "macro avg": {"f1-score": .95}},
    }
    source = {"checkpoint_epoch": 2}
    result = locked.metric_row(
        "three_channel_full", source, metrics, tmp_path / "evaluation_1"
    )
    assert result["fake_threshold"] == .55
    assert result["fake_macro_f1"] == .90
    assert "selected_threshold" not in result
