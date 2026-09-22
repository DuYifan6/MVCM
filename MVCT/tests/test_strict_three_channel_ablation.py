import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_strict_replication as strict
import run_strict_three_channel_ablation as three


def row(variant, fake_f1):
    return {
        "seed": 42,
        "variant": variant,
        "checkpoint_epoch": 1,
        "reference_threshold": 0.55,
        "fake_accuracy": fake_f1 + 0.01,
        "fake_macro_f1": fake_f1,
        "ai_accuracy": fake_f1 + 0.03,
        "ai_macro_f1": fake_f1 + 0.02,
        "val_loss": 1.0 - fake_f1,
        "selected_threshold": 0.55,
        "selected_validation_fake_macro_f1": fake_f1,
        "run_dir": f"runs/{variant}",
    }


def references():
    return {
        "three_channel_full": row("without_factual", 0.90),
        "without_mvcm": row("without_mvcm", 0.86),
    }


def test_three_channel_variants_keep_factual_disabled():
    assert three.VARIANTS == (
        "three_without_semantic",
        "three_without_logical",
        "three_without_local",
    )
    for weights in three.VARIANT_WEIGHTS.values():
        assert weights[1] == 0.0
        assert sum(weight > 0 for weight in weights) == 2
    assert three.VARIANT_WEIGHTS["three_without_semantic"] == (0, 0, .2, .15)
    assert three.VARIANT_WEIGHTS["three_without_logical"] == (.4, 0, 0, .15)
    assert three.VARIANT_WEIGHTS["three_without_local"] == (.4, 0, .2, 0)


def test_run_config_records_active_channels_without_mutation():
    base = SimpleNamespace(
        OUTPUT_DIR="old", DEVICE="cuda", LR=2e-5,
        MASK_MISSING_EVIDENCE=False, SAVE_BEST_CHECKPOINT=False,
        SAVE_LAST_CHECKPOINT=True, SAVE_EPOCH_CHECKPOINTS=True,
    )
    original = copy.deepcopy(vars(base))
    config = three.run_config(base, "three_without_logical", "new")
    assert vars(base) == original
    assert config.MVCM_ABLATION_WEIGHTS == [0.4, 0.0, 0.0, 0.15]
    assert config.ACTIVE_MVCM_CHANNELS == ["semantic", "local"]
    assert config.SAVE_BEST_CHECKPOINT and not config.SAVE_LAST_CHECKPOINT
    with pytest.raises(ValueError, match="Unplanned"):
        three.run_config(base, "without_factual", "x")


def test_source_protocol_check_is_fail_closed():
    source = {
        **{key: {"value": key} for key in three.COMPARABLE_PROTOCOL_FIELDS},
        "entry_sha256": {"core.py": "abc"},
    }
    saved = copy.deepcopy(source)
    three.verify_source_protocol(source, saved)
    saved["inventory"] = {"changed": True}
    with pytest.raises(ValueError, match="inventory"):
        three.verify_source_protocol(source, saved)
    saved = copy.deepcopy(source)
    saved["entry_sha256"]["core.py"] = "changed"
    with pytest.raises(ValueError, match="entry_sha256.core.py"):
        three.verify_source_protocol(source, saved)


def test_summary_reuses_reference_and_reports_three_deltas():
    rows = [
        row("three_without_semantic", 0.88),
        row("three_without_logical", 0.89),
        row("three_without_local", 0.84),
    ]
    report = three.summarize(rows, references())
    assert report["planned_new_runs"] == 3
    assert report["completed_new_runs"] == 3
    assert set(report["by_variant"]) == {
        "three_channel_full", "without_mvcm", *three.VARIANTS,
    }
    assert report["three_channel_full_minus_ablation"]["without_mvcm"]["fake_macro_f1"] == pytest.approx(.04)
    assert report["three_channel_full_minus_ablation"]["three_without_local"]["fake_macro_f1"] == pytest.approx(.06)
    partial = three.summarize([], references())
    assert partial["completed_new_runs"] == 0
    assert set(partial["by_variant"]) == {"three_channel_full", "without_mvcm"}
    with pytest.raises(ValueError, match="Duplicate"):
        three.summarize(rows + [rows[0]], references())


def test_output_cannot_overlap_sources(tmp_path):
    baseline, source = tmp_path / "baseline", tmp_path / "source"
    baseline.mkdir()
    source.mkdir()
    three.protect_output(tmp_path / "new", (baseline, source))
    for bad in (baseline, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            three.protect_output(bad, (baseline, source))


def test_check_only_creates_no_suite(tmp_path):
    baseline, source, root = (
        tmp_path / "baseline", tmp_path / "source", tmp_path / "new"
    )
    baseline.mkdir()
    source.mkdir()
    protocol = strict.seal({"fixture": True})

    def fake_child(arguments, lock, log=None):
        result_path = arguments[arguments.index("--output") + 1]
        strict.atomic_json(
            result_path,
            {"protocol": protocol, "checkpoint_size": 1,
             "references": references()},
        )

    with patch.object(three, "launch_child", side_effect=fake_child), \
         patch.object(three, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(three, "refuse_unmanaged_jobs"), \
         patch.object(three.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        three.main([
            "--baseline-suite", str(baseline),
            "--source-suite", str(source),
            "--suite-dir", str(root),
            "--check-only",
        ])
    assert not root.exists()

