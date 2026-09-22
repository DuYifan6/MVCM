import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_strict_channel_ablation as channel
import run_strict_replication as strict


def metric_row(variant, score):
    return {
        "seed": 42,
        "variant": variant,
        "checkpoint_epoch": 1,
        "reference_threshold": 0.55,
        "fake_accuracy": score,
        "fake_macro_f1": score - 0.01,
        "ai_accuracy": score + 0.02,
        "ai_macro_f1": score + 0.01,
        "val_loss": 1.0 - score,
        "selected_threshold": 0.55,
        "selected_validation_fake_macro_f1": score - 0.01,
        "run_dir": f"runs/{variant}",
    }


def test_six_prespecified_variants_and_masks():
    assert channel.SEED == 42
    assert channel.VARIANTS == (
        "full", "without_mvcm", "without_semantic", "without_factual",
        "without_logical", "without_local",
    )
    assert channel.VARIANT_WEIGHTS["without_semantic"] == (0.0, 0.25, 0.20, 0.15)
    assert channel.VARIANT_WEIGHTS["without_factual"] == (0.40, 0.0, 0.20, 0.15)
    assert channel.VARIANT_WEIGHTS["without_logical"] == (0.40, 0.25, 0.0, 0.15)
    assert channel.VARIANT_WEIGHTS["without_local"] == (0.40, 0.25, 0.20, 0.0)


def test_run_config_preserves_frozen_training_fields():
    reference = SimpleNamespace(
        OUTPUT_DIR="old", DEVICE="cuda", FAKE_THRESHOLD=0.55,
        LR=2e-5, EPOCHS=4, MASK_MISSING_EVIDENCE=False,
        SAVE_BEST_CHECKPOINT=False, SAVE_LAST_CHECKPOINT=True,
        SAVE_EPOCH_CHECKPOINTS=True,
    )
    original = copy.deepcopy(vars(reference))
    config = channel.run_config(reference, "without_logical", "new/run")
    assert vars(reference) == original
    assert config.REPLICATION_SEED == 42
    assert config.EXPERIMENT_NAME == "without_logical"
    assert config.MVCM_ABLATION_WEIGHTS == [0.40, 0.25, 0.0, 0.15]
    assert config.LR == reference.LR and config.EPOCHS == reference.EPOCHS
    assert not config.MASK_MISSING_EVIDENCE
    assert config.SAVE_BEST_CHECKPOINT
    assert not config.SAVE_LAST_CHECKPOINT and not config.SAVE_EPOCH_CHECKPOINTS
    with pytest.raises(ValueError, match="Unplanned"):
        channel.run_config(reference, "best_seed_only", "x")


def test_protocol_adds_ablation_contract_without_mutating_source(tmp_path):
    source = strict.seal({
        "purpose": "source",
        "seeds": [42],
        "variants": ["full", "without_mvcm"],
        "entry_sha256": {"run_strict_replication.py": "abc"},
        "runtime": {"gpu": "fixture"},
    })
    untouched = copy.deepcopy(source)
    reference = SimpleNamespace(FAKE_THRESHOLD=0.55)
    with patch.object(strict, "prepare", return_value=(source, reference, [1], [2], 123)):
        protocol, config, train, val, size = channel.prepare(tmp_path)
    assert source == untouched
    assert protocol["variants"] == list(channel.VARIANTS)
    assert protocol["seeds"] == [42]
    assert protocol["source_strict_preflight_protocol_id"] == source["protocol_id"]
    assert "run_strict_channel_ablation.py" in protocol["entry_sha256"]
    assert protocol["test_evaluation"] is False
    assert (config, train, val, size) == (reference, [1], [2], 123)
    assert strict.seal({k: v for k, v in protocol.items() if k != "protocol_id"}) == protocol


def test_summary_reports_full_minus_each_ablation():
    rows = [metric_row("full", 0.90), metric_row("without_mvcm", 0.86),
            metric_row("without_semantic", 0.88)]
    report = channel.summarize(rows)
    assert report["planned_runs"] == 6 and report["completed_runs"] == 3
    assert report["by_variant"]["without_factual"] is None
    assert report["full_minus_ablation"]["without_mvcm"]["fake_accuracy"] == pytest.approx(0.04)
    assert report["full_minus_ablation"]["without_semantic"]["val_loss"] == pytest.approx(-0.02)
    with pytest.raises(ValueError, match="Duplicate"):
        channel.summarize(rows + [rows[0]])
    with pytest.raises(ValueError, match="Unplanned"):
        channel.summarize([metric_row("other", 0.5)])


def test_check_only_writes_no_suite(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "baseline"
    baseline.mkdir()
    protocol = strict.seal({"fixture": True})

    def fake_child(arguments, lock, log=None):
        assert arguments[arguments.index("--worker") + 1] == "preflight"
        result_path = arguments[arguments.index("--output") + 1]
        strict.atomic_json(result_path, {"protocol": protocol, "checkpoint_size": 1})

    with patch.object(channel, "launch_child", side_effect=fake_child), \
         patch.object(channel, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(channel, "refuse_unmanaged_jobs"), \
         patch.object(channel.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        channel.main([
            "--check-only", "--baseline-suite", str(baseline),
            "--suite-dir", str(root), "--max-new-runs", "6",
        ])
    assert not root.exists()


def test_supervisor_runs_all_six_once(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "baseline"
    baseline.mkdir()
    protocol = strict.seal({"fixture": True})
    completed = {}

    def fake_completed(slot, current_protocol, variant, record=None):
        if record is not None:
            completed[variant] = record
            return record
        return completed.get(variant)

    def fake_child(arguments, lock, log=None):
        def value(name):
            return arguments[arguments.index(name) + 1]
        if value("--worker") == "preflight":
            strict.atomic_json(value("--output"), {"protocol": protocol, "checkpoint_size": 1})
            return
        variant = value("--variant")
        output = Path(value("--output"))
        strict.atomic_json(output / "worker_result.json", metric_row(variant, 0.9))

    with patch.object(channel, "launch_child", side_effect=fake_child), \
         patch.object(channel, "completed_record", side_effect=fake_completed), \
         patch.object(channel, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(channel, "refuse_unmanaged_jobs"), \
         patch.object(channel.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        channel.main([
            "--baseline-suite", str(baseline), "--suite-dir", str(root),
            "--max-new-runs", "6",
        ])
    assert set(completed) == set(channel.VARIANTS)
    assert strict.read_json(root / "ablation_summary.json")["completed_runs"] == 6
    assert len(list(root.glob("seed_42/*/attempt_*"))) == 6

