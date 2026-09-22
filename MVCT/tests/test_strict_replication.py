import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_strict_replication as strict


def score_row(variant, seed, score):
    return {"seed": seed, "variant": variant,
            **{m: score if m != "val_loss" else 1 - score for m in strict.MEASURES}}


def test_protocol_integrity_and_runtime_drift(tmp_path):
    path = tmp_path / "protocol.json"
    original = strict.seal({"runtime": {"gpu": "A"}, "config": {"threshold": .55}})
    strict.atomic_json(path, original)
    assert strict.verified_protocol(path) == original
    changed = copy.deepcopy(original)
    changed["runtime"]["gpu"] = "B"
    with pytest.raises(ValueError, match="runtime"):
        strict.same_protocol(original, changed)
    strict.atomic_json(path, changed)
    with pytest.raises(ValueError, match="integrity"):
        strict.verified_protocol(path)


def test_strict_environment_overrides_without_mutating_parent():
    with patch.dict(os.environ, {"OMP_NUM_THREADS": "garbled", "PYTHONHASHSEED": "17"}):
        env = strict.strict_environment()
        assert all(env[k] == v for k, v in strict.STRICT_ENV.items())
        assert os.environ["OMP_NUM_THREADS"] == "garbled"
        assert os.environ["PYTHONHASHSEED"] == "17"


def test_launcher_module_does_not_import_torch():
    subprocess.run([sys.executable, "-c", "import sys; import run_strict_replication; assert 'torch' not in sys.modules"],
                   cwd=str(strict.SOURCE_DIR), check=True)


def test_only_prespecified_run_config_changes():
    reference = SimpleNamespace(OUTPUT_DIR="old", DEVICE="cuda", FAKE_THRESHOLD=.55,
                                LR=2e-5, EPOCHS=10, FAKE_CLASS_WEIGHT=None, MASK_MISSING_EVIDENCE=False)
    original = vars(reference).copy()
    strict.validate_reference(reference)
    a = vars(strict.run_config(reference, 42, "full", "new/full"))
    b = vars(strict.run_config(reference, 42, "without_mvcm", "new/without"))
    assert vars(reference) == original
    assert {k for k in a if a[k] != b[k]} == {"OUTPUT_DIR", "EXPERIMENT_NAME"}
    for key in ("LR", "EPOCHS", "FAKE_THRESHOLD", "FAKE_CLASS_WEIGHT"):
        assert a[key] == original[key]
    assert not a["SAVE_LAST_CHECKPOINT"] and not a["SAVE_EPOCH_CHECKPOINTS"]
    with pytest.raises(ValueError):
        strict.run_config(reference, 45, "full", "x")
    with pytest.raises(ValueError):
        strict.run_config(reference, 42, "missing_evidence", "x")
    reference.FAKE_THRESHOLD = .51
    with pytest.raises(ValueError):
        strict.validate_reference(reference)


def test_original_loss_selector_tolerance_and_invalid_values():
    # A sub-tolerance decrease must NOT choose a newer epoch.
    assert strict.selected_loss_epoch([.3, .2, .1999995, .25]) == (2, .2)
    assert strict.selected_loss_epoch([.3, .2, .19, .25]) == (3, .19)
    for values in ([], [.2, float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            strict.selected_loss_epoch(values)


def test_six_run_summary_pairs_sd_and_duplicates():
    rows = [score_row("full", 42, .81), score_row("without_mvcm", 42, .8),
            score_row("full", 43, .86), score_row("without_mvcm", 43, .9),
            score_row("full", 44, .88)]
    report = strict.summarize(rows)
    assert report["planned_runs"] == 6 and report["completed_runs"] == 5
    assert set(report["by_variant"]) == {"full", "without_mvcm"}
    pair = report["paired_full_minus_without_mvcm"]
    assert pair["n_pairs"] == 2
    assert pair["fake_macro_f1"]["mean"] == pytest.approx(-.015)
    assert pair["fake_macro_f1"]["positive_pairs"] == 1
    assert report["by_variant"]["without_mvcm"]["fake_macro_f1"]["sd"] == pytest.approx(.1 / 2**.5)
    assert strict.summarize([])["by_variant"]["full"]["fake_macro_f1"]["mean"] is None
    with pytest.raises(ValueError):
        strict.summarize(rows + [rows[0]])


def test_protect_old_directories_and_untracked_outputs(tmp_path):
    baseline = tmp_path / "old"
    baseline.mkdir()
    protocol = strict.seal({"x": 1})
    for root in (baseline, baseline / "new", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            strict.check_root(root, baseline, protocol)
    new = tmp_path / "new"
    strict.check_root(new, baseline, protocol)
    new.mkdir()
    (new / "old_result.txt").write_text("preserve")
    with pytest.raises(ValueError, match="Nonempty"):
        strict.check_root(new, baseline, protocol)


def make_completion(tmp_path, seed=42, variant="full", output=None):
    slot = tmp_path / f"seed_{seed}" / variant
    output = output or slot / "attempt_fixture"
    diagnostic = output / "validation_diagnostic_fixture"
    diagnostic.mkdir(parents=True)
    report = {"checkpoint_epoch": 2, "reference_threshold": .55, "combined_task_loss": .2,
              "loss_reproduced": True, "selected_on_validation": {"threshold": .51, "macro_f1": .82},
              "at_reference_threshold": {"fake_report": {"accuracy": .8, "macro avg": {"f1-score": .79}},
                                         "ai_report": {"accuracy": .9, "macro avg": {"f1-score": .89}}}}
    strict.atomic_json(diagnostic / "validation_metrics.json", report)
    strict.atomic_json(diagnostic / "validation_predictions.json", [])
    strict.atomic_json(output / "history.json", {"val_loss": [.3, .2, .25]})
    strict.atomic_json(output / "config.json", {})
    strict.atomic_json(output / "run_identity.json", {})
    (output / "best_checkpoint.pt").write_bytes(b"fixture-checkpoint-not-a-real-model")
    names = [*strict.ARTIFACT_NAMES,
             str((diagnostic / "validation_metrics.json").relative_to(output)),
             str((diagnostic / "validation_predictions.json").relative_to(output))]
    row = {"protocol_id": "p", "seed": seed, "variant": variant, "run_dir": str(output),
           "diagnostic_dir": str(diagnostic), "checkpoint_epoch": 2, "reference_threshold": .55,
           "val_loss": .2, "loss_reproduced": True, "fake_accuracy": .8, "fake_macro_f1": .79,
           "ai_accuracy": .9, "ai_macro_f1": .89, "selected_threshold": .51,
           "selected_validation_fake_macro_f1": .82,
           "artifact_sha256": {n: strict.file_hash(output / n) for n in names}}
    return slot, output, row


def test_completion_verification_before_commit_and_resume(tmp_path):
    slot, output, row = make_completion(tmp_path)
    protocol = {"protocol_id": "p"}
    assert strict.completed_record(slot, protocol, "full", 42) is None
    assert strict.completed_record(slot, protocol, "full", 42, record=row) == row
    assert not (slot / "completed.json").exists()
    strict.atomic_json(slot / "completed.json", row)
    assert strict.completed_record(slot, protocol, "full", 42) == row
    with pytest.raises(ValueError):
        strict.completed_record(slot, {"protocol_id": "other"}, "full", 42)
    forged = {**row, "fake_macro_f1": .99}
    with pytest.raises(ValueError, match="metrics"):
        strict.completed_record(slot, protocol, "full", 42, record=forged)
    with pytest.raises(ValueError, match="outside"):
        strict.completed_record(slot, protocol, "full", 42, record={**row, "run_dir": str(tmp_path)})
    (output / "best_checkpoint.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        strict.completed_record(slot, protocol, "full", 42)


def test_check_only_creates_no_suite_and_never_starts_training(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "old"
    baseline.mkdir()
    calls = []
    def fake_child(arguments, lock, log=None):
        calls.append(arguments)
        assert arguments[arguments.index("--worker") + 1] == "preflight"
        strict.atomic_json(arguments[arguments.index("--output") + 1],
                           {"protocol": strict.seal({"fixture": True}), "checkpoint_size": 1})
    with patch.object(strict, "launch_child", side_effect=fake_child), \
         patch.object(strict, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(strict, "refuse_unmanaged_jobs"), \
         patch.object(strict.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        strict.main(["--check-only", "--baseline-suite", str(baseline), "--suite-dir", str(root)])
    assert len(calls) == 1 and not root.exists()


def test_worker_failure_preserves_attempt_without_completion(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "old"
    baseline.mkdir()
    def fake_child(arguments, lock, log=None):
        if arguments[arguments.index("--worker") + 1] == "preflight":
            strict.atomic_json(arguments[arguments.index("--output") + 1],
                               {"protocol": strict.seal({"fixture": True}), "checkpoint_size": 1})
        else:
            raise RuntimeError("simulated training failure")
    with patch.object(strict, "launch_child", side_effect=fake_child), \
         patch.object(strict, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(strict, "refuse_unmanaged_jobs"), \
         patch.object(strict.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        with pytest.raises(RuntimeError, match="simulated"):
            strict.main(["--baseline-suite", str(baseline), "--suite-dir", str(root)])
    slot = root / "seed_42" / "full"
    assert len(list(slot.glob("attempt_*"))) == 1
    assert not (slot / "completed.json").exists()
    assert strict.read_json(root / "replication_summary.json")["completed_runs"] == 0


def test_supervisor_six_runs_resume_and_no_duplicate_training(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "old"
    baseline.mkdir()
    protocol = strict.seal({"fixture": True})
    launched = []
    def fake_child(arguments, lock, log=None):
        def arg(name):
            return arguments[arguments.index(name) + 1]
        if arg("--worker") == "preflight":
            strict.atomic_json(arg("--output"), {"protocol": protocol, "checkpoint_size": 1})
            return
        seed, variant, output = int(arg("--seed")), arg("--variant"), Path(arg("--output"))
        launched.append((seed, variant))
        _, _, row = make_completion(root, seed, variant, output)
        row["protocol_id"] = protocol["protocol_id"]
        strict.atomic_json(output / "worker_result.json", row)
    with patch.object(strict, "launch_child", side_effect=fake_child), \
         patch.object(strict, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(strict, "refuse_unmanaged_jobs"), \
         patch.object(strict.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        args = ["--baseline-suite", str(baseline), "--suite-dir", str(root)]
        for count in (2, 4, 6, 6):
            strict.main(args)
            assert len(launched) == count
            assert strict.read_json(root / "replication_summary.json")["completed_runs"] == count
    assert launched == [(s, v) for s in strict.SEEDS for v in strict.VARIANTS]
    assert len(list(root.glob("seed_*/*/attempt_*"))) == 6


def test_preflight_disk_failure_does_not_create_suite(tmp_path):
    root, baseline = tmp_path / "new", tmp_path / "old"
    baseline.mkdir()
    def fake_child(arguments, lock, log=None):
        strict.atomic_json(arguments[arguments.index("--output") + 1],
                           {"protocol": strict.seal({"fixture": True}), "checkpoint_size": 2**30})
    with patch.object(strict, "launch_child", side_effect=fake_child), \
         patch.object(strict, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(strict, "refuse_unmanaged_jobs"), \
         patch.object(strict.shutil, "disk_usage", return_value=SimpleNamespace(free=1024)):
        with pytest.raises(RuntimeError, match="GiB"):
            strict.main(["--baseline-suite", str(baseline), "--suite-dir", str(root)])
    assert not root.exists()
