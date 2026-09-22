import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_strict_factual_confirmation as factual
import run_strict_replication as strict


def row(seed, variant, fake_f1, ai_f1=None):
    return {
        "seed": seed,
        "variant": variant,
        "checkpoint_epoch": 1,
        "reference_threshold": 0.55,
        "fake_accuracy": fake_f1 + 0.01,
        "fake_macro_f1": fake_f1,
        "ai_accuracy": (ai_f1 if ai_f1 is not None else fake_f1) + 0.01,
        "ai_macro_f1": ai_f1 if ai_f1 is not None else fake_f1,
        "val_loss": 1.0 - fake_f1,
        "selected_threshold": 0.55,
        "selected_validation_fake_macro_f1": fake_f1,
        "run_dir": f"runs/{seed}/{variant}",
    }


def references():
    return {
        "full": {
            "42": row(42, "full", 0.89),
            "43": row(43, "full", 0.90),
            "44": row(44, "full", 0.88),
        },
        "without_factual": {
            "42": row(42, "without_factual", 0.895),
        },
    }


def test_targeted_constants_and_config():
    assert factual.SEEDS == (43, 44)
    assert factual.VARIANT == "without_factual"
    assert factual.WEIGHTS == (0.40, 0.00, 0.20, 0.15)
    base = SimpleNamespace(
        OUTPUT_DIR="old", DEVICE="cuda", LR=2e-5,
        MASK_MISSING_EVIDENCE=False, SAVE_BEST_CHECKPOINT=False,
        SAVE_LAST_CHECKPOINT=True, SAVE_EPOCH_CHECKPOINTS=True,
    )
    original = copy.deepcopy(vars(base))
    config = factual.run_config(base, 43, "new")
    assert vars(base) == original
    assert config.REPLICATION_SEED == 43
    assert config.MVCM_ABLATION_WEIGHTS == [0.40, 0.00, 0.20, 0.15]
    assert config.SAVE_BEST_CHECKPOINT and not config.SAVE_LAST_CHECKPOINT
    with pytest.raises(ValueError, match="Unplanned"):
        factual.run_config(base, 42, "x")


def test_protocol_compatibility_is_fail_closed():
    source = {
        **{key: {"value": key} for key in factual.COMPARABLE_PROTOCOL_FIELDS},
        "entry_sha256": {"core.py": "abc"},
    }
    compatible = copy.deepcopy(source)
    factual.verify_compatible_channel_protocol(source, compatible)
    compatible["runtime"] = {"gpu": "different"}
    with pytest.raises(ValueError, match="runtime"):
        factual.verify_compatible_channel_protocol(source, compatible)
    compatible = copy.deepcopy(source)
    compatible["entry_sha256"]["core.py"] = "different"
    with pytest.raises(ValueError, match="entry_sha256.core.py"):
        factual.verify_compatible_channel_protocol(source, compatible)


def test_summary_combines_seed42_and_two_new_pairs():
    new = [
        row(43, "without_factual", 0.89),
        row(44, "without_factual", 0.885),
    ]
    report = factual.summarize(new, references())
    assert report["completed_new_runs"] == 2
    assert report["completed_pairs"] == 3
    paired = report["paired_full_minus_without_factual"]
    assert paired["n_pairs"] == 3
    assert paired["fake_macro_f1"]["mean"] == pytest.approx(0.0)
    assert paired["fake_macro_f1"]["positive_pairs"] == 1
    assert [item["seed"] for item in paired["per_seed"]] == [42, 43, 44]
    partial = factual.summarize([], references())
    assert partial["completed_pairs"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        factual.summarize(new + [new[0]], references())


def test_output_cannot_overlap_sources(tmp_path):
    sources = (tmp_path / "a", tmp_path / "b", tmp_path / "c")
    for source in sources:
        source.mkdir()
    factual._protect_output(tmp_path / "new", sources)
    for bad in (sources[0], sources[1] / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            factual._protect_output(bad, sources)


def test_check_only_creates_no_suite(tmp_path):
    baseline = tmp_path / "baseline"
    full_suite = tmp_path / "full"
    seed42_suite = tmp_path / "seed42"
    root = tmp_path / "new"
    for path in (baseline, full_suite, seed42_suite):
        path.mkdir()
    protocol = strict.seal({"fixture": True})

    def fake_child(arguments, lock, log=None):
        result_path = arguments[arguments.index("--output") + 1]
        strict.atomic_json(
            result_path,
            {"protocol": protocol, "checkpoint_size": 1,
             "references": references()},
        )

    with patch.object(factual, "launch_child", side_effect=fake_child), \
         patch.object(factual, "default_lock_path", return_value=tmp_path / "lock"), \
         patch.object(factual, "refuse_unmanaged_jobs"), \
         patch.object(factual.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)):
        factual.main([
            "--baseline-suite", str(baseline),
            "--full-suite", str(full_suite),
            "--seed42-suite", str(seed42_suite),
            "--suite-dir", str(root),
            "--check-only",
        ])
    assert not root.exists()

