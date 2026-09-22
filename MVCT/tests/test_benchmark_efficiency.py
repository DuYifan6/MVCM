import benchmark_efficiency as efficiency


def test_percentile_and_parameter_count():
    import torch

    model = torch.nn.Linear(3, 2)
    total, trainable = efficiency.parameter_counts(model)
    assert total == 8
    assert trainable == 8
    assert efficiency.percentile([1.0, 2.0, 3.0], 95) > 2.0


def test_cache_profile_marks_reused_manifest_as_not_cold(tmp_path):
    import json
    import numpy as np

    np.save(tmp_path / "mvc_sample.npy", np.zeros((3, 3, 4), dtype=np.float32))
    (tmp_path / "cache_manifest.json").write_text(json.dumps({
        "unique_samples": 1,
        "built": 0,
        "reused": 1,
        "failed": 0,
        "elapsed_seconds": 0.1,
    }), encoding="utf-8")
    profile = efficiency.cache_profile(tmp_path)
    assert profile["cache_files"] == 1
    assert profile["manifest_cold_build_valid"] is False


def test_omp_threads_must_be_positive_integer(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert efficiency.validated_omp_threads() == 4
    monkeypatch.setenv("OMP_NUM_THREADS", "not-an-integer")
    try:
        efficiency.validated_omp_threads()
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("Expected invalid OMP_NUM_THREADS to be rejected")
