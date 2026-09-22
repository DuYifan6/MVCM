import copy

import pytest

from aggregate_crosshost_confirmation import comparable_protocol, runtime_for_comparison, sealed


def protocol(uuid="GPU-a", torch_version="2.1.1+cu121"):
    return {
        "baseline_protocol_id": "baseline", "reference_config": {"LR": 2e-5},
        "inventory": {"dataset_sha": "d", "cache": "c"}, "selection": "loss",
        "primary_metric": "f1", "threshold_scan": "descriptive", "test_evaluation": False,
        "initialization": "fresh", "runtime": {
            "packages": {"torch": torch_version}, "cuda": "12.1", "cudnn": 8900,
            "gpu": "RTX 4080 SUPER", "gpu_bytes": 32760, "nvidia_smi": f"{uuid}, RTX 4080 SUPER, 595.71.05, 32760 MiB",
            "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "matmul_tf32": False, "cudnn_tf32": False,
        },
    }


def test_crosshost_allows_only_uuid_difference():
    a, b = protocol("GPU-a"), protocol("GPU-b")
    comparable_protocol(a, b)
    assert runtime_for_comparison(a["runtime"]) == runtime_for_comparison(b["runtime"])


def test_crosshost_rejects_frozen_input_or_runtime_change():
    a, b = protocol(), protocol("GPU-b")
    changed = copy.deepcopy(b)
    changed["reference_config"]["LR"] = 1e-4
    with pytest.raises(ValueError, match="frozen inputs"):
        comparable_protocol(a, changed)
    changed = copy.deepcopy(b)
    changed["runtime"]["packages"]["torch"] = "different"
    with pytest.raises(ValueError, match="dependency"):
        comparable_protocol(a, changed)
    changed = copy.deepcopy(b)
    changed["runtime"]["nvidia_smi"] = "GPU-b, RTX 4080 SUPER, 999.0, 32760 MiB"
    with pytest.raises(ValueError, match="dependency"):
        comparable_protocol(a, changed)


def test_aggregate_seal_detects_edits():
    record = sealed({"hosts": ["a", "b"], "rule": "paired"})
    assert "aggregate_id" in record
    modified = copy.deepcopy(record)
    modified["rule"] = "edited"
    payload = {k: v for k, v in modified.items() if k != "aggregate_id"}
    assert sealed(payload) != modified
