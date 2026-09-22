"""Read-only aggregation of paired strict MVCM runs completed on two hosts.

Host A supplies the already-completed seed 42 pair. Host B supplies the newly
completed seed 43/44 pairs. Host identity remains explicit: this script never
pretends that GPU UUIDs are identical or mutates either source suite.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_strict_replication import (
    MEASURES, SEEDS, VARIANTS, atomic_json, completed_record, file_hash,
    read_json, same_protocol, summarize, verified_protocol, write_summary,
)


def sealed(payload):
    return {**payload, "aggregate_id": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def verify_aggregate(path):
    saved = read_json(path)
    payload = {k: v for k, v in saved.items() if k != "aggregate_id"}
    if sealed(payload) != saved:
        raise ValueError(f"Aggregate protocol integrity failure: {path}")
    return saved


def suite_rows(root, label, required_seeds, allow_extra_planned=False):
    protocol = verified_protocol(root / "protocol.json")
    if protocol.get("variants") != list(VARIANTS):
        raise ValueError(f"{label}: variants are not full/without_mvcm.")
    declared = tuple(protocol.get("seeds", ()))
    if allow_extra_planned:
        if not set(required_seeds).issubset(declared):
            raise ValueError(f"{label}: does not declare required seeds {required_seeds}.")
    elif declared != tuple(required_seeds):
        raise ValueError(f"{label}: expected exactly seeds {required_seeds}, found {declared}.")
    rows = []
    for seed in required_seeds:
        for variant in VARIANTS:
            row = completed_record(root / f"seed_{seed}" / variant, protocol, variant, seed)
            if row is None:
                raise FileNotFoundError(f"{label}: missing completed result for seed={seed}, variant={variant}.")
            rows.append({**row, "execution_host": label})
    return protocol, rows


def runtime_for_comparison(runtime):
    """GPU UUID is intentionally excluded; it remains retained in provenance."""
    smi = [part.strip() for part in runtime["nvidia_smi"].split(",")]
    if len(smi) != 4:
        raise ValueError("Unexpected nvidia-smi identity format.")
    return {
        "packages": runtime["packages"], "cuda": runtime["cuda"], "cudnn": runtime["cudnn"],
        "gpu": runtime["gpu"], "gpu_bytes": runtime["gpu_bytes"],
        "driver_version": smi[2], "driver_memory": smi[3],
        "environment": runtime["environment"],
        "deterministic_algorithms": runtime["deterministic_algorithms"],
        "cudnn_benchmark": runtime["cudnn_benchmark"],
        "cudnn_deterministic": runtime["cudnn_deterministic"],
        "matmul_tf32": runtime["matmul_tf32"], "cudnn_tf32": runtime["cudnn_tf32"],
    }


def comparable_protocol(a, b):
    fields = ("baseline_protocol_id", "reference_config", "inventory", "selection",
              "primary_metric", "threshold_scan", "test_evaluation", "initialization")
    changed = [field for field in fields if a.get(field) != b.get(field)]
    if changed:
        raise ValueError("Source suites differ in frozen inputs: " + ", ".join(changed))
    if runtime_for_comparison(a["runtime"]) != runtime_for_comparison(b["runtime"]):
        raise ValueError("Source suites differ in dependency/CUDA/driver/strict-runtime settings.")


def source_evidence(root, protocol, rows):
    return {
        "root": str(root), "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_hash(root / "protocol.json"),
        "runtime": protocol["runtime"],
        "completion_sha256": {f"{row['seed']}/{row['variant']}": file_hash(
            root / f"seed_{row['seed']}" / row["variant"] / "completed.json") for row in rows},
    }


def aggregate_payload(host_a_root, protocol_a, rows_a, host_b_root, protocol_b, rows_b):
    comparable_protocol(protocol_a, protocol_b)
    payload = {
        "schema": 1,
        "purpose": "Cross-host blocked aggregation of paired strict MVCM confirmation runs",
        "seeds": list(SEEDS), "variants": list(VARIANTS),
        "host_a": source_evidence(host_a_root, protocol_a, rows_a),
        "host_b": source_evidence(host_b_root, protocol_b, rows_b),
        "cross_host_rule": "Each full/without_mvcm pair is trained on the same host within its seed. GPU UUID may differ across seed blocks; it is retained above and not treated as identical hardware.",
        "comparison_scope": "Paired module differences by seed; cross-host execution is a blocking factor, not an independently replicated test set.",
    }
    return sealed(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-a-suite", required=True, help="Imported seed 42 source suite")
    parser.add_argument("--host-b-suite", required=True, help="New seed 43/44 source suite")
    parser.add_argument("--output-dir", required=True, help="New aggregate directory; source suites are read only")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    host_a, host_b, output = (Path(value).resolve() for value in
                              (args.host_a_suite, args.host_b_suite, args.output_dir))
    if output in (host_a, host_b) or output in host_a.parents or output in host_b.parents or host_a in output.parents or host_b in output.parents:
        raise ValueError("Aggregate output must not overlap either source suite.")
    protocol_a, rows_a = suite_rows(host_a, "host_a", (42,), allow_extra_planned=True)
    protocol_b, rows_b = suite_rows(host_b, "host_b", (43, 44))
    payload = aggregate_payload(host_a, protocol_a, rows_a, host_b, protocol_b, rows_b)
    rows = rows_a + rows_b
    result = summarize(rows, SEEDS)
    result["execution_hosts"] = {"seed_42": "host_a", "seed_43": "host_b", "seed_44": "host_b"}
    result["cross_host_note"] = payload["cross_host_rule"]
    result["aggregate_id"] = payload["aggregate_id"]
    if args.check_only:
        print(json.dumps({"status": "validated", "aggregate_id": payload["aggregate_id"],
                          "completed_runs": result["completed_runs"], "summary": result}, ensure_ascii=False, indent=2))
        return
    if (output / "aggregate_protocol.json").exists():
        same_protocol(verify_aggregate(output / "aggregate_protocol.json"), payload)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Aggregate output is nonempty without its protocol.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "aggregate_protocol.json", payload)
    # Keep the original source rows intact; only add explicit host provenance to the aggregate CSV.
    write_summary(output, rows, SEEDS, extra_columns=("execution_host",))
    atomic_json(output / "replication_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Cross-host aggregate written:", output)


if __name__ == "__main__":
    main()
