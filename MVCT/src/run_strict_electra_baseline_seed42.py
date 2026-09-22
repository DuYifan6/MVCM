"""Strict post-hoc ELECTRA-base extension for the frozen MVCT protocol.

This runner intentionally uses a new suite instead of modifying the sealed
four-baseline v1 suite.  It trains one ``google/electra-base-discriminator``
dual-head baseline, selects its checkpoint only by combined validation loss,
and permits one locked test evaluation at the fixed Fake/Real threshold 0.55.
The extension was specified after the existing main and baseline test results
were observed; that timing is recorded in the protocol and must be disclosed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import run_locked_three_channel_test as locked
import run_strict_baselines_seed42 as base
import run_strict_replication as strict
import run_strict_three_channel_ablation as three
from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


SOURCE_DIR = Path(__file__).resolve().parent
SEED = 42
VARIANT = "electra"
FAKE_THRESHOLD = 0.55
MEASURES = base.MEASURES
MODEL_SPEC = {
    "kind": "neural",
    "tokenizer": "electra",
    "pretrained_model": "google/electra-base-discriminator",
    "dropout": 0.1,
    "lr": 2e-5,
    "weight_decay": 0.01,
}


def prepare(
    baseline_source: Path,
    channel_suite: Path,
    three_suite: Path,
    locked_main_suite: Path,
    existing_baseline_suite: Path,
    electra_model: Path,
):
    source_protocol, reference, train, val, _ = strict.prepare(
        baseline_source, (SEED,)
    )
    channel_protocol = strict.verified_protocol(channel_suite / "protocol.json")
    three_protocol = strict.verified_protocol(three_suite / "protocol.json")

    def verify_gpu_relocated_source_protocol(current, historical):
        """Allow only GPU UUID relocation; all substantive runtime fields stay fixed."""
        import copy

        current_checked = copy.deepcopy(current)
        historical_checked = copy.deepcopy(historical)

        current_smi = current_checked["runtime"].get("nvidia_smi")
        historical_smi = historical_checked["runtime"].get("nvidia_smi")

        def without_uuid(value):
            parts = [part.strip() for part in str(value).split(",")]
            if len(parts) != 4 or not parts[0].startswith("GPU-"):
                raise ValueError(f"Unexpected nvidia-smi identity: {value}")
            return parts[1:]

        current_hardware = without_uuid(current_smi)
        historical_hardware = without_uuid(historical_smi)

        if current_hardware != historical_hardware:
            raise ValueError(
                "GPU runtime is substantively different: "
                f"{historical_hardware} -> {current_hardware}"
            )

        # Normalize only the non-substantive physical-device UUID.
        current_checked["runtime"]["nvidia_smi"] = current_hardware
        historical_checked["runtime"]["nvidia_smi"] = historical_hardware

        three.verify_source_protocol(current_checked, historical_checked)

    verify_gpu_relocated_source_protocol(source_protocol, channel_protocol)
    verify_gpu_relocated_source_protocol(source_protocol, three_protocol)

    locked_protocol = strict.verified_protocol(locked_main_suite / "protocol.json")
    if (
        locked_protocol.get("channel_protocol_id") != channel_protocol["protocol_id"]
        or locked_protocol.get("three_channel_protocol_id")
        != three_protocol["protocol_id"]
    ):
        raise ValueError("Locked main-model test does not match verified sources.")
    main_summary_path = locked_main_suite / "locked_test_summary.json"
    main_summary = strict.read_json(main_summary_path)
    if main_summary.get("completed_evaluations") != 5:
        raise ValueError("The five locked main-model evaluations must be complete.")

    existing_protocol = strict.verified_protocol(
        existing_baseline_suite / "protocol.json"
    )
    existing_summary_path = existing_baseline_suite / "baseline_test_summary.json"
    existing_summary = strict.read_json(existing_summary_path)
    if existing_summary.get("completed") != 4:
        raise ValueError("The sealed four-baseline test suite must be complete.")
    if existing_protocol.get("locked_main_test_protocol_id") != locked_protocol["protocol_id"]:
        raise ValueError("Existing baselines and the ELECTRA extension use different main protocols.")
    if float(existing_summary.get("fake_threshold")) != FAKE_THRESHOLD:
        raise ValueError("Existing baselines did not use the fixed threshold 0.55.")

    electra_model = electra_model.resolve()
    required = ("config.json", "tokenizer_config.json")
    if not electra_model.is_dir() or any(
        not (electra_model / name).is_file() for name in required
    ):
        raise FileNotFoundError(
            "Local ELECTRA model is incomplete: " + str(electra_model)
        )

    payload = {
        key: value for key, value in source_protocol.items() if key != "protocol_id"
    }
    entries = dict(payload["entry_sha256"])
    for name in (
        Path(__file__).name,
        "run_strict_baselines_seed42.py",
        "baseline_models.py",
        "dataset_cat.py",
        "train_cat.py",
        "utils.py",
    ):
        entries[name] = strict.file_hash(SOURCE_DIR / name)

    spec = dict(MODEL_SPEC)
    spec["model_path"] = str(electra_model)
    payload.update(
        {
            "purpose": "Controlled post-hoc ELECTRA-base dual-head baseline extension",
            "seeds": [SEED],
            "variants": [VARIANT],
            "entry_sha256": entries,
            "channel_protocol_id": channel_protocol["protocol_id"],
            "three_channel_protocol_id": three_protocol["protocol_id"],
            "locked_main_test_protocol_id": locked_protocol["protocol_id"],
            "locked_main_summary_sha256": strict.file_hash(main_summary_path),
            "existing_baseline_protocol_id": existing_protocol["protocol_id"],
            "existing_baseline_summary_sha256": strict.file_hash(existing_summary_path),
            "model_specs": {VARIANT: spec},
            "training_protocol": {
                "input": "title + body + description with the frozen token budgets",
                "seed": SEED,
                "batch_size": int(reference.BATCH_SIZE),
                "epochs": int(reference.EPOCHS),
                "early_stopping_patience": int(reference.EARLYSTOP_PATIENCE),
                "fake_loss": {
                    "name": "focal",
                    "alpha": float(reference.FOCAL_ALPHA),
                    "gamma": float(reference.FOCAL_GAMMA),
                },
                "ai_loss": "0.5 * cross_entropy",
                "checkpoint_selection": "minimum combined validation task loss",
                "fake_threshold": FAKE_THRESHOLD,
                "test_gate": "test evaluation forbidden until the ELECTRA validation checkpoint is frozen",
            },
            "selection": "One fixed ELECTRA-base specification; no hyperparameter grid or test-based selection",
            "primary_metric": "Test Fake/Real macro-F1 at fixed strict > 0.55",
            "threshold_scan": "Forbidden",
            "test_evaluation": "Second stage only after the validation-selected checkpoint is frozen",
            "timing_disclosure": (
                "ELECTRA was added after the existing main-model and four-baseline "
                "test results were observed. Its architecture and training protocol "
                "were fixed before its own locked test evaluation."
            ),
            "literature_position": (
                "ELECTRA is a general pretrained encoder introduced in 2020 and "
                "evaluated as a fake-news backbone in the 2024 GossipCop++ study; "
                "this dual-head adaptation is task-relevant, not a new task-specific architecture."
            ),
        }
    )
    return strict.seal(payload), reference, len(train), len(val)


def train_worker(args) -> None:
    import torch

    protocol = strict.verified_protocol(Path(args.suite_dir) / "protocol.json")
    if strict.file_hash(__file__) != protocol["entry_sha256"][Path(__file__).name]:
        raise ValueError("ELECTRA runner changed after protocol sealing.")
    slot = Path(args.suite_dir) / VARIANT
    output = Path(args.output).resolve()
    if output.parent != slot.resolve() or (slot / "completed.json").exists():
        raise ValueError("Invalid or completed ELECTRA training slot.")
    if not torch.cuda.is_available():
        raise RuntimeError("The ELECTRA training protocol requires CUDA.")

    config = SimpleNamespace(**protocol["reference_config"])
    strict.atomic_json(output / "config.json", vars(config))
    metrics, epoch, loss = base.train_neural(VARIANT, protocol, config, output)
    row = base.metrics_row(VARIANT, metrics, output, epoch, loss)
    artifacts = base.TRAIN_ARTIFACTS["neural"]
    row.update(
        {
            "protocol_id": protocol["protocol_id"],
            "stage": "validation_frozen",
            "artifact_sha256": {
                name: strict.file_hash(output / name) for name in artifacts
            },
        }
    )
    strict.atomic_json(output / "worker_result.json", row)
    print("ELECTRA validation checkpoint frozen.", flush=True)


def evaluate_test_worker(args) -> None:
    import torch
    from transformers import AutoTokenizer

    protocol = strict.verified_protocol(Path(args.suite_dir) / "protocol.json")
    train_row = base.training_record(
        Path(args.suite_dir) / VARIANT, protocol, VARIANT
    )
    output = Path(args.output).resolve()
    test_slot = Path(args.suite_dir) / "locked_test" / VARIANT
    if output.parent != test_slot.resolve() or (test_slot / "completed.json").exists():
        raise ValueError("Invalid or completed ELECTRA test slot.")

    config = SimpleNamespace(
        **strict.read_json(Path(train_row["output_dir"]) / "config.json")
    )
    frame, split_path = base.restore_frame(config, "test")
    _, model_path = base.model_paths(protocol, VARIANT)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    loader = base.make_loader(frame, tokenizer, config, False)
    model = base.build_neural(VARIANT, protocol, tokenizer)
    checkpoint = torch.load(
        str(Path(train_row["output_dir"]) / "best_checkpoint.pt"),
        map_location="cpu",
        mmap=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    values = base.neural_probabilities(model, loader, device)
    metrics = base.save_evaluation(output, "test", *values)
    metadata = {
        "protocol_id": protocol["protocol_id"],
        "variant": VARIANT,
        "seed": SEED,
        "split": "test",
        "saved_split": str(split_path.resolve()),
        "fake_threshold": FAKE_THRESHOLD,
        "test_based_selection": False,
        "training_completion_sha256": strict.file_hash(
            Path(args.suite_dir) / VARIANT / "completed.json"
        ),
    }
    strict.atomic_json(output / "metadata.json", metadata)
    row = base.metrics_row(
        VARIANT,
        metrics,
        output,
        train_row["checkpoint_epoch"],
        train_row["val_loss"],
    )
    row.update(
        {
            "protocol_id": protocol["protocol_id"],
            "stage": "locked_test",
            "artifact_sha256": {
                name: strict.file_hash(output / name)
                for name in base.TEST_ARTIFACTS
            },
        }
    )
    strict.atomic_json(output / "worker_result.json", row)
    del model, loader, frame
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Locked ELECTRA test completed.", flush=True)


def summarize(row, stage: str) -> dict:
    return {
        "stage": stage,
        "seed": SEED,
        "planned": 1,
        "completed": int(row is not None),
        "fake_threshold": FAKE_THRESHOLD,
        "by_variant": {
            VARIANT: (
                {key: row[key] for key in (*MEASURES, "val_loss")}
                if row is not None
                else None
            )
        },
        "note": (
            "Post-hoc controlled ELECTRA-base extension. Validation loss selects "
            "the checkpoint; no ELECTRA test result changes a model or threshold."
        ),
    }


def write_summary(root: Path, row, stage: str) -> None:
    prefix = "electra_validation" if stage == "validation_frozen" else "electra_test"
    strict.atomic_json(root / f"{prefix}_summary.json", summarize(row, stage))
    temporary = root / f"{prefix}_results.csv.tmp"
    columns = (
        "variant",
        "seed",
        "checkpoint_epoch",
        "val_loss",
        "fake_threshold",
        *MEASURES,
        "output_dir",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        if row is not None:
            writer.writerow(row)
    os.replace(temporary, root / f"{prefix}_results.csv")


def launch_child(arguments, lock, log=None) -> None:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *arguments]
    child = subprocess.Popen(
        command,
        cwd=str(SOURCE_DIR),
        env=strict.strict_environment(),
        stdout=log,
        stderr=subprocess.STDOUT if log else None,
        **lock.subprocess_options(),
    )
    code = child.wait()
    if code:
        raise RuntimeError(
            f"ELECTRA worker exited with code {code}; inspect worker.log."
        )


def main(argv=None) -> None:
    from config import cfg

    replications = Path(cfg.OUTPUT_DIR) / "replications"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-source", default=str(replications / "missing_evidence_v1")
    )
    parser.add_argument(
        "--channel-suite",
        default=str(replications / "strict_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--three-suite",
        default=str(replications / "strict_three_channel_ablation_seed42_v1"),
    )
    parser.add_argument(
        "--locked-main-suite",
        default=str(replications / "locked_three_channel_test_v1"),
    )
    parser.add_argument(
        "--existing-baseline-suite",
        default=str(replications / "strict_baselines_seed42_v1"),
    )
    parser.add_argument(
        "--suite-dir",
        default=str(replications / "strict_electra_baseline_seed42_v1"),
    )
    parser.add_argument(
        "--electra-model", default="models/electra-base-discriminator"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--worker", choices=("preflight", "train", "test"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    path_names = (
        "baseline_source",
        "channel_suite",
        "three_suite",
        "locked_main_suite",
        "existing_baseline_suite",
        "suite_dir",
    )
    paths = {name: Path(getattr(args, name)).resolve() for name in path_names}
    for name, value in paths.items():
        setattr(args, name, str(value))
    args.electra_model = str(Path(args.electra_model).resolve())

    if args.worker == "preflight":
        protocol, _, train_n, val_n = prepare(
            paths["baseline_source"],
            paths["channel_suite"],
            paths["three_suite"],
            paths["locked_main_suite"],
            paths["existing_baseline_suite"],
            Path(args.electra_model),
        )
        strict.atomic_json(
            args.output,
            {"protocol": protocol, "train_n": train_n, "val_n": val_n},
        )
        return
    if args.worker == "train":
        train_worker(args)
        return
    if args.worker == "test":
        evaluate_test_worker(args)
        return

    root = paths["suite_dir"]
    sources = [paths[name] for name in path_names if name != "suite_dir"]
    locked.protect_output(root, sources)
    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        with tempfile.TemporaryDirectory(prefix="mvct_electra_preflight_") as temporary:
            result_path = Path(temporary) / "result.json"
            launch_child(
                [
                    "--worker",
                    "preflight",
                    "--baseline-source",
                    args.baseline_source,
                    "--channel-suite",
                    args.channel_suite,
                    "--three-suite",
                    args.three_suite,
                    "--locked-main-suite",
                    args.locked_main_suite,
                    "--existing-baseline-suite",
                    args.existing_baseline_suite,
                    "--electra-model",
                    args.electra_model,
                    "--output",
                    str(result_path),
                ],
                lock,
            )
            result = strict.read_json(result_path)
        protocol = result["protocol"]
        if (root / "protocol.json").exists():
            strict.same_protocol(
                strict.verified_protocol(root / "protocol.json"), protocol
            )
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Nonempty ELECTRA suite has no protocol; use a new directory.")

        train_row = base.training_record(root / VARIANT, protocol, VARIANT)
        test_row = base.test_record(
            root / "locked_test" / VARIANT, protocol, VARIANT
        )
        print(
            f"Preflight passed: seed 42; train/val={result['train_n']}/{result['val_n']}; "
            f"ELECTRA frozen={int(train_row is not None)}/1; "
            f"test evaluated={int(test_row is not None)}/1.",
            flush=True,
        )
        if args.check_only:
            print("Check only: no ELECTRA suite files written.")
            return

        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            strict.atomic_json(root / "protocol.json", protocol)

        if not args.evaluate_test:
            write_summary(root, train_row, "validation_frozen")
            if train_row is None:
                slot = root / VARIANT
                output = slot / (
                    "attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
                output.mkdir(parents=True, exist_ok=False)
                print(f"Train ELECTRA; log: {output / 'worker.log'}", flush=True)
                with (output / "worker.log").open("w", encoding="utf-8") as log:
                    launch_child(
                        [
                            "--worker",
                            "train",
                            "--suite-dir",
                            str(root),
                            "--output",
                            str(output),
                        ],
                        lock,
                        log,
                    )
                candidate = strict.read_json(output / "worker_result.json")
                base.training_record(slot, protocol, VARIANT, candidate)
                strict.atomic_json(slot / "completed.json", candidate)
                train_row = candidate
                write_summary(root, train_row, "validation_frozen")
            print("ELECTRA frozen on validation. Run again with --evaluate-test.")
            return

        if train_row is None:
            raise RuntimeError(
                "Test gate closed: freeze the ELECTRA validation checkpoint first."
            )
        write_summary(root, test_row, "locked_test")
        if test_row is None:
            slot = root / "locked_test" / VARIANT
            output = slot / (
                "evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            output.mkdir(parents=True, exist_ok=False)
            print(f"Evaluate ELECTRA; log: {output / 'worker.log'}", flush=True)
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                launch_child(
                    [
                        "--worker",
                        "test",
                        "--suite-dir",
                        str(root),
                        "--output",
                        str(output),
                    ],
                    lock,
                    log,
                )
            candidate = strict.read_json(output / "worker_result.json")
            base.test_record(slot, protocol, VARIANT, candidate)
            strict.atomic_json(slot / "completed.json", candidate)
            test_row = candidate
            write_summary(root, test_row, "locked_test")
        print("Controlled ELECTRA extension completed:", root)


if __name__ == "__main__":
    main()
