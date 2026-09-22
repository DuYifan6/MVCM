"""Independent short-trace reproducibility audit; no test evaluation/checkpoints.

The parent is stdlib-only. Each child starts a fresh interpreter and observes the
shared train_one_epoch plus either the old or new validation routine. This is
NOT a replay of the complete historical runner or checkpoint serialization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from datetime import datetime
from itertools import islice

from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    """Hash nested dense tensors including dtype, shape and exact raw bytes."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().contiguous().cpu()
            digest.update(str((str(x.dtype), list(x.shape))).encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode() + b"[")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode() + b"\0")
    visit(value)
    return digest.hexdigest()


def rng_hash():
    import numpy as np
    import torch
    n = np.random.get_state()
    return fingerprint({"python": random.getstate(),
                        "numpy": (n[0], n[1].tobytes(), *n[2:]),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


class TraceLoader:
    def __init__(self, loader, steps, phase, trace, current, cache_paths):
        self.loader, self.steps, self.phase = loader, steps, phase
        self.trace, self.current, self.cache_paths = trace, current, cache_paths

    def __len__(self):
        return min(self.steps, len(self.loader))

    def __iter__(self):
        for step, batch in enumerate(islice(self.loader, self.steps), 1):
            ids = batch["sample_id"].tolist()
            record = {"phase": self.phase, "batch": step, "sample_ids": ids,
                      "input_hash": fingerprint(batch), "rng_before": rng_hash(),
                      "cache_file_hashes": {str(i): file_hash(self.cache_paths[i])
                                            for i in ids if i in self.cache_paths}}
            self.trace.append(record)
            self.current["record"] = record
            yield batch
        self.current["record"] = None


def trace_training(model, optimizer, train_loader, val_loader, config, steps, val_steps, entry, cache_paths):
    import torch
    import train_cat
    import train_checkpoint_selection
    focal = train_cat.FocalLossCE(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                                 weight=config.FAKE_CLASS_WEIGHT)
    if int(getattr(config, "ACCUM_STEPS", 1)) != 1:
        raise ValueError("Short per-update audit currently requires ACCUM_STEPS=1; no automatic change made.")
    trace, phases, current = [], [], {"record": None}
    initial = fingerprint(model.state_dict())
    initial_parts = {k: fingerprint(v) for k, v in model.state_dict().items()}
    initial_rng = rng_hash()
    initial_optimizer = fingerprint(optimizer.state_dict())
    original_step = optimizer.step
    original_loss, original_regularizer = train_cat._task_loss, train_cat.attention_kl_regularizer

    def loss_observer(*args, **kwargs):
        loss = original_loss(*args, **kwargs)
        if current["record"] is not None:
            current["record"]["task_loss"] = float(loss.detach())
        return loss

    def regularizer_observer(*args, **kwargs):
        reg = original_regularizer(*args, **kwargs)
        if current["record"] is not None:
            current["record"]["attention_kl"] = float(reg.detach())
        return reg

    def forward_observer(module, inputs, output):
        if current["record"] is not None:
            current["record"]["logits_hash"] = fingerprint(output[:2])

    def step_observer(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if current["record"] is not None:
            current["record"]["weights_after"] = fingerprint(model.state_dict())
            current["record"]["rng_after"] = rng_hash()
            print(f"Traced {current['record']['phase']} batch {current['record']['batch']}", flush=True)
        return result

    hook = model.register_forward_hook(forward_observer)
    optimizer.step = step_observer
    train_cat._task_loss = loss_observer
    train_cat.attention_kl_regularizer = regularizer_observer
    try:
        for phase in ("train_1", "validation", "train_2"):
            loader = TraceLoader(val_loader if phase == "validation" else train_loader,
                                 val_steps if phase == "validation" else steps,
                                 phase, trace, current, cache_paths)
            if phase != "validation":
                stats = train_cat.train_one_epoch(model, loader, optimizer, config.DEVICE, config, focal)
                common = stats
            elif entry == "old":
                loss = train_cat.evaluate_epoch(model, loader, config.DEVICE, focal)
                common = {"val_loss": loss}
            else:
                stats = train_checkpoint_selection.evaluate_selection_epoch(
                    model, loader, config.DEVICE, focal, config.FAKE_THRESHOLD)
                common = {"val_loss": stats["val_loss"]}
            phases.append({"phase": phase, "metrics": common,
                           "weights": fingerprint(model.state_dict()),
                           "optimizer": fingerprint(optimizer.state_dict()), "rng": rng_hash()})
    finally:
        hook.remove()
        optimizer.step = original_step
        train_cat._task_loss, train_cat.attention_kl_regularizer = original_loss, original_regularizer
    # The new validator imports _task_loss directly; compare common observations only.
    for record in trace:
        if record["phase"] == "validation":
            record.pop("task_loss", None)
    return {"initial_weights": initial, "initial_parameter_hashes": initial_parts,
            "initial_rng": initial_rng, "initial_optimizer": initial_optimizer,
            "trace": trace, "phases": phases}


def configure_runtime(mode, fixture):
    import torch
    if not fixture and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; real audit requires the existing GPU environment.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if mode == "strict":
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def real_inputs(args):
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from dataset_cat import load_and_split_with_mvc_cat
    from diagnose_validation import signature_from_config
    from run_missing_evidence import check_saved_split

    protocol_path = Path(args.suite_dir) / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != protocol["protocol_id"]:
        raise ValueError("Existing protocol integrity check failed.")
    config = SimpleNamespace(**protocol["reference_config"])
    source_dir = Path(__file__).resolve().parent
    source_hashes = {name: file_hash(source_dir / name) for name in protocol["source_sha256"]}
    if source_hashes != protocol["source_sha256"]:
        raise ValueError("Frozen experiment source files differ; do not edit the old protocol.")
    data_hash = file_hash(config.DATA_PATH)
    split_hashes = {name: file_hash(Path(config.SPLIT_DIR) / f"{name}.csv") for name in ("train", "val", "test")}
    if data_hash != protocol["dataset_sha256"] or split_hashes != protocol["split_sha256"]:
        raise ValueError("Data or saved split file differs from the experiment protocol.")
    signature = signature_from_config(config)
    manifest = json.loads((Path(config.MVC_CACHE_DIR) / "cache_manifest.json").read_text())
    if manifest["builder_signature"] != signature or signature != protocol["cache_signature"]:
        raise ValueError("Cache signature differs; no cache is rebuilt.")
    model_dir = Path(config.BERT_MODEL)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local BERT/tokenizer directory missing: {model_dir}")
    model_files = {str(p.relative_to(model_dir)): file_hash(p) for p in sorted(model_dir.rglob("*"))
                   if p.is_file() and ".cache" not in p.relative_to(model_dir).parts
                   and p.suffix in (".json", ".txt", ".bin", ".safetensors", ".model")}
    if not any(name.endswith((".bin", ".safetensors")) for name in model_files):
        raise FileNotFoundError("Local pretrained model weights missing.")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL, local_files_only=True, use_fast=True)
    train, val, test = load_and_split_with_mvc_cat(
        config.DATA_PATH, tokenizer, SimpleNamespace(cache_signature=signature),
        mvc_cache_dir=config.MVC_CACHE_DIR, precompute=False, random_state=42,
        max_len_title=config.MAX_LEN_TITLE, max_len_text=config.MAX_LEN_TEXT,
        max_len_desc=config.MAX_LEN_DESC, drop_incomplete=config.DROP_INCOMPLETE,
        max_body_chars=config.MAX_BODY_CHARS, group_column=config.GROUP_COLUMN, split_dir=None,
        drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES)
    for name, dataset in (("train", train), ("val", val), ("test", test)):
        check_saved_split(dataset, Path(config.SPLIT_DIR) / f"{name}.csv")
    del test  # Membership only: no test model inputs or predictions.
    cache_paths = {int(row.ID): str(row.mvc_path) for ds in (train, val) for row in ds.df.itertuples()}
    inventory = {"protocol_id": protocol["protocol_id"], "source_hashes": source_hashes,
                 "data_hash": data_hash, "split_hashes": split_hashes, "bert_files": model_files,
                 "cache_signature": signature,
                 "cache_scope": "Only raw cache files actually observed in this short trace are hashed."}
    config.DEVICE = "cuda"
    return config, train, val, cache_paths, inventory


def fixture_inputs():
    """Synthetic CPU fixture for tests only; never presented as a real audit."""
    import torch
    from types import SimpleNamespace
    from torch.utils.data import Dataset

    class TinyDataset(Dataset):
        def __len__(self):
            return 12

        def __getitem__(self, i):
            return {"sequence": {"input_ids": torch.tensor([1, 2 + i % 4, 3, 4]),
                                 "attention_mask": torch.ones(4, dtype=torch.long),
                                 "view_ids": torch.tensor([-1, 0, 1, 2])},
                    "mvc": torch.full((3, 3, 4), .7), "sample_id": torch.tensor(i),
                    "label_fake": torch.tensor(i % 2), "label_ai": torch.tensor((i // 2) % 2)}
    cfg = SimpleNamespace(DEVICE="cpu", BATCH_SIZE=2, TRAIN_NUM_WORKERS=0, VAL_NUM_WORKERS=0,
                          PIN_MEMORY=False, ACCUM_STEPS=1, FOCAL_ALPHA=.8, FOCAL_GAMMA=2.,
                          FAKE_CLASS_WEIGHT=None, FAKE_THRESHOLD=.55, ATTN_REG_WEIGHT=.03,
                          LR=2e-5, WEIGHT_DECAY=.01, CAT_LAYERS=1, CAT_HEADS=2,
                          CAT_FF_MULTIPLIER=2, CAT_DROPOUT=.1, CAT_LAMBDA_INIT=1.,
                          LEARNABLE_MVC_WEIGHTS=True, SEM_WEIGHT=.4, FACT_WEIGHT=.25,
                          LOGIC_WEIGHT=.2, LOCAL_WEIGHT=.15)
    return cfg, TinyDataset(), TinyDataset(), {}, {"fixture": True}


def worker(args):
    import numpy as np
    import torch
    import transformers
    from importlib.metadata import version
    from types import SimpleNamespace
    from model_cat import MVCTTokenCATModel
    from run_repeated_validation import make_loader
    from utils import set_seed

    configure_runtime(args.mode, args.fixture)
    config, train, val, cache_paths, inventory = fixture_inputs() if args.fixture else real_inputs(args)
    if args.steps > (len(train) + config.BATCH_SIZE - 1) // config.BATCH_SIZE:
        raise ValueError("Requested steps exceed available training batches.")
    set_seed(args.seed)
    train_loader = make_loader(train, config, args.seed, True)
    val_loader = make_loader(val, config, args.seed, False)
    encoder = None
    if args.fixture:
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = torch.nn.Embedding(16, 8)

            def forward(self, input_ids, attention_mask, return_dict=True):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        encoder = TinyEncoder()
    model = MVCTTokenCATModel(
        bert_model_name=getattr(config, "BERT_MODEL", "fixture"), encoder=encoder,
        mvc_weights=(config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT),
        cat_layers=config.CAT_LAYERS, cat_heads=config.CAT_HEADS,
        cat_ff_multiplier=config.CAT_FF_MULTIPLIER, dropout_p=config.CAT_DROPOUT,
        lambda_init=config.CAT_LAMBDA_INIT, learnable_mvc_weights=config.LEARNABLE_MVC_WEIGHTS,
        mask_missing_evidence=False).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    result = trace_training(model, optimizer, train_loader, val_loader, config,
                            args.steps, args.val_steps, args.entry, cache_paths)
    gpu_runtime = None
    if torch.cuda.is_available():
        try:
            queried = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                      "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            gpu_runtime = queried.stdout.strip() if queried.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    result.update(entry=args.entry, mode=args.mode, seed=args.seed, fixture=args.fixture,
                  inventory=inventory, config=vars(config),
                  scope="Instrumented short shared train_one_epoch / old-new validation / train_one_epoch; excludes full-run checkpoint I/O.",
                  environment={"python": sys.version, "executable": sys.executable,
                               "torch": torch.__version__, "numpy": np.__version__,
                               "transformers": transformers.__version__, "sklearn": version("scikit-learn"),
                               "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                               "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                               "nvidia_smi_identity": gpu_runtime,
                               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                               "env": {k: os.environ.get(k) for k in (
                                   "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                   "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")}})
    write_json(Path(args.output) / "trace.json", result)


def first_difference(left, right):
    for key in ("inventory", "config", "environment", "initial_weights", "initial_optimizer", "initial_rng"):
        if left.get(key) != right.get(key):
            result = {"stage": key}
            if key == "initial_weights":
                a, b = left["initial_parameter_hashes"], right["initial_parameter_hashes"]
                result["first_parameter"] = next((k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)), None)
            return result
    if len(left["trace"]) != len(right["trace"]):
        return {"stage": "trace_length"}
    for index, (a, b) in enumerate(zip(left["trace"], right["trace"])):
        for key in ("phase", "batch", "sample_ids", "input_hash", "cache_file_hashes", "rng_before",
                    "logits_hash", "task_loss", "attention_kl", "weights_after", "rng_after"):
            if a.get(key) != b.get(key):
                return {"stage": key, "phase": a["phase"], "batch": a["batch"],
                        "left": a.get(key), "right": b.get(key)}
        # Check the phase boundary before considering a later training batch.
        last_in_phase = index + 1 == len(left["trace"]) or left["trace"][index + 1]["phase"] != a["phase"]
        if last_in_phase:
            ap = next(p for p in left["phases"] if p["phase"] == a["phase"])
            bp = next(p for p in right["phases"] if p["phase"] == a["phase"])
            if ap != bp:
                return {"stage": "phase_end", "phase": a["phase"], "left": ap, "right": bp}
    return None


def parent(args):
    if not args.output:
        from config import cfg
        root = Path(cfg.OUTPUT_DIR) / "repro_checks" / (args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    else:
        root = Path(args.output)
    root = root.resolve()
    suite = Path(args.suite_dir).resolve()
    if root == suite or suite in root.parents:
        raise ValueError("Audit output must not be placed inside the frozen experiment suite.")
    source = Path(__file__).resolve()
    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        root.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(name, "4")
        if args.mode == "strict":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["PYTHONHASHSEED"] = str(args.seed)
        plan = {"mode": args.mode, "seed": args.seed, "steps_per_train_phase": args.steps,
                "train_phases_per_process": 2, "validation_batches": args.val_steps,
                "processes": ["old_a", "old_b", "new_a", "new_b"], "fixture": args.fixture,
                "test_evaluation": False, "saved_model_checkpoints": False,
                "audit_sources": {source.name: file_hash(source), "runtime_guard.py": file_hash(source.parent / "runtime_guard.py")}}
        write_json(root / "plan.json", plan)
        reports = {}
        for name in plan["processes"]:
            output = root / name
            output.mkdir()
            command = [sys.executable, "-u", str(source), "--worker", "--entry", name.split("_")[0],
                       "--mode", args.mode, "--seed", str(args.seed), "--steps", str(args.steps),
                       "--val-steps", str(args.val_steps), "--suite-dir", args.suite_dir, "--output", str(output)]
            if args.fixture:
                command.append("--fixture")
            print(f"Starting {name}; log: {output / 'worker.log'}", flush=True)
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(source.parent), env=env,
                                           stdout=log, stderr=subprocess.STDOUT, **lock.subprocess_options())
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    print("Waiting for audit child to exit before releasing lock.", flush=True)
                    code = process.wait()
            if code:
                summary = {"status": "worker_failed", "worker": name, "exit_code": code,
                           "log": str(output / "worker.log"), "output": str(root), "fixture": args.fixture}
                write_json(root / "summary.json", summary)
                print(json.dumps(summary, indent=2), flush=True)
                return code
            reports[name] = json.loads((output / "trace.json").read_text(encoding="utf-8"))
        pairs = [("old_a", "old_b"), ("new_a", "new_b"), ("old_a", "new_a"), ("old_b", "new_b")]
        comparisons = [{"left": a, "right": b, "first_difference": first_difference(reports[a], reports[b])} for a, b in pairs]
        summary = {"status": "trace_matches" if all(x["first_difference"] is None for x in comparisons) else "differences_detected",
                   "mode": args.mode, "seed": args.seed, "fixture": args.fixture, "comparisons": comparisons,
                   "output": str(root), "note": "Exact equality within the observed instrumented short traces only. Not proof of historical reproducibility, full-run equivalence, model correctness or performance improvement."}
        write_json(root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0


def main(argv=None):
    from config import cfg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "strict"), default="current")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10, help="Training batches in EACH of two short phases per process")
    parser.add_argument("--val-steps", type=int, default=2)
    parser.add_argument("--suite-dir", default=str(Path(cfg.OUTPUT_DIR) / "replications" / "checkpoint_selection_v1"))
    parser.add_argument("--output", help="New directory only; no overwrites")
    parser.add_argument("--fixture", action="store_true", help="Synthetic CPU self-test, not a real experiment")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--entry", choices=("old", "new"), default="old", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.val_steps < 1:
        parser.error("steps and val-steps must be positive")
    if args.worker:
        try:
            worker(args)
        except Exception:
            details = traceback.format_exc()
            if args.output:
                write_json(Path(args.output) / "error.json", {"traceback": details})
            print(details, file=sys.stderr)
            return 1
        return 0
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
