"""Read-only MVC cache audit. No model inference, cache repairs, or test-label analysis."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import cfg
from dataset_cat import _cache_key, _clean_and_group_dataframe
from diagnose_validation import signature_from_config

CHANNELS = ("semantic", "factual", "logical", "local")
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ("title_body", "title_description", "body_description")


def inspect_tensor(array):
    if array.shape != (3, 3, 4):
        return ["wrong_shape"]
    if not np.issubdtype(array.dtype, np.floating):
        return ["non_float_dtype"]
    if not np.isfinite(array).all():
        return ["non_finite"]
    issues = []
    if (array < 0).any() or (array > 1).any():
        issues.append("outside_0_1")
    if not np.allclose(array, array.transpose(1, 0, 2), atol=1e-6, rtol=0):
        issues.append("asymmetric")
    if not np.allclose(array[np.arange(3), np.arange(3)], 1, atol=1e-6, rtol=0):
        issues.append("diagonal_not_one")
    return issues


def describe(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return {"n": 0}
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {
        "n": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
        **dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(float, quantiles))),
        "near_zero_fraction": float(np.isclose(values, 0, atol=1e-6, rtol=0).mean()),
        "near_half_fraction": float(np.isclose(values, .5, atol=1e-6, rtol=0).mean()),
        "near_one_fraction": float(np.isclose(values, 1, atol=1e-6, rtol=0).mean()),
        "below_half_fraction": float((values < .5).mean()),
    }


def profile(pairs):
    # Three undirected pairs per sample, not six mirrored observations.
    return {
        channel: {
            "all_three_pairs": describe(pairs[:, :, c]),
            **{name: describe(pairs[:, p, c]) for p, name in enumerate(PAIR_NAMES)},
        }
        for c, channel in enumerate(CHANNELS)
    }


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(value, path):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_splits(frame, split_dir):
    if not frame.ID.is_unique:
        raise ValueError("Dataset IDs are not unique; cannot map cached features safely.")
    source = frame.set_index("ID", drop=False)
    seen_ids, seen_groups, restored = set(), set(), []
    for name in ("train", "val", "test"):
        saved = pd.read_csv(Path(split_dir) / f"{name}.csv", dtype={"_split_group": str})
        if not len(saved) or not saved.ID.is_unique:
            raise ValueError(f"Empty split or duplicate IDs: {name}")
        selected = source.loc[saved.ID].copy().reset_index(drop=True)
        for column in ("label_fake", "label_ai", "_split_group"):
            if selected[column].astype(str).tolist() != saved[column].astype(str).tolist():
                raise ValueError(f"Dataset differs from saved split: {name}/{column}")
        ids, groups = set(selected.ID), set(selected._split_group)
        if seen_ids & ids or seen_groups & groups:
            raise ValueError(f"Cross-split ID/group overlap: {name}")
        seen_ids.update(ids)
        seen_groups.update(groups)
        selected["audit_split"] = name
        restored.append(selected)
    if seen_ids != set(frame.ID):
        raise ValueError("Saved splits do not cover the cleaned dataset exactly.")
    return pd.concat(restored, ignore_index=True)


def example_rows(frame, pairs):
    """Deterministic validation examples, explicitly selected for inspection."""
    mask = frame.audit_split.eq("val").to_numpy()
    selected_frame = frame.loc[mask].reset_index(drop=True)
    scores = pairs[mask]
    if not len(scores):
        return []
    factual_zero = np.isclose(scores[:, :, 1], 0, atol=1e-6, rtol=0)
    logic_half = np.isclose(scores[:, :, 2], .5, atol=1e-6, rtol=0)
    categories = {
        "factual_zero_and_logic_half_proxy": np.flatnonzero((factual_zero & logic_half).any(axis=1))[:4],
        "lowest_mean_logical": np.argsort(scores[:, :, 2].mean(axis=1), kind="stable")[:4],
        "largest_semantic_logical_gap": np.argsort(-np.abs(scores[:, :, 0] - scores[:, :, 2]).mean(axis=1), kind="stable")[:4],
        "random_validation": np.random.default_rng(42).choice(len(scores), min(4, len(scores)), replace=False),
    }
    examples = []
    for reason, indices in categories.items():
        for index in indices:
            row = selected_frame.iloc[int(index)]
            examples.append({
                "selection": reason, "sample_id": int(row.ID), "split": "val",
                "label_fake": int(row.label_fake), "label_ai": int(row.label_ai),
                "text_preview": {field: str(row[field])[:1200] for field in ("title", "text", "desc")},
                "text_lengths": {field: len(str(row[field])) for field in ("title", "text", "desc")},
                "pair_scores": {name: dict(zip(CHANNELS, scores[int(index), p].tolist())) for p, name in enumerate(PAIR_NAMES)},
            })
    return examples


def audit(args):
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    config = SimpleNamespace(**json.loads(config_path.read_text(encoding="utf-8")))
    cache_dir = Path(config.MVC_CACHE_DIR)
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = signature_from_config(config)
    if signature != manifest["builder_signature"]:
        raise ValueError("Saved run configuration and cache signature differ; audit stopped without changing files.")
    source = _clean_and_group_dataframe(
        config.DATA_PATH, drop_incomplete=config.DROP_INCOMPLETE,
        max_body_chars=config.MAX_BODY_CHARS, group_column=config.GROUP_COLUMN,
        drop_conflicting_duplicates=config.DROP_CONFLICTING_DUPLICATES,
        drop_exact_duplicates=config.DROP_EXACT_DUPLICATES,
    )
    frame = restore_splits(source, config.SPLIT_DIR)
    holder = SimpleNamespace(cache_signature=signature)
    issues, valid_indices, pair_values, expected_files = [], [], [], set()
    dtypes = Counter()
    for i, row in enumerate(tqdm(frame.itertuples(index=False), total=len(frame), desc="Audit MVC cache")):
        path = cache_dir / f"mvc_{_cache_key(row, holder)}.npy"
        expected_files.add(path.name)
        problems = []
        if not path.is_file():
            problems = ["missing"]
        else:
            try:
                array = np.load(path, allow_pickle=False)
                dtypes[str(array.dtype)] += 1
                problems = inspect_tensor(array)
            except Exception as error:
                problems = ["unreadable:" + type(error).__name__]
        if problems:
            issues.append({"sample_id": int(row.ID), "split": row.audit_split, "file": str(path), "issues": problems})
        else:
            valid_indices.append(i)
            pair_values.append(np.stack([array[a, b] for a, b in PAIR_INDICES]))
    valid = frame.iloc[valid_indices].reset_index(drop=True)
    pairs = np.stack(pair_values) if pair_values else np.empty((0, 3, 4), dtype=np.float32)
    observed = {path.name for path in cache_dir.glob("mvc_*.npy")}
    report = {
        "provenance": {"config": str(config_path.resolve()), "config_sha256": sha256(config_path),
                       "dataset_sha256": sha256(config.DATA_PATH), "cache_manifest_sha256": sha256(manifest_path),
                       "script_sha256": sha256(__file__), "signature": signature,
                       "split_sha256": {s: sha256(Path(config.SPLIT_DIR) / f"{s}.csv") for s in ("train", "val", "test")}},
        "integrity": {"expected_samples": len(frame), "valid_samples": len(valid),
                      "invalid_or_missing_samples": len(issues), "issue_counts": dict(Counter(k for row in issues for k in row["issues"])),
                      "dtypes": dict(dtypes), "split_counts": frame.audit_split.value_counts().to_dict(),
                      "valid_split_counts": valid.audit_split.value_counts().to_dict(),
                      "cross_split_group_overlap": 0, "unreferenced_npy_files": len(observed - expected_files),
                      "unreferenced_note": "May include old version caches. Not considered corrupt and not deleted."},
        "profiles": {}, "label_profiles": {}, "default_value_proxies": {},
        "channel_spearman_sample_means": {}, "initial_bias_diagnostic": {},
        "limitations": [
            "Only train and validation feature distributions/labels are analysed; test files receive structural checks only.",
            "Three undirected pairs per sample; within-sample pairs are not independent statistical observations. No significance tests.",
            "Factual=0 and logical=0.5 can indicate absent comparable triples but do not establish OpenIE failure.",
            "Saved matrices lack raw triples, candidate counts and NLI outputs; root causes cannot be recovered from matrices alone.",
            "Cache sanitization already clips/replaces non-finite values and forces diagonal=1, concealing some upstream failures.",
            "Label associations are descriptive; consistency is not an external truth check and no causal benefit is established.",
            "Signature checks settings, not model weight contents. Historical model snapshot integrity cannot be certified.",
        ],
    }
    weights = np.asarray([config.SEM_WEIGHT, config.FACT_WEIGHT, config.LOGIC_WEIGHT, config.LOCAL_WEIGHT])
    for split in ("train", "val"):
        mask = valid.audit_split.eq(split).to_numpy()
        part, metadata = pairs[mask], valid.loc[mask]
        report["profiles"][split] = profile(part)
        if not len(part):
            continue
        zeros = np.isclose(part[:, :, 1], 0, atol=1e-6, rtol=0)
        halves = np.isclose(part[:, :, 2], .5, atol=1e-6, rtol=0)
        report["default_value_proxies"][split] = {
            "pair_factual_zero_and_logical_half_fraction": float((zeros & halves).mean()),
            "samples_with_any_proxy_pair_fraction": float((zeros & halves).any(axis=1).mean()),
            "samples_with_all_three_proxy_pairs_fraction": float((zeros & halves).all(axis=1).mean()),
            "per_pair_proxy_fraction": dict(zip(PAIR_NAMES, (zeros & halves).mean(axis=0).tolist())),
        }
        means = part.mean(axis=1)
        report["channel_spearman_sample_means"][split] = pd.DataFrame(means, columns=CHANNELS).corr(method="spearman").to_dict()
        report["label_profiles"][split] = {}
        for label in ("label_fake", "label_ai"):
            report["label_profiles"][split][label] = {
                str(value): {"samples": int((metadata[label].to_numpy() == value).sum()),
                             "channel_sample_mean_scores": {c: describe(means[metadata[label].to_numpy() == value, k]) for k, c in enumerate(CHANNELS)}}
                for value in sorted(metadata[label].unique())
            }
        combined = (part * weights).sum(axis=2)
        report["initial_bias_diagnostic"][split] = {
            "weights": weights.tolist(), "lambda_init": config.CAT_LAMBDA_INIT,
            "cross_view_combined_consistency": describe(combined),
            "cross_view_bias": describe(config.CAT_LAMBDA_INIT * (2 * combined - 1)),
            "within_minus_cross_bias": describe(2 * config.CAT_LAMBDA_INIT * (1 - combined)),
            "note": "Initial parameters only; logit difference attributable to bias alone, not actual learned attention.",
        }
    log_path = Path(args.log) if args.log else cache_dir.parent / "logs" / "cache_full.log"
    counts, snippets = Counter(), []
    if log_path.is_file():
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for marker in ("OpenIE failed", "Failed sample", "Traceback", "Creating a new one with MEAN pooling"):
                    if marker in line:
                        counts[marker] += 1
                        if len(snippets) < 10:
                            snippets.append(line.strip()[:500])
    report["build_log"] = {"path": str(log_path), "exists": log_path.is_file(), "warning_line_counts": dict(counts),
                           "snippets": snippets, "note": "Counts log lines, not affected samples; absent warnings do not prove extraction success."}
    output = run_dir / ("cache_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    save_json(report, output / "cache_audit.json")
    save_json(issues, output / "file_issues.json")
    save_json(example_rows(valid, pairs), output / "validation_examples.json")
    lines = ["MVC cache audit (read-only)", json.dumps(report["integrity"], ensure_ascii=False), "",
             "Cross-view scores: mean / std / near-zero / near-0.5 / near-one fractions"]
    for split in ("train", "val"):
        lines.append(f"[{split}]")
        for channel in CHANNELS:
            values = report["profiles"][split][channel]["all_three_pairs"]
            if values["n"]:
                lines.append(channel + ": " + " / ".join(f"{values[k]:.4f}" for k in ("mean", "std", "near_zero_fraction", "near_half_fraction", "near_one_fraction")))
        lines.append("Default proxies: " + json.dumps(report["default_value_proxies"].get(split, {})))
    lines.extend(["", "Build log: " + json.dumps(report["build_log"], ensure_ascii=False), "", "Limitations:"] + report["limitations"])
    summary = "\n".join(lines)
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Saved to:", output.resolve())
    return report, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=cfg.OUTPUT_DIR, help="Original main run containing config.json")
    parser.add_argument("--log", default=None, help="Optional full cache build log")
    audit(parser.parse_args())
