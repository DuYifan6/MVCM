import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from audit_mvc_cache import audit, inspect_tensor, sha256
from config import cfg
from dataset_cat import _cache_key, _clean_and_group_dataframe
from diagnose_validation import signature_from_config


class CacheAuditTests(unittest.TestCase):
    def test_detects_corruption_without_sanitizing(self):
        array = np.ones((3, 3, 4), dtype=np.float32)
        array[0, 1, 0] = -.2
        self.assertIn("outside_0_1", inspect_tensor(array))
        self.assertIn("asymmetric", inspect_tensor(array))
        array[0, 1, 0] = np.nan
        self.assertEqual(inspect_tensor(array), ["non_finite"])
        self.assertEqual(inspect_tensor(np.ones((3, 3, 3))), ["wrong_shape"])

    def test_end_to_end_read_only_and_test_labels_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, cache, splits = root / "run", root / "cache", root / "splits"
            for path in (run, cache, splits):
                path.mkdir()
            data = root / "data.csv"
            pd.DataFrame([
                {"ID": i, "title": f"Title {i}", "text": f"Body content {i}",
                 "desc": f"Description {i}", "label_fake": i % 2, "label_ai": (i + 1) % 2}
                for i in range(3)
            ]).to_csv(data, index=False)
            config = replace(cfg, DATA_PATH=str(data), MVC_CACHE_DIR=str(cache),
                             SPLIT_DIR=str(splits), OUTPUT_DIR=str(run), USE_NLI=True)
            (run / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
            signature = signature_from_config(config)
            (cache / "cache_manifest.json").write_text(json.dumps({"builder_signature": signature}), encoding="utf-8")
            frame = _clean_and_group_dataframe(data)
            for i, split in enumerate(("train", "val", "test")):
                frame.iloc[[i]][["ID", "label_fake", "label_ai", "_split_group"]].to_csv(splits / f"{split}.csv", index=False)
                array = np.zeros((3, 3, 4), dtype=np.float32)
                array[:, :, 0] = .8
                array[:, :, 2] = .5
                array[:, :, 3] = .7
                array[np.arange(3), np.arange(3)] = 1
                if split == "test":
                    array[0, 1, 0] = np.nan
                row = next(frame.iloc[[i]].itertuples(index=False))
                np.save(cache / f"mvc_{_cache_key(row, SimpleNamespace(cache_signature=signature))}.npy", array)
            np.save(cache / "mvc_old_version.npy", np.ones((3, 3, 4)))
            before = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
            report, output = audit(SimpleNamespace(run_dir=str(run), log=None))
            self.assertEqual(report["integrity"]["invalid_or_missing_samples"], 1)
            self.assertEqual(report["integrity"]["unreferenced_npy_files"], 1)
            self.assertEqual(set(report["profiles"]), {"train", "val"})
            self.assertEqual(report["default_value_proxies"]["val"]["samples_with_all_three_proxy_pairs_fraction"], 1)
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest)
            examples = json.loads((output / "validation_examples.json").read_text(encoding="utf-8"))
            self.assertTrue(examples)
            self.assertTrue(all(row["split"] == "val" for row in examples))


if __name__ == "__main__":
    unittest.main()
