import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from run_repeated_validation import make_loader, summarize, validate_completion


class RepeatedValidationTests(unittest.TestCase):
    def test_pair_order_is_independent_of_global_rng_consumption(self):
        config = SimpleNamespace(TRAIN_NUM_WORKERS=0, VAL_NUM_WORKERS=0,
                                 BATCH_SIZE=4, PIN_MEMORY=False)
        data = TensorDataset(torch.arange(24))
        first = make_loader(data, config, 42, True)
        torch.rand(1000)
        second = make_loader(data, config, 42, True)
        for _ in range(2):
            order_a = torch.cat([b[0] for b in first])
            torch.rand(71)
            order_b = torch.cat([b[0] for b in second])
            self.assertTrue(torch.equal(order_a, order_b))
        different = make_loader(data, config, 43, True)
        self.assertFalse(torch.equal(order_a, torch.cat([b[0] for b in different])))

    def test_summary_uses_sample_sd_and_only_matched_seeds(self):
        def row(variant, seed, score):
            return {"variant": variant, "seed": seed, "fake_accuracy": score,
                    "fake_macro_f1": score, "ai_accuracy": score,
                    "ai_macro_f1": score, "val_loss": 1 - score}
        report = summarize([row("without_mvcm", 42, .8), row("without_mvcm", 43, .9),
                            row("missing_evidence", 42, .85), row("missing_evidence", 44, .7)])
        baseline = report["by_variant"]["without_mvcm"]["fake_macro_f1"]
        self.assertAlmostEqual(baseline["mean"], .85)
        self.assertAlmostEqual(baseline["sd"], .1 / 2**.5)
        paired = report["paired_vs_without_mvcm"]["missing_evidence"]
        self.assertEqual(paired["n_pairs"], 1)
        self.assertAlmostEqual(paired["fake_macro_f1_mean_difference"], .05)
        self.assertEqual(paired["fake_macro_f1_positive_pairs"], 1)
        self.assertIsNone(report["by_variant"]["full"]["fake_macro_f1"]["mean"])

    def test_resume_rejects_wrong_protocol_or_nonreproduced_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best_checkpoint.pt").write_bytes(b"fixture")
            (root / "validation_metrics.json").write_text("{}", encoding="utf-8")
            done = root / "completed.json"
            record = {"protocol_id": "a", "variant": "full", "seed": 42,
                      "loss_reproduced": True, "run_dir": str(root), "diagnostic_dir": str(root)}
            done.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(validate_completion(done, "a", "full", 42), record)
            with self.assertRaises(ValueError):
                validate_completion(done, "b", "full", 42)
            record["loss_reproduced"] = False
            done.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_completion(done, "a", "full", 42)


if __name__ == "__main__":
    unittest.main()
