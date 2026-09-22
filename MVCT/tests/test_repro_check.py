import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from runtime_guard import RuntimeLock
from run_repro_check import fingerprint, first_difference, fixture_inputs, trace_training
from model_cat import MVCTTokenCATModel
from run_repeated_validation import make_loader
from utils import set_seed
from types import SimpleNamespace


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(16, 8)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ReproCheckTests(unittest.TestCase):
    def test_tensor_hash_covers_dtype_shape_values_and_noncontiguous(self):
        x = torch.arange(4).reshape(2, 2)
        self.assertNotEqual(fingerprint(x), fingerprint(x.reshape(4)))
        self.assertNotEqual(fingerprint(x), fingerprint(x.float()))
        self.assertEqual(fingerprint(x.T), fingerprint(x.T.contiguous()))
        self.assertEqual(fingerprint(torch.tensor(3.)), fingerprint(torch.tensor(3.)))
        self.assertEqual(fingerprint({"x": x, "a": 2}), fingerprint({"a": 2, "x": x.clone()}))

    def test_lock_excludes_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.lock"
            command = [sys.executable, "-c",
                       "from runtime_guard import RuntimeLock; import sys; "
                       "lock=RuntimeLock(sys.argv[1]); lock.__enter__(); lock.__exit__()", str(path)]
            with RuntimeLock(path):
                blocked = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("Another guarded job", blocked.stderr)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertTrue(path.exists())

    def test_comparator_reports_earliest_difference(self):
        a = {"inventory": {}, "config": {}, "environment": {}, "initial_weights": "a",
             "initial_rng": "b", "initial_optimizer": "c", "initial_parameter_hashes": {"head": "a"},
             "trace": [{"phase": "train_1", "batch": 1, "input_hash": "x", "weights_after": "y"}],
             "phases": [{"phase": "train_1", "weights": "y"}]}
        self.assertIsNone(first_difference(a, copy.deepcopy(a)))
        b = copy.deepcopy(a)
        b["trace"][0]["weights_after"] = "z"
        self.assertEqual(first_difference(a, b)["stage"], "weights_after")
        b["trace"][0]["input_hash"] = "other"
        self.assertEqual(first_difference(a, b)["stage"], "input_hash")
        b["initial_weights"] = "other"
        b["initial_parameter_hashes"]["head"] = "other"
        self.assertEqual(first_difference(a, b)["first_parameter"], "head")

    def test_real_shared_train_and_both_validators_match_on_cpu_fixture(self):
        outputs = []
        for entry in ("old", "old", "new"):
            cfg, train, val, paths, inventory = fixture_inputs()
            set_seed(42)
            train_loader = make_loader(train, cfg, 42, True)
            val_loader = make_loader(val, cfg, 42, False)
            model = MVCTTokenCATModel(encoder=Encoder(), cat_layers=1, cat_heads=2,
                                     cat_ff_multiplier=2, dropout_p=.1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            result = trace_training(model, optimizer, train_loader, val_loader, cfg, 2, 1, entry, paths)
            self.assertEqual(len(result["trace"]), 5)
            self.assertEqual(len(result["phases"]), 3)
            outputs.append(result)
        self.assertIsNone(first_difference(outputs[0], outputs[1]))
        self.assertIsNone(first_difference(outputs[0], outputs[2]))


if __name__ == "__main__":
    unittest.main()
