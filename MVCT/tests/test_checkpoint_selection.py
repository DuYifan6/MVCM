import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_checkpoint_selection import SelectionTracker, evaluate_selection_epoch, train_model_cat
from train_cat import FocalLossCE, evaluate_epoch
from run_checkpoint_selection import main, summarize, verify_baseline, verify_selector_diagnostic, validate_completion


class FixedModel(torch.nn.Module):
    def forward(self, sequence, mvc):
        return sequence["fake"], sequence["ai"], None


class CheckpointSelectionTests(unittest.TestCase):
    def test_f1_does_not_reset_loss_early_stopping_and_ties_keep_earlier(self):
        tracker = SelectionTracker(3)
        flags = [tracker.update(i + 1, loss, score) for i, (loss, score) in enumerate(
            [(.1, .7), (.2, .8), (.3, .8), (.4, .9)])]
        self.assertEqual(flags[-1], (False, True, True))
        self.assertEqual(tracker.loss_epoch, 1)
        self.assertEqual(tracker.f1_epoch, 4)
        tracker = SelectionTracker(3)
        tracker.update(1, .2, .8)
        tracker.update(2, .1, .8)
        self.assertEqual(tracker.f1_epoch, 1)
        self.assertEqual(tracker.loss_epoch, 2)
        with self.assertRaises(ValueError):
            tracker.update(3, float("nan"), .9)

    def test_validation_matches_old_loss_and_strict_threshold(self):
        batch = {"sequence": {"fake": torch.tensor([[0., 0.], [0., 2.], [2., 0.]]),
                              "ai": torch.tensor([[0., 0.], [0., 2.], [2., 0.]])},
                 "mvc": torch.zeros(3, 3, 3, 4),
                 "label_fake": torch.tensor([0, 1, 0]),
                 "label_ai": torch.tensor([0, 1, 0])}
        old_loss = evaluate_epoch(FixedModel(), [batch], "cpu", FocalLossCE())
        state = torch.get_rng_state().clone()
        result = evaluate_selection_epoch(FixedModel(), [batch], "cpu", FocalLossCE(), .5)
        self.assertEqual(result["val_loss"], old_loss)
        self.assertEqual(result["fake_macro_f1"], 1.)
        self.assertEqual(result["ai_macro_f1"], 1.)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            evaluate_selection_epoch(FixedModel(), [], "cpu", FocalLossCE(), .55)

    def test_training_saves_two_selectors_same_trajectory(self):
        config = SimpleNamespace(DEVICE="cpu", FAKE_THRESHOLD=.55, EPOCHS=10,
                                 EARLYSTOP_PATIENCE=3, FAKE_CLASS_WEIGHT=None)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        counter = [0]

        def fake_train(*args):
            counter[0] += 1
            with torch.no_grad():
                model.weight.fill_(counter[0])
            return {"loss": .3, "task_loss": .29, "attention_kl": .01}

        validations = [{"val_loss": loss, "fake_macro_f1": score, "ai_macro_f1": .9}
                       for loss, score in [(.1, .7), (.2, .8), (.3, .8), (.4, .75)]]
        with tempfile.TemporaryDirectory() as directory:
            with patch("train_checkpoint_selection.train_one_epoch", side_effect=fake_train), \
                 patch("train_checkpoint_selection.evaluate_selection_epoch", side_effect=validations):
                _, history = train_model_cat(model, None, None, config, optimizer, directory)
            f1 = torch.load(str(Path(directory) / "best_checkpoint.pt"), map_location="cpu")
            loss = torch.load(str(Path(directory) / "loss_selected" / "best_checkpoint.pt"), map_location="cpu")
            self.assertEqual(counter[0], 4)
            self.assertEqual(len(history["val_loss"]), 4)
            self.assertEqual(f1["epoch"], 2)
            self.assertEqual(loss["epoch"], 1)
            self.assertEqual(loss["config"]["CHECKPOINT_SELECTION"], "val_loss")
            self.assertEqual(f1["val_loss"], .2)
            self.assertEqual(f1["best_val_loss"], .1)
            self.assertEqual(f1["selection_score"], .8)
            self.assertEqual(f1["model_state_dict"]["weight"].item(), 2.)
            self.assertEqual(loss["model_state_dict"]["weight"].item(), 1.)
            self.assertEqual(model.weight.item(), 2.)
            self.assertFalse((Path(directory) / "last_checkpoint.pt").exists())
            report = {"checkpoint_epoch": 2, "reference_threshold": .55, "loss_reproduced": True,
                      "at_reference_threshold": {"fake_report": {"macro avg": {"f1-score": .8}}}}
            self.assertTrue(verify_selector_diagnostic(directory, report, "val_fake_macro_f1"))
            report["at_reference_threshold"]["fake_report"]["macro avg"]["f1-score"] = .81
            with self.assertRaises(ValueError):
                verify_selector_diagnostic(directory, report, "val_fake_macro_f1")
            with self.assertRaises(FileExistsError):
                train_model_cat(model, None, None, config, optimizer, directory)

    def test_baseline_protocol_and_nine_completions_required(self):
        protocol = {"versions": {"torch": "fixture"}, "source_sha256": {"model_cat.py": "same"}}
        ident = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"protocol_id": ident, **protocol}), encoding="utf-8")
            with patch("run_repeated_validation.validate_completion") as check:
                self.assertEqual(verify_baseline(protocol, root), ident)
                self.assertEqual(check.call_count, 9)
                with self.assertRaisesRegex(ValueError, "versions"):
                    verify_baseline({**protocol, "versions": {"torch": "changed"}}, root)
            (root / "protocol.json").write_text(json.dumps({"protocol_id": "tampered", **protocol}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_baseline(protocol, root)

    def test_summary_exposes_same_run_comparison(self):
        row = {"variant": "full", "seed": 42, "checkpoint_epoch": 2, "loss_selected_epoch": 1,
               "selection_fake_f1_change": .02, "selection_ai_f1_change": -.01,
               "fake_accuracy": .9, "fake_macro_f1": .88, "ai_accuracy": .95,
               "ai_macro_f1": .94, "val_loss": .15}
        report = summarize([row])
        self.assertEqual(report["planned_runs"], 9)
        self.assertEqual(report["within_run_selection"]["full"]["mean_fake_f1_change"], .02)
        self.assertEqual(report["within_run_selection"]["full"]["different_epoch_count"], 1)
        self.assertIsNone(report["within_run_selection"]["missing_evidence"]["mean_fake_f1_change"])

    def test_completion_needs_both_selector_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best_checkpoint.pt").touch()
            (root / "validation_metrics.json").write_text("{}", encoding="utf-8")
            record = {"protocol_id": "a", "variant": "full", "seed": 42,
                      "loss_reproduced": True, "run_dir": str(root), "diagnostic_dir": str(root)}
            done = root / "completed.json"
            done.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_completion(done, "a", "full", 42)
            (root / "loss_selected").mkdir()
            (root / "loss_selected" / "best_checkpoint.pt").touch()
            record.update(selection_score_reproduced=True, loss_selector_reproduced=True,
                          loss_diagnostic_dir=str(root))
            done.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(validate_completion(done, "a", "full", 42), record)

    def test_preflight_is_read_only_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "cache_manifest.json").write_text('{"builder_signature": "fixture"}', encoding="utf-8")
            config = {"MASK_MISSING_EVIDENCE": False, "FAKE_CLASS_WEIGHT": None,
                      "MVC_CACHE_DIR": str(cache), "DATA_PATH": "fixture.csv",
                      "SPLIT_DIR": "splits", "FAKE_THRESHOLD": .55}
            checkpoint = {"config": config, "model_state_dict": {"channel_mask": torch.ones(4)}}
            output = root / "new_suite"
            with patch("run_checkpoint_selection.torch.cuda.is_available", return_value=True), \
                 patch("run_checkpoint_selection.torch.load", return_value=checkpoint), \
                 patch("run_checkpoint_selection.signature_from_config", return_value="fixture"), \
                 patch("run_checkpoint_selection.file_hash", return_value="fixture"), \
                 patch("run_checkpoint_selection.verify_baseline", return_value="old-protocol") as verify, \
                 patch("run_checkpoint_selection.train_model_cat") as train, \
                 patch("run_checkpoint_selection.diagnose_validation") as diagnose:
                main(["--reference-run", str(root / "reference"),
                      "--suite-dir", str(output), "--baseline-suite", str(root / "old_suite"),
                      "--check-only"])
                verify.assert_called_once()
                train.assert_not_called()
                diagnose.assert_not_called()
                self.assertFalse(output.exists())
                with self.assertRaisesRegex(ValueError, "separate"):
                    main(["--suite-dir", str(root / "old_suite"),
                          "--baseline-suite", str(root / "old_suite"), "--check-only"])


if __name__ == "__main__":
    unittest.main()
