import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset_cat import MVCNewsDatasetCAT, _compute_single_mvc, _group_stratified_split
from consistency_hybrid import ConsistencyBuilderHybrid
from model_cat import MVCTTokenCATModel, attention_kl_regularizer


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        max_length = kwargs["max_length"]
        tokens = [10 + index for index, _ in enumerate(str(text).split())]
        return {"input_ids": tokens[:max_length]}


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyBuilder:
    cache_signature = "dummy-v1"

    def __init__(self):
        self.calls = 0

    def build_mvc(self, title, text, desc):
        self.calls += 1
        mvc = np.ones((3, 3, 4), dtype=np.float32)
        return mvc, mvc.mean(axis=-1)


class CATCoreTests(unittest.TestCase):
    def test_missing_evidence_masks_only_joint_proxy_and_preserves_cache(self):
        original = MVCTTokenCATModel(encoder=DummyEncoder(), cat_heads=2)
        masked = MVCTTokenCATModel(encoder=DummyEncoder(), cat_heads=2, mask_missing_evidence=True)
        masked.load_state_dict(original.state_dict(), strict=True)
        mvc = torch.full((1, 3, 3, 4), .8)
        mvc[0, 0, 1] = torch.tensor([.8, 0, .5, .6])
        mvc[0, 1, 0] = mvc[0, 0, 1]
        mvc[0, 0, 2] = torch.tensor([.8, 0, .8, .6])
        mvc[0, 2, 0] = torch.tensor([.8, .7, .5, .6])
        mvc[:, torch.arange(3), torch.arange(3)] = 1
        before = mvc.clone()
        old_bias, new_bias = original.combine_mvc(mvc), masked.combine_mvc(mvc)
        self.assertAlmostEqual(float(old_bias[0, 0, 1]), .02, places=6)
        self.assertAlmostEqual(float(new_bias[0, 0, 1]), (.4 * .6 + .15 * .2) / .55, places=6)
        self.assertEqual(float(new_bias[0, 0, 2]), float(old_bias[0, 0, 2]))
        self.assertEqual(float(new_bias[0, 2, 0]), float(old_bias[0, 2, 0]))
        self.assertTrue(torch.equal(new_bias.diagonal(dim1=1, dim2=2), old_bias.diagonal(dim1=1, dim2=2)))
        self.assertTrue(torch.equal(mvc, before))
        new_bias.sum().backward()
        self.assertTrue(torch.isfinite(masked.channel_logits.grad).all())

    def test_missing_evidence_no_available_channels_is_neutral(self):
        model = MVCTTokenCATModel(encoder=DummyEncoder(), cat_heads=2,
                                 mvc_weights=(0, .5, .5, 0), mask_missing_evidence=True)
        mvc = torch.zeros((1, 3, 3, 4))
        mvc[..., 2] = .5
        bias = model.combine_mvc(mvc)
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))
        bias.sum().backward()
        self.assertTrue(torch.isfinite(model.channel_logits.grad).all())

    def test_no_proxy_forward_and_old_checkpoint_compatibility(self):
        original = MVCTTokenCATModel(encoder=DummyEncoder(), cat_heads=2, cat_layers=1)
        masked = MVCTTokenCATModel(encoder=DummyEncoder(), cat_heads=2, cat_layers=1, mask_missing_evidence=True)
        masked.load_state_dict(original.state_dict(), strict=True)
        original.eval()
        masked.eval()
        sequence = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.ones((1, 4), dtype=torch.long),
                    "view_ids": torch.tensor([[-1, 0, 1, 2]])}
        mvc = torch.full((1, 3, 3, 4), .7)
        for old, new in zip(original(sequence, mvc), masked(sequence, mvc)):
            self.assertTrue(torch.equal(old, new))

    def test_dataset_builds_concatenated_sequence_and_view_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            mvc_path = Path(directory) / "mvc.npy"
            np.save(mvc_path, np.ones((3, 3, 4), dtype=np.float32))
            frame = pd.DataFrame(
                [
                    {
                        "ID": 7,
                        "title": "one two three",
                        "text": "four five six seven",
                        "desc": "eight nine",
                        "label_fake": 1,
                        "label_ai": 0,
                        "mvc_path": str(mvc_path),
                    }
                ]
            )
            dataset = MVCNewsDatasetCAT(
                frame,
                DummyTokenizer(),
                directory,
                max_len_title=2,
                max_len_text=3,
                max_len_desc=2,
            )
            sample = dataset[0]
            self.assertEqual(sample["sequence"]["input_ids"].shape[0], 11)
            self.assertEqual(
                sample["sequence"]["view_ids"].tolist(),
                [-1, 0, 0, -1, 1, 1, 1, -1, 2, 2, -1],
            )

    def test_token_bias_and_forward_shapes(self):
        model = MVCTTokenCATModel(
            mvc_weights=(0.4, 0.25, 0.2, 0.15),
            cat_layers=2,
            cat_heads=2,
            cat_ff_multiplier=2,
            encoder=DummyEncoder(),
        )
        sequence = {
            "input_ids": torch.randint(1, 100, (2, 8)),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
            ),
            "view_ids": torch.tensor(
                [[-1, 0, 0, -1, 1, 1, 2, -1], [-1, 0, 0, -1, 1, 1, 2, 2]]
            ),
        }
        mvc = torch.full((2, 3, 3, 4), 0.25)
        for channel in range(4):
            mvc[:, :, :, channel].diagonal(dim1=1, dim2=2).fill_(1.0)
        bias = model.build_token_bias(
            mvc, sequence["view_ids"], sequence["attention_mask"]
        )
        self.assertTrue(torch.allclose(bias[:, 0], torch.zeros_like(bias[:, 0])))
        self.assertAlmostEqual(float(bias[0, 1, 4]), -0.5, places=5)
        y_fake, y_ai, attention = model(sequence, mvc)
        self.assertEqual(y_fake.shape, (2, 2))
        self.assertEqual(y_ai.shape, (2, 2))
        self.assertEqual(attention.shape, (2, 2, 8, 8))
        valid_row_sums = attention.sum(dim=-1)
        self.assertTrue(torch.allclose(valid_row_sums, torch.ones_like(valid_row_sums)))
        regularizer = attention_kl_regularizer(
            attention, sequence["attention_mask"], sequence["view_ids"]
        )
        self.assertTrue(torch.isfinite(regularizer))
        self.assertGreaterEqual(float(regularizer), 0.0)

    def test_attention_regularizer_penalizes_peaked_attention(self):
        mask = torch.ones((1, 3), dtype=torch.long)
        views = torch.tensor([[0, 1, 2]])
        uniform = torch.full((1, 1, 3, 3), 1.0 / 3.0)
        peaked = torch.eye(3).view(1, 1, 3, 3)
        uniform_loss = attention_kl_regularizer(uniform, mask, views)
        peaked_loss = attention_kl_regularizer(peaked, mask, views)
        self.assertLess(float(uniform_loss), 1e-6)
        self.assertGreater(float(peaked_loss), float(uniform_loss))

    def test_cache_is_content_addressed_and_resumable(self):
        row = SimpleNamespace(ID=1, title="t", text="b", desc="d")
        builder = DummyBuilder()
        with tempfile.TemporaryDirectory() as directory:
            first = _compute_single_mvc(row, builder, directory)
            second = _compute_single_mvc(row, builder, directory)
            self.assertEqual(first, second)
            self.assertEqual(builder.calls, 1)
            self.assertTrue(Path(first).exists())

    def test_group_split_keeps_exact_duplicates_together(self):
        rows = []
        for label_fake in (0, 1):
            for label_ai in (0, 1):
                for group_index in range(10):
                    group = f"{label_fake}-{label_ai}-{group_index}"
                    rows.append(
                        {
                            "ID": len(rows),
                            "label_fake": label_fake,
                            "label_ai": label_ai,
                            "_split_group": group,
                        }
                    )
                    if group_index == 0:
                        rows.append(
                            {
                                "ID": len(rows),
                                "label_fake": label_fake,
                                "label_ai": label_ai,
                                "_split_group": group,
                            }
                        )
        frame = pd.DataFrame(rows)
        train, val, test = _group_stratified_split(frame, 0.1, 0.1, 42)
        group_sets = [set(part["_split_group"]) for part in (train, val, test)]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

    def test_structured_logic_conflicts(self):
        self.assertEqual(
            ConsistencyBuilderHybrid._temporal_conflict(
                "The event happened in 2020", "The event happened in 2021"
            ),
            1.0,
        )
        self.assertGreater(
            ConsistencyBuilderHybrid._numeric_conflict(
                "Revenue was 10 million", "Revenue was 10 billion"
            ),
            0.9,
        )
        self.assertEqual(
            ConsistencyBuilderHybrid._negation_conflict(
                "The minister resigned", "The minister did not resign"
            ),
            1.0,
        )
        builder = ConsistencyBuilderHybrid.__new__(ConsistencyBuilderHybrid)
        builder.alignment_threshold = 0.55
        builder.max_logic_pairs = 12
        builder.coverage_target = 5
        builder.logic_weights = np.array([0.55, 0.15, 0.15, 0.15])
        builder.use_nli = False
        left = [{"sentence": "The minister resigned"}]
        right = [{"sentence": "The minister did not resign"}]
        score = builder._logic_score(left, right, np.array([[0.95]]))
        self.assertLess(score, 0.1)


if __name__ == "__main__":
    unittest.main()
