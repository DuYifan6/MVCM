from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baseline_models import BaselineNewsDataset, DualHeadTextCNN, DualHeadTransformer
import run_strict_baselines_seed42 as baselines


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 32

    def __call__(self, text, **kwargs):
        limit = kwargs["max_length"]
        return {"input_ids": [3] * min(len(str(text)), limit)}


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def sample_frame():
    return pd.DataFrame([
        {"ID": 1, "title": "abc", "text": "body", "desc": "de",
         "label_fake": 0, "label_ai": 1},
    ])


def result_row(name, value=.8):
    return {
        "variant": name, "seed": 42, "checkpoint_epoch": 2,
        "val_loss": .2, "fake_threshold": .55,
        "fake_accuracy": value, "fake_macro_f1": value - .01,
        "ai_accuracy": value + .02, "ai_macro_f1": value + .01,
        "output_dir": f"runs/{name}",
    }


def test_predeclared_suite_has_four_non_mvcm_baselines():
    assert baselines.VARIANTS == ("tfidf_logreg", "textcnn", "bert", "roberta")
    assert baselines.NEURAL_VARIANTS == ("textcnn", "bert", "roberta")
    assert baselines.MODEL_SPECS["tfidf_logreg"]["C"] == 1.0
    assert baselines.FAKE_THRESHOLD == .55


def test_baseline_dataset_preserves_three_view_token_budget():
    dataset = BaselineNewsDataset(sample_frame(), TinyTokenizer(), 2, 3, 1)
    row = dataset[0]
    assert row["input_ids"].shape == (10,)  # CLS + three SEP + 2+3+1
    assert row["attention_mask"].sum().item() == 10
    assert row["label_fake"].item() == 0
    assert row["label_ai"].item() == 1


def test_two_neural_baselines_produce_two_binary_heads():
    inputs = torch.randint(0, 32, (2, 12))
    mask = torch.ones_like(inputs)
    transformer = DualHeadTransformer("unused", encoder=TinyEncoder())
    cnn = DualHeadTextCNN(32, 0, embedding_dim=8, filters=4, kernels=(2, 3))
    for model in (transformer, cnn):
        fake, ai = model(inputs, mask)
        assert fake.shape == (2, 2)
        assert ai.shape == (2, 2)


def test_fixed_threshold_metrics_do_not_scan_test_probabilities():
    metrics, rows = baselines.predictions_and_metrics(
        [1, 2], [0, 1], [0, 1], [.55, .551], [.49, .51]
    )
    assert [row["pred_fake"] for row in rows] == [0, 1]
    assert [row["pred_ai"] for row in rows] == [0, 1]
    assert metrics["fake_report"]["accuracy"] == 1.0


def test_probability_loss_is_finite_and_rewards_correct_predictions():
    good = baselines.combined_probability_loss(
        [0, 1], [0, 1], [.05, .95], [.05, .95]
    )
    bad = baselines.combined_probability_loss(
        [0, 1], [0, 1], [.95, .05], [.95, .05]
    )
    assert np.isfinite(good) and np.isfinite(bad)
    assert good < bad


def test_summary_is_partial_safe_and_rejects_duplicates():
    rows = [result_row("tfidf_logreg"), result_row("bert", .9)]
    report = baselines.summarize(rows, "validation_frozen")
    assert report["completed"] == 2
    assert report["by_variant"]["textcnn"] is None
    assert report["by_variant"]["bert"]["fake_accuracy"] == .9
    with pytest.raises(ValueError, match="Duplicate"):
        baselines.summarize(rows + [rows[0]], "validation_frozen")
