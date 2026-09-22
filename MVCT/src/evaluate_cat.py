from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from utils import compute_metrics_twoheads, save_json


def _off_diagonal_values(matrix):
    mask = ~torch.eye(3, device=matrix.device, dtype=torch.bool)
    return matrix[:, mask]


def _correlation(left, right, function):
    if left.size < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan"), float("nan")
    coefficient, p_value = function(left, right)
    return float(coefficient), float(p_value)


def evaluate_cat(model, dataloader, device, output_path=None, fake_threshold=0.5):
    model.eval()
    preds_fake, trues_fake = [], []
    preds_ai, trues_ai = [], []
    attention_pairs, mvc_pairs = [], []
    prediction_rows = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            sequence = {
                key: value.to(device) for key, value in batch["sequence"].items()
            }
            mvc = batch["mvc"].float().to(device)
            labels_fake = batch["label_fake"].to(device)
            labels_ai = batch["label_ai"].to(device)
            y_fake, y_ai, attention = model(sequence, mvc)

            view_attention = model.aggregate_attention_by_view(
                attention, sequence["view_ids"], sequence["attention_mask"]
            )
            channels = mvc.size(-1)
            weights = model.mvc_weights[:channels]
            raw_combined_mvc = (mvc * weights.view(1, 1, 1, channels)).sum(dim=-1)
            attention_pairs.append(_off_diagonal_values(view_attention).cpu())
            mvc_pairs.append(_off_diagonal_values(raw_combined_mvc).cpu())

            fake_probabilities = torch.softmax(y_fake, dim=1)
            ai_probabilities = torch.softmax(y_ai, dim=1)
            fake_predictions = (fake_probabilities[:, 1] > fake_threshold).long()
            ai_predictions = ai_probabilities.argmax(dim=1)
            preds_fake.extend(fake_predictions.cpu().tolist())
            preds_ai.extend(ai_predictions.cpu().tolist())
            trues_fake.extend(labels_fake.cpu().tolist())
            trues_ai.extend(labels_ai.cpu().tolist())

            sample_ids = batch["sample_id"].cpu().tolist()
            for index, sample_id in enumerate(sample_ids):
                prediction_rows.append(
                    {
                        "sample_id": int(sample_id),
                        "true_fake": int(labels_fake[index].item()),
                        "pred_fake": int(fake_predictions[index].item()),
                        "prob_fake": float(fake_probabilities[index, 1].item()),
                        "true_ai": int(labels_ai[index].item()),
                        "pred_ai": int(ai_predictions[index].item()),
                        "prob_ai": float(ai_probabilities[index, 1].item()),
                    }
                )

    attention_values = torch.cat(attention_pairs).numpy().reshape(-1)
    mvc_values = torch.cat(mvc_pairs).numpy().reshape(-1)
    spearman_r, spearman_p = _correlation(
        attention_values, mvc_values, spearmanr
    )
    pearson_r, pearson_p = _correlation(attention_values, mvc_values, pearsonr)
    metrics = compute_metrics_twoheads(preds_fake, trues_fake, preds_ai, trues_ai)
    metrics["consistency_attention_correlation"] = {
        "unit": "sample_view_pair",
        "n": int(attention_values.size),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }
    metrics["learned_parameters"] = {
        "mvc_channel_weights": model.mvc_weights.detach().cpu().tolist(),
        "cat_lambda": [
            float(layer.attention.lambda_bias.detach().cpu().item())
            for layer in model.cat_layers
        ],
    }

    if output_path:
        save_json(metrics, output_path)
        prediction_path = Path(output_path).with_name("test_predictions.json")
        save_json(prediction_rows, prediction_path)
    return metrics
