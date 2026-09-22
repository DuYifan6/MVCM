"""Render an auditable illustrative-case figure from frozen JSON evidence."""

from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation

from ijmlc_style import COLORS, METHOD_COLORS, apply_style


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "representative_case"

QA_SCRIPTS = os.environ.get("NATURE_FIGURE_SCRIPTS")
if not QA_SCRIPTS:
    raise RuntimeError("Set NATURE_FIGURE_SCRIPTS before rendering the publication figure.")
sys.path.insert(0, QA_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# Explicit declarations retained for the static publication preflight.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def fixed_panel_label(ax, text: str) -> None:
    """Place panel letters at a fixed physical offset from the axes corner."""

    transform = ax.transAxes + ScaledTranslation(-22 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        text,
        transform=transform,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def load_and_validate() -> tuple[dict, dict]:
    selection = json.loads(
        (DATA / "representative_case_selection.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (DATA / "representative_case_evidence.json").read_text(encoding="utf-8")
    )
    selected = selection["primary_recommendation"]["selected"]

    assert evidence["purpose"] == "Read-only diagnostic extraction for an illustrative case"
    assert evidence["selection_is_post_hoc"] is True
    assert evidence["case_type"] == "primary"
    assert selected["sample_id"] == evidence["sample_id"] == 3894
    assert selected["group"] == evidence["group"] == "HF"
    assert selected["true_fake"] == evidence["true_fake"] == 1
    assert selected["true_ai"] == evidence["true_ai"] == 0
    assert evidence["active_channels_for_final_fusion"] == ["semantic", "logical", "local"]

    full = evidence["variants"]["three_channel_full"]
    no_mvcm = evidence["variants"]["without_mvcm"]
    assert full["channel_mask"] == [True, False, True, True]
    assert no_mvcm["channel_mask"] == [False, False, False, False]
    assert np.isclose(sum(full["masked_softmax_channel_weights"]), 1.0, atol=1e-6)
    assert full["masked_softmax_channel_weights"][1] == 0.0
    assert np.isclose(
        full["prob_fake"], selected["predictions"]["mvct"]["prob_fake"], atol=1e-5
    )
    assert np.isclose(
        no_mvcm["prob_fake"],
        selected["predictions"]["without_mvcm"]["prob_fake"],
        atol=1e-5,
    )
    return selection, evidence


def write_source_data(selection: dict, evidence: dict) -> None:
    selected = selection["primary_recommendation"]["selected"]
    full = evidence["variants"]["three_channel_full"]
    rows: list[dict[str, object]] = []

    for model_key, display in [
        ("roberta", "RoBERTa"),
        ("without_mvcm", "w/o MVCM"),
        ("mvct", "MVCT"),
    ]:
        pred = selected["predictions"][model_key]
        rows.append(
            {
                "section": "prediction",
                "item": display,
                "row": "",
                "column": "prob_fake",
                "value": pred["prob_fake"],
                "note": "pred_fake=" + str(pred["pred_fake"]),
            }
        )

    active_indices = {"semantic": 0, "logical": 2, "local": 3}
    for channel, index in active_indices.items():
        rows.append(
            {
                "section": "fusion_weight",
                "item": channel,
                "row": "",
                "column": "weight",
                "value": full["masked_softmax_channel_weights"][index],
                "note": "active three-channel model",
            }
        )

    matrices = {
        "semantic": evidence["mvc_channels"]["semantic"],
        "logical": evidence["mvc_channels"]["logical"],
        "local": evidence["mvc_channels"]["local"],
        "fused_signed_view_bias": full["fused_signed_view_bias"],
    }
    view_names = ["Title", "Description", "Body"]
    for matrix_name, matrix in matrices.items():
        for row_index, row_name in enumerate(view_names):
            for column_index, column_name in enumerate(view_names):
                rows.append(
                    {
                        "section": "matrix",
                        "item": matrix_name,
                        "row": row_name,
                        "column": column_name,
                        "value": matrix[row_index][column_index],
                        "note": "raw extracted value",
                    }
                )

    output = OUT / "Fig6_representative_case_source_data.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["section", "item", "row", "column", "value", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)


def add_matrix(ax, matrix, title: str, cmap, *, signed: bool = False) -> None:
    values = np.asarray(matrix, dtype=float)
    vmin, vmax = (-1.0, 1.0) if signed else (0.0, 1.0)
    ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    labels = ["T", "D", "B"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.tick_params(length=0, pad=1)
    ax.set_title(title, fontsize=6.7, pad=4, fontweight="bold")
    for row in range(3):
        for column in range(3):
            value = values[row, column]
            if signed:
                color = "white" if abs(value) >= 0.72 else COLORS["ink"]
            else:
                color = "white" if value >= 0.68 else COLORS["ink"]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_case_panel(ax, selection: dict, evidence: dict) -> None:
    selected = selection["primary_recommendation"]["selected"]
    ax.set_axis_off()
    ax.set_title("Selected HF test case", loc="left", pad=5)
    fixed_panel_label(ax, "a")

    ax.text(
        0.01,
        0.87,
        "Ground truth",
        transform=ax.transAxes,
        fontsize=6.4,
        color=COLORS["muted"],
        va="top",
    )
    ax.text(
        0.16,
        0.87,
        "Fake · Human-written",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["fake"],
        va="top",
    )
    ax.text(
        0.99,
        0.87,
        f"ID {evidence['sample_id']}  ·  {selected['word_count']} words",
        transform=ax.transAxes,
        fontsize=6.2,
        color=COLORS["muted"],
        ha="right",
        va="top",
    )

    ax.text(0.01, 0.68, "Title", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"])
    ax.text(
        0.11,
        0.68,
        textwrap.fill(evidence["text"]["title"], width=78),
        transform=ax.transAxes,
        fontsize=6.6,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.01, 0.49, "Description", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"]
    )
    ax.text(
        0.16,
        0.49,
        textwrap.fill(evidence["text"]["description"], width=72),
        transform=ax.transAxes,
        fontsize=6.4,
        va="center",
    )

    excerpt = evidence["text"]["body_excerpt"][:330].rstrip() + " …"
    ax.text(
        0.01,
        0.32,
        "Body excerpt",
        transform=ax.transAxes,
        fontsize=6.2,
        color=COLORS["muted"],
        va="top",
    )
    ax.text(
        0.16,
        0.32,
        textwrap.fill(excerpt, width=90),
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )
    ax.text(
        0.01,
        0.02,
        "Post-hoc illustrative selection from 61 strict candidates; internal matrices and attention were not used for selection.",
        transform=ax.transAxes,
        fontsize=5.7,
        color=COLORS["muted"],
        va="bottom",
    )


def draw_matrix_panel(ax, evidence: dict) -> list:
    ax.set_axis_off()
    ax.set_title("View relations and fused bias", loc="left", pad=5)
    fixed_panel_label(ax, "a")

    positions = [0.015, 0.265, 0.515, 0.765]
    heat_axes = [ax.inset_axes([x, 0.16, 0.205, 0.70]) for x in positions]
    cmaps = [
        LinearSegmentedColormap.from_list("semantic_scale", ["#F7FAFC", COLORS["semantic"]]),
        LinearSegmentedColormap.from_list("logical_scale", ["#FFF9F2", COLORS["logical"]]),
        LinearSegmentedColormap.from_list("local_scale", ["#FAF8FC", COLORS["local"]]),
        LinearSegmentedColormap.from_list(
            "signed_scale", [COLORS["mvct"], "#FFFFFF", COLORS["negative"]]
        ),
    ]
    full = evidence["variants"]["three_channel_full"]
    add_matrix(heat_axes[0], evidence["mvc_channels"]["semantic"], "Semantic", cmaps[0])
    add_matrix(heat_axes[1], evidence["mvc_channels"]["logical"], "Logical", cmaps[1])
    add_matrix(heat_axes[2], evidence["mvc_channels"]["local"], "Local", cmaps[2])
    add_matrix(
        heat_axes[3], full["fused_signed_view_bias"], "Fused bias", cmaps[3], signed=True
    )
    ax.text(
        0.5,
        0.02,
        "T, title; D, description; B, body. Channel matrices: 0–1; signed bias: −1 to 1.",
        transform=ax.transAxes,
        fontsize=5.7,
        color=COLORS["muted"],
        ha="center",
        va="bottom",
    )
    return heat_axes


def draw_weight_panel(ax, evidence: dict) -> None:
    fixed_panel_label(ax, "b")
    ax.set_title("Active-channel weights", loc="left", pad=5)
    full = evidence["variants"]["three_channel_full"]
    channels = ["Semantic", "Logical", "Local"]
    indices = [0, 2, 3]
    weights = [full["masked_softmax_channel_weights"][index] for index in indices]
    colors = [COLORS["semantic"], COLORS["logical"], COLORS["local"]]
    y = np.arange(3)
    ax.barh(y, weights, color=colors, height=0.55)
    ax.set_yticks(y, channels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("Masked-softmax fusion weight")
    for row, value in enumerate(weights):
        ax.text(
            value / 2,
            row,
            f"{value * 100:.1f}%",
            va="center",
            ha="center",
            fontsize=6.3,
            color="white",
            fontweight="bold",
        )


def draw_prediction_panel(ax, selection: dict) -> None:
    fixed_panel_label(ax, "c")
    ax.set_title("Model predictions for the selected case", loc="left", pad=5)
    selected = selection["primary_recommendation"]["selected"]
    specs = [
        ("RoBERTa", "roberta"),
        ("w/o MVCM", "without_mvcm"),
        ("MVCT", "mvct"),
    ]
    probabilities = [selected["predictions"][key]["prob_fake"] for _, key in specs]
    predictions = [selected["predictions"][key]["pred_fake"] for _, key in specs]
    colors = [METHOD_COLORS[label] for label, _ in specs]
    y = np.arange(len(specs))
    ax.barh(y, probabilities, color=colors, height=0.52)
    ax.axvline(0.55, color=COLORS["muted"], linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(y, [label for label, _ in specs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Predicted probability of Fake")
    ax.set_xticks([0.0, 0.5, 1.0])
    for row, (value, prediction) in enumerate(zip(probabilities, predictions)):
        label = "Fake" if prediction == 1 else "Real"
        if prediction == 0:
            label_color = "white" if row == 0 else COLORS["ink"]
            ax.text(
                value - 0.018,
                row,
                f"{value:.3f} · {label}",
                va="center",
                ha="right",
                fontsize=6.0,
                color=label_color,
                fontweight="bold",
            )
        else:
            ax.text(value + 0.025, row, f"{value:.3f} · {label}", va="center", fontsize=6.2)
    ax.text(
        0.99,
        0.02,
        "True label: Fake",
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["fake"],
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def export_pub(fig, output_base: Path, *, axes: list, panel_ids: dict, exclude_axes: list) -> None:
    """Run final-geometry alignment QA and export the publication bundle."""

    qa_dir = output_base.parent / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=qa_dir / f"{output_base.name}.alignment.json",
        overlay_svg=qa_dir / f"{output_base.name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        axes=axes,
        panel_ids=panel_ids,
        exclude_axes=exclude_axes,
    )
    fig.savefig(output_base.with_suffix(".svg"), format="svg")
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_base.with_suffix(".png"), format="png", dpi=600)
    fig.savefig(output_base.with_suffix(".tiff"), format="tiff", dpi=600)
    plt.close(fig)


def main() -> None:
    apply_style()
    selection, evidence = load_and_validate()
    write_source_data(selection, evidence)

    fig = plt.figure(figsize=(7.09, 103 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.08, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.13,
        top=0.925,
        wspace=0.34,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    heat_axes = draw_matrix_panel(ax_a, evidence)
    draw_weight_panel(ax_b, evidence)
    draw_prediction_panel(ax_c, selection)

    output_base = OUT / "Fig6_representative_case"
    export_pub(
        fig,
        output_base,
        axes=[ax_a, ax_b, ax_c],
        panel_ids={ax_a: "a", ax_b: "b", ax_c: "c"},
        exclude_axes=heat_axes,
    )


if __name__ == "__main__":
    main()
