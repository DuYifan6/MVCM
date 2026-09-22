"""Render Fig. 2: conversion from multi-view consistency to token attention bias."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
QA_SCRIPTS = os.environ.get("NATURE_FIGURE_SCRIPTS")
if not QA_SCRIPTS:
    raise RuntimeError("Set NATURE_FIGURE_SCRIPTS before rendering the publication figure.")
sys.path.insert(0, QA_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


COLORS = {
    "ink": "#263238",
    "muted": "#66737F",
    "line": "#7B8794",
    "panel": "#F5F7F9",
    "blue": "#3B78A5",
    "blue_fill": "#E6F0F7",
    "violet": "#7356A8",
    "violet_fill": "#EEE9F7",
    "teal": "#2A8C82",
    "teal_fill": "#E3F3F0",
    "orange": "#C7772A",
    "orange_fill": "#FAEEDB",
    "green": "#4D8B57",
    "green_fill": "#E8F3E9",
    "white": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
    }
)


def rounded_box(ax, x, y, w, h, face, edge, *, radius=0.012, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=7.0, weight="normal", color=None, ha="center", va="center"):
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        zorder=5,
    )


def arrow(ax, x1, y1, x2, y2, *, color=None, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=1.0,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    ax.add_patch(patch)


def badge(ax, x, y, number, color):
    rounded_box(ax, x, y, 0.030, 0.070, color, color, radius=0.017)
    text(ax, x + 0.015, y + 0.035, str(number), size=7.0, weight="bold", color=COLORS["white"])


def matrix(ax, x, y, size, color, values):
    n = len(values)
    cell = size / n
    for row in range(n):
        for col in range(n):
            alpha = 0.18 + 0.72 * values[row][col]
            ax.add_patch(
                Rectangle(
                    (x + col * cell, y + (n - 1 - row) * cell),
                    cell,
                    cell,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=COLORS["white"],
                    linewidth=0.45,
                    alpha=alpha,
                    zorder=4,
                )
            )


def draw_figure():
    fig = plt.figure(figsize=(7.09, 3.12), facecolor="white")
    ax = fig.add_axes([0.015, 0.045, 0.97, 0.92])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    text(ax, 0.035, 0.955, "VIEW-LEVEL CONSISTENCY FUSION", size=7.6, weight="bold", color=COLORS["blue"], ha="left")
    text(ax, 0.035, 0.475, "TOKEN-LEVEL ATTENTION GUIDANCE", size=7.6, weight="bold", color=COLORS["violet"], ha="left")

    rounded_box(ax, 0.012, 0.525, 0.976, 0.405, "#F1F6FA", "#BED0DC")
    rounded_box(ax, 0.012, 0.055, 0.976, 0.390, "#F8F7FB", "#CEC4DE")

    badge(ax, 0.035, 0.810, 1, COLORS["blue"])
    text(ax, 0.073, 0.845, "Active consistency channels", size=7.3, weight="bold", ha="left")
    channels = [
        (0.045, COLORS["blue"], COLORS["blue_fill"], "Semantic", [[1, .7, .5], [.7, 1, .6], [.5, .6, 1]]),
        (0.136, COLORS["violet"], COLORS["violet_fill"], "Logical", [[1, .35, .75], [.35, 1, .45], [.75, .45, 1]]),
        (0.227, COLORS["teal"], COLORS["teal_fill"], "Local", [[1, .55, .4], [.55, 1, .65], [.4, .65, 1]]),
    ]
    for x, color, fill, name, values in channels:
        rounded_box(ax, x, 0.590, 0.078, 0.170, fill, color, radius=0.009)
        text(ax, x + 0.039, 0.728, name, size=6.3, weight="bold", color=color)
        matrix(ax, x + 0.018, 0.610, 0.043, color, values)

    badge(ax, 0.349, 0.810, 2, COLORS["orange"])
    text(ax, 0.387, 0.845, "Masked channel fusion", size=7.3, weight="bold", ha="left")
    rounded_box(ax, 0.348, 0.590, 0.218, 0.170, COLORS["orange_fill"], COLORS["orange"])
    text(ax, 0.457, 0.714, "αk = masked-softmax(wk)", size=6.5, weight="bold", color=COLORS["orange"])
    text(ax, 0.457, 0.666, "αsem + αlog + αloc = 1", size=6.2)
    text(ax, 0.457, 0.617, "Buv = Σk αk(2Ck_uv − 1)", size=6.4)
    arrow(ax, 0.311, 0.674, 0.342, 0.674, color=COLORS["orange"])

    badge(ax, 0.625, 0.810, 3, COLORS["green"])
    text(ax, 0.663, 0.845, "View-level bias", size=7.3, weight="bold", ha="left")
    rounded_box(ax, 0.625, 0.590, 0.165, 0.170, COLORS["green_fill"], COLORS["green"])
    matrix(ax, 0.648, 0.618, 0.105, COLORS["green"], [[1, .65, .25], [.65, 1, .55], [.25, .55, 1]])
    text(ax, 0.707, 0.742, "B in R³×³", size=6.2, weight="bold", color=COLORS["green"])
    arrow(ax, 0.568, 0.674, 0.619, 0.674, color=COLORS["green"])

    rounded_box(ax, 0.823, 0.590, 0.142, 0.170, COLORS["panel"], COLORS["line"])
    text(ax, 0.894, 0.714, "Interpretation", size=6.7, weight="bold")
    text(ax, 0.894, 0.668, "positive B: strengthen", size=5.9, color=COLORS["blue"])
    text(ax, 0.894, 0.630, "negative B: suppress", size=5.9, color=COLORS["orange"])

    badge(ax, 0.035, 0.332, 4, COLORS["violet"])
    text(ax, 0.073, 0.367, "View-aware token sequence", size=7.2, weight="bold", ha="left")
    tokens = [
        ("[CLS]", COLORS["panel"], COLORS["line"], 0.045),
        ("Title", COLORS["blue_fill"], COLORS["blue"], 0.058),
        ("[SEP]", COLORS["panel"], COLORS["line"], 0.050),
        ("Body", COLORS["teal_fill"], COLORS["teal"], 0.058),
        ("[SEP]", COLORS["panel"], COLORS["line"], 0.050),
        ("Description", COLORS["violet_fill"], COLORS["violet"], 0.087),
        ("[SEP]", COLORS["panel"], COLORS["line"], 0.050),
    ]
    cursor = 0.045
    for value, face, edge, width in tokens:
        rounded_box(ax, cursor, 0.157, width, 0.105, face, edge, radius=0.006, lw=0.75)
        text(ax, cursor + width / 2, 0.210, value, size=5.1, weight="bold" if value in {"Title", "Body", "Description"} else "normal", color=edge)
        cursor += width + 0.003
    text(ax, 0.229, 0.122, "view IDs vi in {T, B, D}", size=6.0, color=COLORS["muted"])

    badge(ax, 0.466, 0.332, 5, COLORS["orange"])
    text(ax, 0.504, 0.367, "View-to-token expansion", size=7.2, weight="bold", ha="left")
    rounded_box(ax, 0.466, 0.137, 0.218, 0.140, COLORS["orange_fill"], COLORS["orange"])
    text(ax, 0.575, 0.232, "Btok[i,j] = B[vi,vj]", size=6.5, weight="bold", color=COLORS["orange"])
    text(ax, 0.575, 0.184, "special and padding positions → 0", size=5.9, color=COLORS["muted"])
    arrow(ax, 0.785, 0.602, 0.680, 0.280, color=COLORS["green"], connectionstyle="arc3,rad=0.16")

    badge(ax, 0.741, 0.332, 6, COLORS["green"])
    text(ax, 0.779, 0.367, "Consistency-aware attention", size=7.2, weight="bold", ha="left")
    rounded_box(ax, 0.741, 0.137, 0.224, 0.140, COLORS["green_fill"], COLORS["green"])
    text(ax, 0.853, 0.232, "A = Softmax(QKᵀ/√d + λBtok)", size=6.3, weight="bold", color=COLORS["green"])
    text(ax, 0.853, 0.184, "context = AV; λ is learnable", size=6.0)
    arrow(ax, 0.686, 0.210, 0.735, 0.210, color=COLORS["green"])

    require_matplotlib_panel_alignment(
        fig,
        json_out=OUT / "qa" / "Fig2_MVCM_to_token_attention_bias.alignment.json",
        overlay_svg=OUT / "qa" / "Fig2_MVCM_to_token_attention_bias.alignment.svg",
        require_panel_labels=False,
        strict=True,
    )

    output = OUT / "Fig2_MVCM_to_token_attention_bias"
    fig.savefig(output.with_suffix(".svg"), format="svg")
    fig.savefig(output.with_suffix(".pdf"), format="pdf")
    fig.savefig(output.with_suffix(".png"), format="png", dpi=600)
    fig.savefig(output.with_suffix(".tiff"), format="tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
