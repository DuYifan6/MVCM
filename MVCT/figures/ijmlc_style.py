"""Shared visual style and export helpers for the IJML figure set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MM = 1.0 / 25.4
FULL_WIDTH = 180 * MM
SINGLE_WIDTH = 86 * MM

COLORS = {
    "ink": "#263238",
    "muted": "#66737F",
    "grid": "#D7DEE4",
    "baseline": "#AEB8C1",
    "roberta": "#4F5B66",
    "electra": "#5F7F71",
    "without": "#88B4D1",
    "mvct": "#1F5A85",
    "fake": "#1F5A85",
    "ai": "#D27A32",
    "semantic": "#3D7EA6",
    "logical": "#D27A32",
    "local": "#785AA6",
    "positive": "#3F8B62",
    "negative": "#B44C4C",
    "panel_fill": "#F5F7F9",
    "white": "#FFFFFF",
}

METHOD_COLORS = {
    "TF-IDF + LR": COLORS["baseline"],
    "TextCNN": "#C7CED4",
    "BERT": "#7F8A94",
    "RoBERTa": COLORS["roberta"],
    "ELECTRA-base": COLORS["electra"],
    "w/o MVCM": COLORS["without"],
    "MVCT": COLORS["mvct"],
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.6,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 6.7,
            "legend.frameon": False,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, text: str) -> None:
    ax.text(
        -0.12,
        1.06,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def light_x_grid(ax) -> None:
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def load_alignment_helper():
    scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if not scripts:
        raise RuntimeError(
            "Set NATURE_FIGURE_SCRIPTS to the nature-figure scripts directory before rendering."
        )
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def export_figure(fig, output_base: Path, *, dpi: int = 600, alignment_options=None) -> None:
    """Run rendered alignment QA and export an exact-size publication bundle."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    qa_dir = output_base.parent / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()

    require_alignment = load_alignment_helper()
    options = {} if alignment_options is None else dict(alignment_options)
    require_alignment(
        fig,
        json_out=qa_dir / f"{output_base.name}.alignment.json",
        overlay_svg=qa_dir / f"{output_base.name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        **options,
    )

    fig.savefig(output_base.with_suffix(".svg"), format="svg")
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_base.with_suffix(".png"), format="png", dpi=dpi)
    fig.savefig(output_base.with_suffix(".tiff"), format="tiff", dpi=dpi)
    plt.close(fig)
