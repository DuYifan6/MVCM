"""Render IJML-ready quantitative figures from the frozen experimental evidence."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ijmlc_style import COLORS, METHOD_COLORS, apply_style, light_x_grid, panel_label


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

QA_SCRIPTS = os.environ.get("NATURE_FIGURE_SCRIPTS")
if not QA_SCRIPTS:
    raise RuntimeError("Set NATURE_FIGURE_SCRIPTS before rendering the publication figures.")
sys.path.insert(0, QA_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


# These explicit declarations are intentionally kept in the rendering source so
# that a static submission preflight can verify the font and editable-text contract.
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def export_pub(fig, output_base: Path) -> None:
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
    )
    fig.savefig(output_base.with_suffix(".svg"), format="svg")
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_base.with_suffix(".png"), format="png", dpi=600)
    fig.savefig(output_base.with_suffix(".tiff"), format="tiff", dpi=600)
    plt.close(fig)


PERFORMANCE = [
    {"model": "TF-IDF + LR", "fake_f1": 86.78, "ai_f1": 92.93},
    {"model": "TextCNN", "fake_f1": 80.31, "ai_f1": 91.40},
    {"model": "BERT", "fake_f1": 86.27, "ai_f1": 94.82},
    {"model": "RoBERTa", "fake_f1": 86.88, "ai_f1": 95.65},
    {"model": "ELECTRA-base", "fake_f1": 90.18, "ai_f1": 95.74},
    {"model": "w/o MVCM", "fake_f1": 85.84, "ai_f1": 93.89},
    {"model": "MVCT", "fake_f1": 90.30, "ai_f1": 95.44},
]

PAIRED_EFFECTS = [
    {"label": "RoBERTa · Fake/Real", "effect": 3.42, "lo": 2.15, "hi": 4.71, "task": "fake"},
    {"label": "RoBERTa · AI/Human", "effect": -0.20, "lo": -0.79, "hi": 0.38, "task": "ai"},
    {"label": "w/o MVCM · Fake/Real", "effect": 4.45, "lo": 3.16, "hi": 5.78, "task": "fake"},
    {"label": "w/o MVCM · AI/Human", "effect": 1.55, "lo": 0.82, "hi": 2.29, "task": "ai"},
]

ABLATIONS = [
    {"label": "w/o semantic", "fake": 1.17, "ai": 0.79},
    {"label": "w/o logical", "fake": 2.97, "ai": 0.20},
    {"label": "w/o local", "fake": 0.99, "ai": 1.09},
    {"label": "w/o MVCM", "fake": 4.45, "ai": 1.55},
]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_metric_panel(ax, key: str, title: str, xlim: tuple[float, float], show_labels: bool) -> None:
    y = np.arange(len(PERFORMANCE))
    for index, row in enumerate(PERFORMANCE):
        value = row[key]
        color = METHOD_COLORS[row["model"]]
        size = 42 if row["model"] == "MVCT" else 30
        ax.scatter(value, index, s=size, color=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([row["model"] for row in PERFORMANCE] if show_labels else [])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_title(title, loc="left", pad=5)
    light_x_grid(ax)


def figure3() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 118 / 25.4))
    fig.subplots_adjust(left=0.15, right=0.975, top=0.91, bottom=0.12, wspace=0.48, hspace=0.58)

    draw_metric_panel(axes[0, 0], "fake_f1", "Fake/Real detection", (78.5, 92.0), True)
    draw_metric_panel(axes[0, 1], "ai_f1", "AI/Human detection", (90.5, 96.6), False)

    ax = axes[1, 0]
    y = np.arange(len(PAIRED_EFFECTS))
    for index, row in enumerate(PAIRED_EFFECTS):
        color = COLORS[row["task"]]
        ax.plot([row["lo"], row["hi"]], [index, index], color=color, lw=1.7, solid_capstyle="round", zorder=2)
        ax.scatter(row["effect"], index, s=34, color=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax.axvline(0, color=COLORS["muted"], lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in PAIRED_EFFECTS])
    ax.invert_yaxis()
    ax.set_xlim(-1.25, 6.15)
    ax.set_xlabel("MVCT minus comparator (Macro-F1 points)")
    ax.set_title("Paired test-set effects (95% CI)", loc="left", pad=5)
    light_x_grid(ax)

    ax = axes[1, 1]
    y = np.arange(len(ABLATIONS))
    height = 0.31
    ax.barh(y - height / 2, [row["fake"] for row in ABLATIONS], height=height, color=COLORS["fake"], label="Fake/Real")
    ax.barh(y + height / 2, [row["ai"] for row in ABLATIONS], height=height, color=COLORS["ai"], label="AI/Human")
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in ABLATIONS])
    ax.invert_yaxis()
    ax.set_xlim(0, 5.05)
    ax.set_xlabel("Decrease from full MVCT (Macro-F1 points)")
    ax.set_title("Channel and module ablation", loc="left", pad=5)
    ax.legend(loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.0)

    for text, axis in zip("abcd", axes.flat):
        panel_label(axis, text)

    export_pub(fig, OUT / "Fig3_performance_and_ablation")


def plot_subgroup_panel(ax, rows, metric: str, title: str, ylim: tuple[float, float]) -> None:
    groups = ["HR", "MR", "HF", "MF"]
    models = ["MVCT", "RoBERTa", "w/o MVCM"]
    source_names = {"MVCT": "mvct", "RoBERTa": "roberta", "w/o MVCM": "without_mvcm"}
    offsets = {"MVCT": -0.20, "RoBERTa": 0.0, "w/o MVCM": 0.20}
    markers = {"MVCT": "o", "RoBERTa": "s", "w/o MVCM": "^"}
    n_by_group = {row["group"]: int(row["n"]) for row in rows if row["model"] == "mvct"}
    lookup = {(row["model"], row["group"]): row for row in rows}

    for model in models:
        xs = np.arange(len(groups), dtype=float) + offsets[model]
        values = []
        lows = []
        highs = []
        for group in groups:
            row = lookup[(source_names[model], group)]
            values.append(float(row[f"{metric}_accuracy"]) * 100)
            lows.append(float(row[f"{metric}_ci95_low"]) * 100)
            highs.append(float(row[f"{metric}_ci95_high"]) * 100)
        values = np.asarray(values)
        errors = np.vstack([values - np.asarray(lows), np.asarray(highs) - values])
        ax.errorbar(
            xs,
            values,
            yerr=errors,
            fmt=markers[model],
            ms=4.5,
            mfc=METHOD_COLORS[model],
            mec="white",
            mew=0.5,
            ecolor=METHOD_COLORS[model],
            elinewidth=1.0,
            capsize=2.1,
            capthick=0.9,
            label=model,
            zorder=3,
        )

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{group}\nn = {n_by_group[group]}" for group in groups])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Conditional accuracy (%)")
    ax.set_title(title, loc="left", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def figure4() -> None:
    rows = read_csv(ROOT / "p0_submission_evidence" / "subgroups" / "subgroup_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 78 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.23, wspace=0.28)

    plot_subgroup_panel(axes[0], rows, "veracity", "Veracity correctness", (35, 102))
    plot_subgroup_panel(axes[1], rows, "provenance", "Provenance correctness", (80, 101.5))
    for text, axis in zip("ab", axes):
        panel_label(axis, text)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=3, columnspacing=1.8, handletextpad=0.5)
    export_pub(fig, OUT / "Fig4_subgroup_behavior")


def figure5() -> None:
    rows = read_csv(
        ROOT
        / "artifacts"
        / "p0_20260918"
        / "efficiency_clean"
        / "efficiency_clean"
        / "efficiency_results.csv"
    )
    display = {"three_channel_full": "MVCT", "without_mvcm": "w/o MVCM", "roberta": "RoBERTa"}
    normalized = []
    for row in rows:
        normalized.append(
            {
                "model": display[row["variant"]],
                "batch": int(row["batch_size"]),
                "median": float(row["median_ms_per_article"]),
                "p95": float(row["p95_ms_per_article"]),
                "throughput": float(row["throughput_articles_per_second"]),
                "memory": float(row["peak_allocated_gib"]),
            }
        )
    f1 = {row["model"]: row["fake_f1"] for row in PERFORMANCE if row["model"] in {"MVCT", "w/o MVCM", "RoBERTa"}}

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 79 / 25.4))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.23, wspace=0.47)

    ax = axes[0]
    order = [(model, batch) for model in ["MVCT", "w/o MVCM", "RoBERTa"] for batch in [1, 8]]
    y = np.arange(len(order))
    lookup = {(row["model"], row["batch"]): row for row in normalized}
    for index, key in enumerate(order):
        row = lookup[key]
        color = METHOD_COLORS[row["model"]]
        ax.plot([row["median"], row["p95"]], [index, index], color=color, lw=1.4, zorder=2)
        median_marker = "o" if row["batch"] == 1 else "s"
        ax.scatter(row["median"], index, marker=median_marker, s=27, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(row["p95"], index, marker="|", s=50, color=color, linewidth=1.3, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{model} · B{batch}" for model, batch in order])
    ax.invert_yaxis()
    ax.set_xlim(3.5, 20.2)
    ax.set_xlabel("Latency (ms/article)")
    ax.set_title("Warm-cache latency", loc="left", pad=5)
    light_x_grid(ax)

    ax = axes[1]
    models = ["MVCT", "w/o MVCM", "RoBERTa"]
    x = np.arange(len(models))
    for index, model in enumerate(models):
        values = [lookup[(model, batch)]["memory"] for batch in (1, 8)]
        ax.plot([index, index], values, color=METHOD_COLORS[model], lw=1.35, zorder=2)
        ax.scatter(index, values[0], marker="o", s=34, color=METHOD_COLORS[model], edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(index, values[1], marker="s", s=34, color=METHOD_COLORS[model], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["MVCT", "w/o\nMVCM", "RoBERTa"])
    ax.set_ylim(0.43, 0.72)
    ax.set_ylabel("Peak allocated GPU memory (GiB)")
    ax.set_title("Memory footprint", loc="left", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75, zorder=0)

    ax = axes[2]
    marker = {1: "o", 8: "s"}
    for model in models:
        batch_rows = sorted([row for row in normalized if row["model"] == model], key=lambda item: item["batch"])
        ax.plot([row["median"] for row in batch_rows], [f1[model]] * len(batch_rows), color=METHOD_COLORS[model], lw=1.1, alpha=0.75)
        for row in batch_rows:
            ax.scatter(
                row["median"],
                f1[model],
                marker=marker[row["batch"]],
                s=38,
                color=METHOD_COLORS[model],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        midpoint = np.mean([row["median"] for row in batch_rows])
        ax.text(midpoint, f1[model] + 0.20, model, fontsize=5.8, color=METHOD_COLORS[model], ha="center", va="bottom")
    ax.set_xlim(3.5, 15.8)
    ax.set_ylim(85.2, 91.0)
    ax.set_xlabel("Median latency (ms/article)")
    ax.set_ylabel("Fake/Real Macro-F1 (%)")
    ax.set_title("Performance–latency trade-off", loc="left", pad=5)
    batch_handles = [
        mlines.Line2D([], [], color=COLORS["ink"], marker="o", linestyle="None", markersize=4.5, label="Batch 1"),
        mlines.Line2D([], [], color=COLORS["ink"], marker="s", linestyle="None", markersize=4.5, label="Batch 8"),
    ]

    for text, axis in zip("abc", axes):
        panel_label(axis, text)

    fig.legend(handles=batch_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=2, columnspacing=1.4, handletextpad=0.4)

    export_pub(fig, OUT / "Fig5_efficiency_and_resource_cost")


def main() -> None:
    apply_style()
    figure3()
    figure4()
    figure5()


if __name__ == "__main__":
    main()
