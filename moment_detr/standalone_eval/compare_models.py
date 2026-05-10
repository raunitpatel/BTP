"""
Model Benchmark Comparison Tool (Simplified)
=============================================
Plots 3 metrics across temperatures and video styles.
Shows ± std deviation over 3 prompt runs.

Plots generated:
  1. Line plots  — metric vs temperature, one subplot per video style
  2. Bar charts  — models × video styles at each temperature
  3. Summary table — rows: (model, style), columns: temperature, cells: mean ± std

Usage:
    python compare_models.py
"""

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "Qwen3-VL": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_summarization_with_qwen3_vl",
        "color": "#4C9BE8",
        "marker": "o",
    },
    "Mistral": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_summarization_with_mistral",
        "color": "#E8734C",
        "marker": "s",
    },
    "InternVl2": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_summarization_with_internvl2",
        "color": "#50C878",
        "marker": "^",
    },
    # Add more models here:
    # "NewModel": {
    #     "results_dir": "/path/to/results",
    #     "color": "#FFD700",
    #     "marker": "D",
    # },
}

VIDEO_STYLES = ["cartoon", "cinematic", "realistic", "scribble"]
RUNS         = [1, 2, 3]
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Only these 3 metrics
MAIN_METRICS = {
    "MR-full-mAP":    "MR Full mAP",
    "MR-full-R1@0.5": "MR Full R1@0.5",
    "MR-full-R1@0.7": "MR Full R1@0.7",
}

OUTPUT_DIR = Path("/home/mohit/moment_detr/standalone_eval/comparison_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC STYLE  (white background, serif fonts, colorblind-safe palette)
# Wong (2011) colorblind-safe palette — standard in IEEE/ACM/NeurIPS papers
# ─────────────────────────────────────────────────────────────────────────────

# Patch model colors to colorblind-safe values after MODEL_CONFIGS is defined
_ACADEMIC_COLORS = {
    "Qwen3-VL":  "#0072B2",   # blue
    "Mistral":   "#D55E00",   # vermillion
    "InternVl2": "#009E73",   # teal-green
}
for _m, _c in _ACADEMIC_COLORS.items():
    if _m in MODEL_CONFIGS:
        MODEL_CONFIGS[_m]["color"] = _c

STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "axes.linewidth":    0.8,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        "#CCCCCC",
    "grid.linewidth":    0.5,
    "text.color":        "#111111",
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#AAAAAA",
    "legend.fontsize":   9,
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_all_results(model_configs: dict) -> dict:
    """
    Returns:
        data[model][style][temp] = {
            "mean": {metric: float},
            "std":  {metric: float},
        }
        Each cell is averaged ± std over 3 prompt runs.
    """
    all_data = {}
    for model_name, cfg in model_configs.items():
        base = cfg["results_dir"]
        all_data[model_name] = {}

        for style in VIDEO_STYLES:
            all_data[model_name][style] = {}

            for temp in TEMPERATURES:
                run_metrics = {k: [] for k in MAIN_METRICS}

                for run in RUNS:
                    fname = os.path.join(
                        base, f"eval_{style}_run{run}_temp_{temp:.1f}.json"
                    )
                    if not os.path.exists(fname):
                        print(f"  [WARN] Missing: {fname}")
                        continue

                    data = load_json(fname)
                    brief = data.get("brief", {})
                    for metric_key in MAIN_METRICS:
                        val = brief.get(metric_key, np.nan)
                        run_metrics[metric_key].append(val)

                means = {k: np.nanmean(v) if v else np.nan for k, v in run_metrics.items()}
                stds  = {k: np.nanstd(v)  if v else np.nan for k, v in run_metrics.items()}
                all_data[model_name][style][temp] = {"mean": means, "std": stds}

    return all_data

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Line plots: metric vs temperature, one subplot per video style
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_temperature(all_data: dict, metric_key: str, metric_label: str):
    """
    One figure per metric.
    Subplots = video styles (side by side).
    X-axis   = temperature, Y-axis = metric value.
    Shaded band = ± std over 3 runs.
    """
    apply_style()
    n_styles = len(VIDEO_STYLES)
    fig, axes = plt.subplots(1, n_styles, figsize=(5 * n_styles, 5), sharey=True)
    for ax, style in zip(axes, VIDEO_STYLES):
        for model_name, cfg in MODEL_CONFIGS.items():
            means = np.array(
                [all_data[model_name][style][t]["mean"][metric_key] for t in TEMPERATURES],
                dtype=float,
            )
            stds = np.array(
                [all_data[model_name][style][t]["std"][metric_key] for t in TEMPERATURES],
                dtype=float,
            )

            ax.plot(
                TEMPERATURES, means,
                color=cfg["color"], marker=cfg["marker"],
                linewidth=2, markersize=6, label=model_name,
            )
            ax.fill_between(
                TEMPERATURES,
                means - stds, means + stds,
                color=cfg["color"], alpha=0.18,
            )

        ax.set_title(style.capitalize(), fontsize=10, pad=6)
        ax.set_xlabel("Temperature", fontsize=9)
        ax.set_xticks(TEMPERATURES)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel(metric_label, fontsize=10)
    axes[-1].legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    safe = metric_key.replace("@", "_at_").replace("-", "_")
    out = OUTPUT_DIR / f"line_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Grouped bar chart: models × video styles at a fixed temperature
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_chart(all_data: dict, metric_key: str, metric_label: str, temp: float):
    """
    One figure per (metric, temperature).
    Groups on X-axis = video styles.
    Bars within each group = models.
    Error bars = ± std over 3 runs.
    """
    apply_style()
    models   = list(all_data.keys())
    n_models = len(models)
    x        = np.arange(len(VIDEO_STYLES))
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, model_name in enumerate(models):
        cfg    = MODEL_CONFIGS[model_name]
        means  = [all_data[model_name][s][temp]["mean"][metric_key] for s in VIDEO_STYLES]
        stds   = [all_data[model_name][s][temp]["std"][metric_key]  for s in VIDEO_STYLES]
        offset = (i - (n_models - 1) / 2) * width

        bars = ax.bar(
            x + offset, means, width * 0.9,
            color=cfg["color"], alpha=0.85, label=model_name,
            yerr=stds, capsize=4,
            error_kw={"ecolor": "#555555", "elinewidth": 1.2},
        )

        for bar, mv in zip(bars, means):
            if not np.isnan(mv):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{mv:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in VIDEO_STYLES], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    safe_m = metric_key.replace("@", "_at_").replace("-", "_")
    out = OUTPUT_DIR / f"bar_{safe_m}_temp{temp:.1f}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Combined summary table  (all 3 metrics in one table)
#   Rows    : (model, video_style)  grouped by model
#   Columns : metric × temperature  — grouped column headers
#   Cells   : mean ± std over 3 prompt runs
#   Style   : academic / journal-ready (white bg, no title)
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(all_data: dict):
    """
    Single table combining all 3 metrics.
    Column groups: [MR Full mAP | MR Full R1@0.5 | MR Full R1@0.7]
    Within each group: one sub-column per temperature.
    Rows: (model, video style).
    Cells: mean ± std over 3 prompt runs.
    No title — suitable for captioning externally in a paper.
    """
    apply_style()

    models       = list(all_data.keys())
    metric_keys  = list(MAIN_METRICS.keys())
    metric_labels= list(MAIN_METRICS.values())
    n_metrics    = len(metric_keys)
    n_temps      = len(TEMPERATURES)

    # ── Build column labels (two header rows via blank row trick) ──
    # We'll simulate a two-row header by using a single header row with
    # "Label (T=x)" notation, grouped visually by background banding.
    col_labels = []
    for mk_label in metric_labels:
        for t in TEMPERATURES:
            short = mk_label.replace("MR Full ", "").replace("MR ", "")
            col_labels.append(f"{short}\nT={t:.1f}")

    # ── Build data rows ───────────────────────────────────────────
    rows, row_labels, row_model_map = [], [], []

    for model_name in models:
        for style in VIDEO_STYLES:
            row = []
            for metric_key in metric_keys:
                for temp in TEMPERATURES:
                    mean_val = all_data[model_name][style][temp]["mean"][metric_key]
                    std_val  = all_data[model_name][style][temp]["std"][metric_key]
                    if np.isnan(mean_val):
                        row.append("—")
                    else:
                        row.append(f"{mean_val:.2f}\n±{std_val:.2f}")
            rows.append(row)
            row_labels.append(f"{model_name}\n{style.capitalize()}")
            row_model_map.append(model_name)

    n_rows = len(rows)
    n_cols = len(col_labels)

    # Figure size: wide enough for all columns
    fig_w = max(14, n_cols * 1.55 + 3.0)
    fig_h = max(5,  n_rows * 0.72 + 0.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 2.2)

    model_colors = {name: cfg["color"] for name, cfg in MODEL_CONFIGS.items()}

    # Light pastel row-fill per model (subtle, print-friendly)
    model_bg = {}
    pastel_fills = ["#EAF2FB", "#FEF5EC", "#E9F7EF"]   # blue / orange / green tint
    for m, bg in zip(models, pastel_fills):
        model_bg[m] = bg

    # Metric-group column banding: alternate very light grey / white
    metric_band = ["#F7F7F7", "white", "#F7F7F7"]

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.5)

        if r == 0:
            # Header row — determine which metric group this column belongs to
            if c == -1:
                cell.set_facecolor("white")
                cell.set_text_props(color="white")   # blank corner
            else:
                group_idx = c // n_temps              # 0, 1, or 2
                hdr_bg    = ["#D6E8F7", "#FDDCCC", "#C8EBD9"][group_idx % 3]
                cell.set_facecolor(hdr_bg)
                cell.set_text_props(color="#111111", fontweight="bold", fontsize=7.5)

        elif c == -1:
            # Row label
            row_model = row_model_map[r - 1]
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(
                color=model_colors.get(row_model, "#000000"),
                fontsize=7.5,
                fontweight="bold",
            )
        else:
            # Data cell — row-model tint + metric-group column band blended
            row_model = row_model_map[r - 1]
            group_idx = c // n_temps
            # Use metric group banding for even/odd columns, light model tint for rows
            if group_idx % 2 == 0:
                cell.set_facecolor(model_bg.get(row_model, "#FFFFFF"))
            else:
                cell.set_facecolor("white")
            cell.set_text_props(color="#111111", fontsize=7.8)

    # Draw thick vertical lines between metric groups
    for (r, c), cell in table.get_celld().items():
        if c > 0 and c % n_temps == 0:
            cell.visible_edges = "LBR" if r == 0 else "LBR"
            # Thicken left edge to visually separate groups
            cell.set_linewidth(1.5)

    fig.tight_layout(pad=0.5)
    out = OUTPUT_DIR / "table_all_metrics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Variance / Instability Plot
#   Y-axis : std deviation over 3 prompt runs
#   X-axis : temperature
#   One figure per metric, subplots = video styles
#   Answers: which model is most consistent across prompt runs?
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_vs_temperature(all_data: dict, metric_key: str, metric_label: str):
    """
    Plots *only* the std deviation (prompt-run instability) vs temperature.
    Lower = more consistent / trustworthy model.

    Layout  : 1 figure per metric  →  1 subplot per video style (side-by-side)
    Y-axis  : std dev over 3 runs
    X-axis  : temperature
    Annotated with a horizontal dashed line at the mean std of each model
    so readers can instantly compare average instability.
    """
    apply_style()
    n_styles = len(VIDEO_STYLES)
    fig, axes = plt.subplots(1, n_styles, figsize=(5 * n_styles, 5), sharey=True)

    for ax, style in zip(axes, VIDEO_STYLES):
        for model_name, cfg in MODEL_CONFIGS.items():
            stds = np.array(
                [all_data[model_name][style][t]["std"][metric_key] for t in TEMPERATURES],
                dtype=float,
            )

            # Main σ line
            ax.plot(
                TEMPERATURES, stds,
                color=cfg["color"], marker=cfg["marker"],
                linewidth=2, markersize=6, label=model_name,
                zorder=3,
            )

            # Dot at each point
            ax.scatter(TEMPERATURES, stds, color=cfg["color"], s=30, zorder=4)

            # Horizontal mean-σ reference line (dashed, thin)
            mean_std = np.nanmean(stds)
            ax.axhline(
                mean_std, color=cfg["color"], linewidth=0.8,
                linestyle=":", alpha=0.55,
            )
            # Label the mean-σ on the right edge
            ax.annotate(
                f"μσ={mean_std:.2f}",
                xy=(TEMPERATURES[-1], mean_std),
                xytext=(4, 0), textcoords="offset points",
                fontsize=6.5, color=cfg["color"], va="center",
            )

        ax.set_title(style.capitalize(), fontsize=10, pad=6)
        ax.set_xlabel("Temperature", fontsize=9)
        ax.set_xticks(TEMPERATURES)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, linestyle="--", alpha=0.5)

        # Zero-line for reference
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="-")

    axes[0].set_ylabel(f"Std Dev (\u03c3) of {metric_label}", fontsize=10)
    axes[-1].legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    safe = metric_key.replace("@", "_at_").replace("-", "_")
    out = OUTPUT_DIR / f"variance_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_variance_summary(all_data: dict):
    """
    Single combined figure: average σ (across all styles and temperatures)
    per model per metric — a quick 'who is most stable overall' bar chart.
    """
    apply_style()
    models  = list(all_data.keys())
    metrics = list(MAIN_METRICS.keys())
    labels  = list(MAIN_METRICS.values())

    x      = np.arange(len(metrics))
    width  = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, model_name in enumerate(models):
        cfg        = MODEL_CONFIGS[model_name]
        avg_stds   = []
        err_stds   = []   # std of the stds (how much instability varies)

        for metric_key in metrics:
            all_stds = [
                all_data[model_name][style][temp]["std"][metric_key]
                for style in VIDEO_STYLES
                for temp in TEMPERATURES
            ]
            all_stds = [v for v in all_stds if not np.isnan(v)]
            avg_stds.append(np.mean(all_stds) if all_stds else np.nan)
            err_stds.append(np.std(all_stds)  if all_stds else np.nan)

        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset, avg_stds, width * 0.9,
            color=cfg["color"], alpha=0.85, label=model_name,
            yerr=err_stds, capsize=4,
            error_kw={"ecolor": "#555555", "elinewidth": 1.2},
        )

        for bar, v in zip(bars, avg_stds):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean \u03c3  (prompt-run std deviation)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUTPUT_DIR / "variance_summary_all_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
#   Single file with all models, styles, temperatures, and runs.
#   Two CSV formats written:
#     1. raw_results.csv  — one row per (model, style, temp, run), raw values
#     2. summary_stats.csv — one row per (model, style, temp), mean ± std over runs
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_data: dict):
    metric_keys = list(MAIN_METRICS.keys())

    # ── 1. Summary stats CSV (mean ± std per model/style/temp) ────
    summary_path = OUTPUT_DIR / "summary_stats.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "video_style", "temperature",
            *[f"{m}_mean" for m in metric_keys],
            *[f"{m}_std"  for m in metric_keys],
        ])
        for model_name in all_data:
            for style in VIDEO_STYLES:
                for temp in TEMPERATURES:
                    cell = all_data[model_name][style][temp]
                    writer.writerow([
                        model_name, style, f"{temp:.1f}",
                        *[f"{cell['mean'][m]:.4f}" for m in metric_keys],
                        *[f"{cell['std'][m]:.4f}"  for m in metric_keys],
                    ])
    print(f"  Saved: {summary_path}")

    # ── 2. Raw per-run CSV (re-reads files to get individual run values) ──
    raw_path = OUTPUT_DIR / "raw_results.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "video_style", "temperature", "run",
            *metric_keys,
        ])
        for model_name, cfg in MODEL_CONFIGS.items():
            base = cfg["results_dir"]
            for style in VIDEO_STYLES:
                for temp in TEMPERATURES:
                    for run in RUNS:
                        fname = os.path.join(
                            base, f"eval_{style}_run{run}_temp_{temp:.1f}.json"
                        )
                        if not os.path.exists(fname):
                            continue
                        with open(fname) as jf:
                            brief = json.load(jf).get("brief", {})
                        writer.writerow([
                            model_name, style, f"{temp:.1f}", run,
                            *[f"{brief.get(m, float('nan')):.4f}" for m in metric_keys],
                        ])
    print(f"  Saved: {raw_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model Comparison Tool  (simplified — 3 metrics)")
    print(f"  Models  : {list(MODEL_CONFIGS.keys())}")
    print(f"  Metrics : {list(MAIN_METRICS.keys())}")
    print(f"  Styles  : {VIDEO_STYLES}")
    print(f"  Temps   : {TEMPERATURES}")
    print(f"  Output  : {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────
    print("\n[1/4] Loading evaluation results...")
    all_data = load_all_results(MODEL_CONFIGS)

    # ── CSV export ─────────────────────────────────────────────────
    print("\n[0/4] Exporting CSVs...")
    export_csv(all_data)

    # ── Line plots ─────────────────────────────────────────────────
    print("\n[2/4] Line plots: metric vs temperature (one figure per metric)...")
    for metric_key, metric_label in MAIN_METRICS.items():
        plot_metric_vs_temperature(all_data, metric_key, metric_label)

    # ── Bar charts ─────────────────────────────────────────────────
    print("\n[2b/4] Bar charts: models × styles at each temperature...")
    for metric_key, metric_label in MAIN_METRICS.items():
        for temp in TEMPERATURES:
            plot_bar_chart(all_data, metric_key, metric_label, temp)

    # ── Summary table (all 3 metrics combined, one file) ───────────
    print("\n[3/4] Combined summary table (all metrics)...")
    make_summary_table(all_data)

    # ── Variance / instability plots ───────────────────────────────
    print("\n[4/4] Variance / instability plots...")
    for metric_key, metric_label in MAIN_METRICS.items():
        plot_variance_vs_temperature(all_data, metric_key, metric_label)
    plot_variance_summary(all_data)   # one combined overview bar chart

    # ── Count outputs ──────────────────────────────────────────────
    all_files  = list(OUTPUT_DIR.iterdir())
    line_files = [f for f in all_files if f.name.startswith("line_")]
    bar_files  = [f for f in all_files if f.name.startswith("bar_")]
    tbl_files  = [f for f in all_files if f.name.startswith("table_")]
    var_files  = [f for f in all_files if f.name.startswith("variance_")]

    csv_files  = [f for f in all_files if f.suffix == ".csv"]

    print(f"\n✓  Done.")
    print(f"   CSV files         : {len(csv_files)}    (raw_results.csv + summary_stats.csv)")
    print(f"   Line plots        : {len(line_files)}   (one per metric)")
    print(f"   Bar charts        : {len(bar_files)}    (one per metric × temperature)")
    print(f"   Summary tables    : {len(tbl_files)}    (all 3 metrics combined)")
    print(f"   Variance plots    : {len(var_files)}    (one per metric + 1 summary)")
    print(f"   Output dir        : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()