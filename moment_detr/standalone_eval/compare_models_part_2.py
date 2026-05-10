import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/mohit/moment_detr/standalone_eval/comparison_outputs_part_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_STYLES = ["cartoon", "cinematic", "realistic", "scribble"]
TEMPERATURES = [0.2]
RUNS = list(range(1, 11))

RUN_LABELS = {
    1:  'Color',
    2:  'Object Placement / Position',
    3:  'Object Addition',
    4:  'Object Removal / Replacement',
    5:  'Lighting / Time of Day',
    6:  'Weather / Environment',
    7:  'Human Expression / Pose',
    8:  'Texture / Material',
    9:  'Scale / Size',
    10: 'Quantity / Count'
}

MAIN_METRICS = [
    "MR-full-mAP",
    "MR-full-R1@0.5",
    "MR-full-R1@0.7",
]

MODEL_CONFIGS = {
    "Qwen3-VL": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_part_2_summarization_with_qwen3_vl",
        "color": "#0072B2",   # Wong (2011) blue
        "marker": "o",
    },
    "Mistral": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_part_2_summarization_with_mistral",
        "color": "#D55E00",   # Wong (2011) vermillion
        "marker": "s",
    },
    "InternVl2": {
        "results_dir": "/home/mohit/moment_detr/standalone_eval/results/results_second_sem_part_2_summarization_with_internvl2",
        "color": "#009E73",   # Wong (2011) teal-green
        "marker": "^",
    },
    # Add more models here:
    # "NewModel": {
    #     "results_dir": "/path/to/results",
    #     "color": "#E69F00",   # Wong (2011) orange
    #     "marker": "D",
    # },
}

METRICS_DISPLAY = {
    "MR-full-mAP":    "MR Full mAP (avg)",
    "MR-full-R1@0.5": "MR Full R1@0.5",
    "MR-full-R1@0.7": "MR Full R1@0.7",
}

# ─────────────────────────────────────────────────────────────────────────────
# ACADEMIC STYLE  (white background, serif fonts, colorblind-safe palette)
# Wong (2011) colorblind-safe palette — standard in IEEE/ACM/NeurIPS papers
# Full palette for reference when adding more models:
#   "#0072B2"  blue          "#D55E00"  vermillion
#   "#009E73"  teal-green    "#E69F00"  orange
#   "#56B4E9"  sky-blue      "#CC79A7"  reddish-purple
#   "#F0E442"  yellow (avoid on white bg)
# ─────────────────────────────────────────────────────────────────────────────

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

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def collect_run_data():
    """
    data[model][style][run][metric] = value (scalar)
    """
    data = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        base = cfg["results_dir"]
        data[model_name] = {}

        for style in VIDEO_STYLES:
            data[model_name][style] = {}

            for run in RUNS:
                data[model_name][style][run] = {}

                for temp in TEMPERATURES:
                    fname = os.path.join(
                        base, f"eval_{style}_run{run}_temp_{temp:.1f}.json"
                    )

                    if not os.path.exists(fname):
                        print(f"[WARN] Missing: {fname}")
                        continue

                    js = load_json(fname)
                    brief = js.get("brief", {})

                    for metric in MAIN_METRICS:
                        val = brief.get(metric, np.nan)
                        data[model_name][style][run][metric] = val

    return data

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_with_subplots(data, metric_key, metric_name):
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    x = np.arange(len(RUNS))
    x_labels = [RUN_LABELS[r] for r in RUNS]

    for idx, style in enumerate(VIDEO_STYLES):
        ax = axes[idx]
        ax.set_facecolor("white")

        for model_name, cfg in MODEL_CONFIGS.items():
            values = np.array([
                data[model_name][style][run].get(metric_key, np.nan)
                for run in RUNS
            ])

            ax.plot(
                x, values,
                label=model_name,
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=1.5,
                markersize=5,
            )

        ax.set_title(style.capitalize(), fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supxlabel("Artifact Category", fontsize=10, y=0.01)
    fig.supylabel(metric_name, fontsize=10, x=0.01)

    # Single legend outside plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(MODEL_CONFIGS),
        fontsize=9, framealpha=0.9, edgecolor="#AAAAAA",
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.95])

    save_path = OUTPUT_DIR / f"{metric_key}_subplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
#   One CSV file per model.
#   Columns: video_style | artifact_category | MR-full-mAP | MR-full-R1@0.5 | MR-full-R1@0.7
#   Rows   : every (style, run) combination — one row per data point
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(data: dict):
    """
    Single CSV containing all models.
    Columns: model | video_style | artifact_category | metric1 | metric2 | metric3
    Rows   : every (model, style, run) combination.
    """
    out_path = OUTPUT_DIR / "artifact_results_all_models.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "video_style", "artifact_category",
            *MAIN_METRICS
        ])
        for model_name in data:
            for style in VIDEO_STYLES:
                for run in RUNS:
                    row_data = data[model_name][style][run]
                    writer.writerow([
                        model_name,
                        style,
                        RUN_LABELS[run],
                        *[f"{row_data.get(m, float('nan')):.4f}" for m in MAIN_METRICS],
                    ])
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE IMAGE
#   Rows    : artifact category (10 rows)
#   Columns : video_style × metric  — grouped by style
#   Cells   : scalar value (no averaging — fixed temp=0.2, single run)
#   One table per model.
#   Academic style: white bg, no title, serif font, pastel column banding
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(data: dict):
    apply_style()

    metric_keys   = MAIN_METRICS
    metric_shorts = ["mAP", "R1@0.5", "R1@0.7"]   # short header labels
    n_styles  = len(VIDEO_STYLES)
    n_metrics = len(metric_keys)

    # ── Column labels ──────────────────────────────────────────────
    # Two-level header: style group / metric sub-column
    # We simulate it with a single header row: "Style\nMetric"
    col_labels = []
    for style in VIDEO_STYLES:
        for ms in metric_shorts:
            col_labels.append(f"{style.capitalize()}\n{ms}")

    # Pastel header tints per style group (print-safe)
    style_hdr_colors  = ["#D6E8F7", "#FDDCCC", "#C8EBD9", "#EDE7F6"]
    style_data_colors = ["#EAF2FB", "#FEF5EC", "#E9F7EF", "#F3EEF9"]

    for model_name in data:
        # ── Build rows ─────────────────────────────────────────────
        rows, row_labels = [], []
        for run in RUNS:
            row = []
            for style in VIDEO_STYLES:
                for metric_key in metric_keys:
                    val = data[model_name][style][run].get(metric_key, float("nan"))
                    row.append("—" if np.isnan(val) else f"{val:.2f}")
            rows.append(row)
            row_labels.append(RUN_LABELS[run])

        n_rows = len(rows)
        n_cols = len(col_labels)

        fig_w = max(14, n_cols * 1.3 + 2.5)
        fig_h = max(5,  n_rows * 0.65 + 0.6)

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
        table.set_fontsize(8.0)
        table.scale(1, 2.0)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#BBBBBB")
            cell.set_linewidth(0.5)

            if r == 0:
                if c == -1:
                    # blank corner
                    cell.set_facecolor("white")
                    cell.set_text_props(color="white")
                else:
                    group_idx = c // n_metrics
                    cell.set_facecolor(style_hdr_colors[group_idx % n_styles])
                    cell.set_text_props(
                        color="#111111", fontweight="bold", fontsize=7.8
                    )
                    # Thicken left edge at each style group boundary
                    if c % n_metrics == 0 and c > 0:
                        cell.set_linewidth(1.5)
            elif c == -1:
                # Row label
                cell.set_facecolor("#F5F5F5")
                cell.set_text_props(color="#111111", fontsize=8.0, fontweight="bold")
            else:
                group_idx = c // n_metrics
                cell.set_facecolor(style_data_colors[group_idx % n_styles])
                cell.set_text_props(color="#111111", fontsize=8.0)
                if c % n_metrics == 0 and c > 0:
                    cell.set_linewidth(1.5)

        fig.tight_layout(pad=0.4)
        out_path = OUTPUT_DIR / f"{model_name}_summary_table.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = collect_run_data()

    print("\nGenerating plots...")
    for metric_key, metric_name in METRICS_DISPLAY.items():
        plot_metric_with_subplots(data, metric_key, metric_name)

    print("\nExporting CSV...")
    export_csv(data)

    print("\nGenerating summary tables...")
    make_summary_table(data)

    print(f"\nDone. All outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()