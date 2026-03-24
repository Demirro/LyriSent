import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from config import RESULTS_DIR, EMOTION_LABELS as DEFAULT_EMOTION_LABELS

# Set style for academic publications
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Consistent method colors across all applicable plots.
METHOD_FAMILY_COLORS = {
    "NaiveBayes": "#4C78A8",      # blue
    "BERT": "#F58518",            # orange
    "OpenAI_zero_shot": "#54A24B",# green
    "OpenAI_few_shot": "#E45756", # red
    "OpenAI_other": "#B279A2",    # purple fallback
    "Other": "#9D9D9D",           # gray fallback
}

# Prefer latest run subfolder (created by testing.py); fallback to RESULTS_DIR.
def resolve_results_dir() -> Path:
    canonical_files = ("methods_comparison_summary.csv", "runtime_cost_summary.csv")
    if all((RESULTS_DIR / name).exists() for name in canonical_files):
        return RESULTS_DIR

    run_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir()] if RESULTS_DIR.exists() else []
    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if all((run_dir / name).exists() for name in canonical_files):
            print(f"Using latest run directory: {run_dir}")
            return run_dir
    return RESULTS_DIR


ACTIVE_RESULTS_DIR = resolve_results_dir()

# Create output directory for figures
FIGURES_DIR = ACTIVE_RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def set_active_results_dir(results_dir: Path) -> None:
    global ACTIVE_RESULTS_DIR, FIGURES_DIR
    ACTIVE_RESULTS_DIR = Path(results_dir)
    FIGURES_DIR = ACTIVE_RESULTS_DIR / 'figures'
    FIGURES_DIR.mkdir(exist_ok=True)


def list_run_dirs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    run_dirs = []
    for p in RESULTS_DIR.iterdir():
        if not p.is_dir():
            continue
        if (p / "methods_comparison_summary.csv").exists():
            run_dirs.append(p)
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime)


def _load_summary_from_run(run_dir: Path) -> pd.DataFrame | None:
    summary_path = run_dir / "methods_comparison_summary.csv"
    if not summary_path.exists():
        return None
    try:
        df = pd.read_csv(summary_path, index_col=0).copy()
        df["method_key"] = df.index.astype(str)
        if "display_name" not in df.columns:
            df["display_name"] = df["method_key"]
        df["run_dir"] = str(run_dir.name)
        return df
    except Exception:
        return None


def create_cross_run_comparison(run_dirs: list[Path], output_dir: Path) -> None:
    if not run_dirs:
        print("No run folders found for cross-run comparison.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for run_dir in run_dirs:
        df = _load_summary_from_run(run_dir)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("No methods comparison summaries could be loaded.")
        return

    all_runs_df = pd.concat(frames, ignore_index=True)
    all_runs_path = output_dir / "all_runs_methods_comparison.csv"
    all_runs_df.to_csv(all_runs_path, index=False)
    print(f"Saved: {all_runs_path}")

    numeric_cols = [
        c for c in ["avg_accuracy", "std_accuracy", "avg_f1_weighted", "avg_f1_pos", "avg_auc_roc", "avg_pr_auc"]
        if c in all_runs_df.columns
    ]
    if not numeric_cols:
        print("No numeric columns found for cross-run aggregation.")
        return

    grouped = (
        all_runs_df.groupby(["method_key", "display_name"], dropna=False)[numeric_cols]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index().sort_values(by=["display_name"])

    summary_path = output_dir / "cross_run_method_summary.csv"
    grouped.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    if "method_key" in grouped.columns:
        openai_grouped = grouped[grouped["method_key"].str.startswith("OpenAI_", na=False)].copy()
        if not openai_grouped.empty:
            openai_path = output_dir / "cross_run_openai_summary.csv"
            openai_grouped.to_csv(openai_path, index=False)
            print(f"Saved: {openai_path}")
    plot_cross_run_summary(grouped, output_dir)


def _plot_cross_run_bar_with_error(
    df: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    if metric_mean_col not in df.columns:
        print(f"Warning: {metric_mean_col} not found; skipping {save_path.name}")
        return
    plot_df = df.copy()
    plot_df = plot_df.sort_values(by=["display_name"])
    means = plot_df[metric_mean_col].astype(float).to_numpy()
    stds = (
        plot_df[metric_std_col].astype(float).fillna(0.0).to_numpy()
        if metric_std_col in plot_df.columns
        else np.zeros(len(plot_df))
    )
    labels = plot_df["display_name"].astype(str).tolist()
    x = np.arange(len(labels))
    bar_colors = [_method_color(name) for name in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.0,
        color=bar_colors,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(_resolve_method_labels(labels), rotation=25, ha="right")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(0.98, float(bar.get_height()) + float(std) + 0.02),
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_run_summary(grouped_df: pd.DataFrame, output_dir: Path) -> None:
    if grouped_df.empty:
        print("No grouped cross-run data available for plotting.")
        return

    # Figure 1: Cross-run avg accuracy (mean ± std)
    _plot_cross_run_bar_with_error(
        grouped_df,
        metric_mean_col="avg_accuracy_mean",
        metric_std_col="avg_accuracy_std",
        title="Cross-Run Method Comparison: Average Accuracy",
        ylabel="Average Accuracy",
        save_path=output_dir / "cross_run_accuracy_comparison.png",
    )

    # Figure 2: Cross-run weighted F1 (mean ± std)
    _plot_cross_run_bar_with_error(
        grouped_df,
        metric_mean_col="avg_f1_weighted_mean",
        metric_std_col="avg_f1_weighted_std",
        title="Cross-Run Method Comparison: Weighted F1",
        ylabel="Average Weighted F1",
        save_path=output_dir / "cross_run_f1_comparison.png",
    )

    # Figure 3: OpenAI-only variants for easier zero-shot vs few-shot reading.
    openai_df = grouped_df[grouped_df["method_key"].astype(str).str.startswith("OpenAI_", na=False)].copy()
    if not openai_df.empty:
        _plot_cross_run_bar_with_error(
            openai_df,
            metric_mean_col="avg_accuracy_mean",
            metric_std_col="avg_accuracy_std",
            title="Cross-Run OpenAI Variants: Average Accuracy",
            ylabel="Average Accuracy",
            save_path=output_dir / "cross_run_openai_accuracy.png",
        )
        _plot_cross_run_bar_with_error(
            openai_df,
            metric_mean_col="avg_f1_weighted_mean",
            metric_std_col="avg_f1_weighted_std",
            title="Cross-Run OpenAI Variants: Weighted F1",
            ylabel="Average Weighted F1",
            save_path=output_dir / "cross_run_openai_f1.png",
        )


def load_metrics_data():
    """Load all metrics summary files."""
    metrics_data = {}
    summary_display_name_by_key = {}
    summary_path = ACTIVE_RESULTS_DIR / 'methods_comparison_summary.csv'
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path, index_col=0)
            if "display_name" in summary_df.columns:
                summary_display_name_by_key = {
                    str(idx): str(row["display_name"])
                    for idx, row in summary_df.iterrows()
                    if pd.notna(row.get("display_name"))
                }
        except Exception:
            summary_display_name_by_key = {}

    metrics_files = sorted(ACTIVE_RESULTS_DIR.glob('metrics_summary_*.csv'))
    if not metrics_files:
        print(f"Warning: no metrics_summary_*.csv files found in {ACTIVE_RESULTS_DIR}")
        return metrics_data

    for file_path in metrics_files:
        key = file_path.stem.replace("metrics_summary_", "", 1)
        summary_key = {
            "naivebayes": "NaiveBayes",
            "bert": "BERT",
        }.get(key.lower(), key)
        display_name = summary_display_name_by_key.get(summary_key, summary_key)
        df = pd.read_csv(file_path)
        metrics_data[display_name] = df
    
    return metrics_data


def load_comparison_summary():
    file_path = ACTIVE_RESULTS_DIR / 'methods_comparison_summary.csv'
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0)
        if "display_name" in df.columns:
            display_names = df["display_name"].astype(str)
            fallback_index = pd.Series(df.index.astype(str), index=df.index)
            missing_mask = df["display_name"].isna()
            display_names.loc[missing_mask] = fallback_index.loc[missing_mask]
            df.index = display_names
        return df
    return None


def infer_emotions(metrics_data: dict, predictions_df: pd.DataFrame | None = None) -> list[str]:
    """
    Infer which emotions were actually evaluated.

    Preference order:
    1) From metrics files (column 'Emotion')
    2) From predictions file columns (Truth_*)
    3) Fallback to config default order (Plutchik 8)

    Returns emotions in a stable order (config order first, then any extras).
    """
    emotions_from_metrics: set[str] = set()
    for df in metrics_data.values():
        if "Emotion" in df.columns:
            emotions_from_metrics.update(df["Emotion"].dropna().astype(str).tolist())

    emotions_from_truth: set[str] = set()
    if predictions_df is not None:
        for c in predictions_df.columns:
            if isinstance(c, str) and c.startswith("Truth_"):
                emotions_from_truth.add(c.replace("Truth_", "", 1))

    candidates = emotions_from_metrics or emotions_from_truth
    if not candidates:
        candidates = set(DEFAULT_EMOTION_LABELS)

    # Stable ordering: config order first, then remaining in sorted order.
    ordered = [e for e in DEFAULT_EMOTION_LABELS if e in candidates]
    extras = sorted([e for e in candidates if e not in ordered])
    return ordered + extras


def _compact_method_label(label: str, max_len: int = 26) -> str:
    text = str(label)
    text = text.replace("OpenAI (", "OA(")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _resolve_method_labels(methods: list[str]) -> list[str]:
    compact = [_compact_method_label(m) for m in methods]
    if len(set(compact)) == len(compact):
        return compact
    # Keep compact labels but make collisions unique.
    counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in compact:
        n = counts.get(label, 0) + 1
        counts[label] = n
        unique_labels.append(label if n == 1 else f"{label} ({n})")
    return unique_labels


def _method_family(method_name: str) -> str:
    text = str(method_name).lower()
    if "naivebayes" in text or text == "nb":
        return "NaiveBayes"
    if "bert" in text:
        return "BERT"
    if "openai" in text:
        if "zero_shot" in text or "zero-shot" in text:
            return "OpenAI_zero_shot"
        if "few_shot" in text or "few-shot" in text:
            return "OpenAI_few_shot"
        return "OpenAI_other"
    return "Other"


def _method_color(method_name: str) -> str:
    return METHOD_FAMILY_COLORS.get(_method_family(method_name), METHOD_FAMILY_COLORS["Other"])


def plot_method_comparison(summary_df, save_path=None):
    """
    Figure 1: Overall method comparison (Accuracy and F1-Score).
    """
    methods = summary_df.index.tolist()
    tick_labels = _resolve_method_labels(methods)
    fig_width = max(12, len(methods) * 2.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 5.6))
    
    x_pos = np.arange(len(methods))
    width = 0.6
    
    # Accuracy comparison
    accuracies = summary_df['avg_accuracy'].values
    acc_stds = summary_df['std_accuracy'].values
    bar_colors = [_method_color(m) for m in methods]
    
    bars1 = ax1.bar(x_pos, accuracies, width, yerr=acc_stds, 
                    capsize=5, alpha=0.9, edgecolor='black', linewidth=1.2, color=bar_colors)
    ax1.set_xlabel('Method', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(a) Average Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tick_labels, rotation=20, ha='right', fontsize=9)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax1.legend()
    
    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars1, accuracies, acc_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{acc:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1-Score comparison
    f1_scores = summary_df['avg_f1_weighted'].values
    
    bars2 = ax2.bar(x_pos, f1_scores, width, alpha=0.9, 
                    edgecolor='black', linewidth=1.2, color=bar_colors)
    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('Weighted F1-Score', fontweight='bold')
    ax2.set_title('(b) Average F1-Score Comparison', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tick_labels, rotation=20, ha='right', fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.legend()
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_per_emotion_performance(metrics_data, save_path=None):
    """
    Figure 2: Per-emotion performance heatmap (Accuracy, Precision, Recall, F1).
    """
    # Prepare data for heatmap
    methods = list(metrics_data.keys())
    method_labels = _resolve_method_labels(methods)
    # Order requested for subplot layout:
    # top-left F1, top-right Accuracy, bottom-left Precision, bottom-right Recall
    metrics_to_plot = ['F1_Pos', 'Accuracy', 'Precision_Pos', 'Recall_Pos']
    emotions = infer_emotions(metrics_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Create matrix: rows = emotions, columns = methods
        heatmap_data = []
        for emotion in emotions:
            row = []
            for method in methods:
                df = metrics_data[method]
                emotion_data = df[df['Emotion'] == emotion]
                if not emotion_data.empty:
                    value = emotion_data[metric].values[0] if metric in emotion_data.columns else np.nan
                    row.append(value)
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=emotions, 
                                 columns=method_labels)
        
        # Create heatmap
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': metric},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold', pad=10)
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel('Emotion', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Per-Emotion Performance Metrics Across Methods', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_emotion_comparison_bar(metrics_data, save_path=None):
    """
    Figure 3: Per-emotion accuracy comparison (grouped bar chart).
    """
    methods = list(metrics_data.keys())
    emotions = infer_emotions(metrics_data)
    x = np.arange(len(emotions))
    width = 0.25
    
    fig_width = max(14, len(methods) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    method_labels = _resolve_method_labels(methods)
    
    for i, method in enumerate(methods):
        accuracies = []
        for emotion in emotions:
            df = metrics_data[method]
            emotion_data = df[df['Emotion'] == emotion]
            if not emotion_data.empty and 'Accuracy' in emotion_data.columns:
                accuracies.append(emotion_data['Accuracy'].values[0])
            else:
                accuracies.append(np.nan)
        
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=method_labels[i], 
                     alpha=0.9, edgecolor='black', linewidth=0.8, color=_method_color(method))
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            if not np.isnan(acc):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Emotion', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Per-Emotion Accuracy Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(metrics_data, save_path=None):
    """
    Figure 4: Confusion matrices for each method (if available in data).
    Note: This requires the full predictions data, not just summary metrics.
    """
    # Load full predictions if available
    predictions_file = ACTIVE_RESULTS_DIR / 'rada_all_predictions.csv'
    if not predictions_file.exists():
        print("Warning: Full predictions file not found. Skipping confusion matrices.")
        return
    
    df = pd.read_csv(predictions_file)
    emotions = infer_emotions(metrics_data, predictions_df=df)
    methods = ['NB', 'BERT']
    openai_prefixes = sorted(
        {
            c.rsplit("_", 1)[0]
            for c in df.columns
            if isinstance(c, str) and c.startswith("OpenAI_") and any(c.endswith(f"_{e}") for e in emotions)
        }
    )
    methods.extend(openai_prefixes)
    
    fig, axes = plt.subplots(1, len(methods), figsize=(max(5 * len(methods), 12), 4.2))
    if len(methods) == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Aggregate confusion matrices across all emotions
        all_true = []
        all_pred = []
        
        for emotion in emotions:
            truth_col = f'Truth_{emotion}'
            pred_col = f'{method}_{emotion}'
            
            if truth_col in df.columns and pred_col in df.columns:
                mask = df[[truth_col, pred_col]].notna().all(axis=1)
                truth = df.loc[mask, truth_col].astype(int)
                pred = df.loc[mask, pred_col]
                
                # Convert probabilities to binary for NB and BERT
                if method in ['NB', 'BERT']:
                    pred = (pred >= 0.5).astype(int)
                else:
                    pred = pred.astype(int)
                
                all_true.extend(truth.values)
                all_pred.extend(pred.values)
        
        if len(all_true) > 0:
            cm = confusion_matrix(all_true, all_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'}, linewidths=0.5)
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_title(f'{_compact_method_label(method, max_len=20)}', fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{_compact_method_label(method, max_len=20)}', fontweight='bold')
    
    plt.suptitle('Confusion Matrices (Aggregated over song×emotion pairs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_precision_recall_comparison(metrics_data, save_path=None):
    """
    Figure 5: Precision-Recall comparison for positive class.
    """
    methods = list(metrics_data.keys())
    method_labels = _resolve_method_labels(methods)
    emotions_all = infer_emotions(metrics_data)
    
    fig, ax = plt.subplots(figsize=(11, 8))

    for idx, (method, method_label) in enumerate(zip(methods, method_labels)):
        color = _method_color(method)
        precisions = []
        recalls = []
        emotions = []
        
        for emotion in emotions_all:
            df = metrics_data[method]
            emotion_data = df[df['Emotion'] == emotion]
            if not emotion_data.empty:
                if 'Precision_Pos' in emotion_data.columns and 'Recall_Pos' in emotion_data.columns:
                    prec = emotion_data['Precision_Pos'].values[0]
                    rec = emotion_data['Recall_Pos'].values[0]
                    if not (np.isnan(prec) or np.isnan(rec)):
                        precisions.append(prec)
                        recalls.append(rec)
                        emotions.append(emotion)
        
        if len(precisions) > 0:
            # Small deterministic jitter by method index helps separate stacked points (e.g., repeated 0,0).
            jitter = (idx - (len(methods) - 1) / 2.0) * 0.008
            recalls_plot = np.clip(np.array(recalls, dtype=float) + jitter, 0.0, 1.0)
            precisions_plot = np.clip(np.array(precisions, dtype=float) + jitter, 0.0, 1.0)
            ax.scatter(recalls_plot, precisions_plot, s=150, alpha=0.7, 
                      label=method_label, color=color, edgecolors='black', linewidth=1.2)
            
            # Add emotion labels
            for i, emotion in enumerate(emotions):
                ax.annotate(
                    emotion,
                    (recalls_plot[i], precisions_plot[i]),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=7.5,
                    alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
                )
    
    ax.set_xlabel('Recall (Positive Class)', fontweight='bold')
    ax.set_ylabel('Precision (Positive Class)', fontweight='bold')
    ax.set_title('Precision-Recall Comparison Across Methods and Emotions', 
                fontweight='bold', pad=15)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add diagonal line (perfect precision-recall balance)
    ax.plot([0, 1], [1, 0], 'r--', alpha=0.3, linewidth=1, label='Perfect Balance')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_metric_distribution(metrics_data, save_path=None):
    """
    Figure 6: Distribution of metrics across emotions (box plots).
    """
    methods = list(metrics_data.keys())
    method_labels = _resolve_method_labels(methods)
    metrics_to_plot = ['Accuracy', 'F1_Pos', 'F1_Weighted']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        
        for i, method in enumerate(methods):
            df = metrics_data[method]
            if metric in df.columns:
                values = df[metric].dropna().values
                data_to_plot.append(values)
                labels.append(method_labels[i])
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.85),
                           medianprops=dict(color='red', linewidth=2))
            for patch, method in zip(bp['boxes'], [m for m in methods if metric in metrics_data[m].columns]):
                patch.set_facecolor(_method_color(method))
                patch.set_alpha(0.85)
            
            ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
            ax.set_title(f'Distribution of {metric.replace("_", " ")}', 
                        fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', labelrotation=25)
            for t in ax.get_xticklabels():
                t.set_ha('right')
    
    plt.suptitle('Metric Distributions Across Emotions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(summary_df, save_path=None):
    """
    Create a LaTeX-formatted summary table for academic papers.
    """
    if summary_df is None:
        return
    
    # Format the dataframe for LaTeX
    summary_df_formatted = summary_df.copy()
    summary_df_formatted['avg_accuracy'] = summary_df_formatted['avg_accuracy'].apply(
        lambda x: f"{x:.3f} ± {summary_df_formatted.loc[summary_df_formatted.index[summary_df_formatted['avg_accuracy'] == x], 'std_accuracy'].values[0]:.3f}")
    summary_df_formatted['avg_f1_weighted'] = summary_df_formatted['avg_f1_weighted'].apply(
        lambda x: f"{x:.3f}")
    
    summary_df_formatted = summary_df_formatted[['avg_accuracy', 'avg_f1_weighted', 'n_emotions']]
    summary_df_formatted.columns = ['Accuracy (Mean ± Std)', 'F1-Score (Weighted)', 'N Emotions']
    
    # Save as CSV and print LaTeX
    if save_path:
        summary_df_formatted.to_csv(save_path.replace('.tex', '.csv'))
        print(f"Saved CSV table: {save_path.replace('.tex', '.csv')}")
    
    # Print LaTeX table code
    latex_table = summary_df_formatted.to_latex(
        float_format="%.3f",
        caption="Overall Performance Comparison of Methods",
        label="tab:method_comparison"
    )
    
    tex_path = FIGURES_DIR / 'summary_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table: {tex_path}")
    print("\nLaTeX Table Code:")
    print(latex_table)


def main():
    """Generate all visualizations."""
    parser = argparse.ArgumentParser(description="Generate per-run and cross-run visualizations.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run folder name (under comparison_results) or absolute path to use for figures.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Generate per-run figures for all run folders and create cross-run aggregate CSVs.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="When used with --all-runs, aggregate only the most recent N runs.",
    )
    args = parser.parse_args()

    def _generate_visuals_for_active_run() -> bool:
        print("=" * 60)
        print("Generating Academic Visualizations for LyriSent_Bert")
        print("=" * 60)
        
        # Load data
        print("\n1. Loading metrics data...")
        metrics_data = load_metrics_data()
        summary_df = load_comparison_summary()
        
        if not metrics_data:
            print("Error: No metrics data found for selected run. Please run testing.py first.")
            return False
        
        print(f"   Loaded metrics for: {', '.join(metrics_data.keys())}")
        emotions = infer_emotions(metrics_data)
        print(f"   Inferred evaluated emotions: {emotions}")
        
        # Generate all figures
        print("\n2. Generating visualizations...")
        
        # Figure 1: Method comparison
        if summary_df is not None:
            print("   Creating method comparison chart...")
            plot_method_comparison(
                summary_df, 
                save_path=FIGURES_DIR / '01_method_comparison.png'
            )
        else:
            print("   Warning: methods_comparison_summary.csv missing; skipping method comparison chart.")
        
        # Figure 2: Per-emotion heatmap
        print("   Creating per-emotion performance heatmap...")
        plot_per_emotion_performance(
            metrics_data,
            save_path=FIGURES_DIR / '02_per_emotion_heatmap.png'
        )
        
        # Figure 3: Per-emotion bar chart
        print("   Creating per-emotion accuracy comparison...")
        plot_emotion_comparison_bar(
            metrics_data,
            save_path=FIGURES_DIR / '03_per_emotion_comparison.png'
        )
        
        # Figure 4: Confusion matrices
        print("   Creating confusion matrices...")
        plot_confusion_matrices(
            metrics_data,
            save_path=FIGURES_DIR / '04_confusion_matrices.png'
        )
        
        # Figure 5: Precision-Recall
        print("   Creating precision-recall comparison...")
        plot_precision_recall_comparison(
            metrics_data,
            save_path=FIGURES_DIR / '05_precision_recall.png'
        )
        
        # Figure 6: Metric distribution
        print("   Creating metric distribution plots...")
        plot_metric_distribution(
            metrics_data,
            save_path=FIGURES_DIR / '06_metric_distribution.png'
        )
        
        # Summary table
        print("\n3. Creating summary table...")
        if summary_df is not None:
            create_summary_table(
                summary_df,
                save_path=str(FIGURES_DIR / 'summary_table.tex')
            )
        else:
            print("   Warning: methods_comparison_summary.csv missing; skipping summary table.")
        
        print("\n" + "=" * 60)
        print("Visualization generation complete!")
        print(f"All figures saved to: {FIGURES_DIR}")
        print("=" * 60)
        return True

    if args.run_dir:
        run_dir_input = Path(args.run_dir)
        selected_dir = run_dir_input if run_dir_input.is_absolute() else (RESULTS_DIR / run_dir_input)
        if selected_dir.exists():
            set_active_results_dir(selected_dir)
            print(f"Using selected run directory: {ACTIVE_RESULTS_DIR}")
        else:
            print(f"Warning: run directory not found: {selected_dir}. Falling back to auto-detected run.")

    if args.all_runs:
        run_dirs = list_run_dirs()
        if args.last_n and args.last_n > 0:
            run_dirs = run_dirs[-args.last_n:]
        print(f"Processing {len(run_dirs)} run folder(s) for per-run figures...")
        for run_dir in run_dirs:
            set_active_results_dir(run_dir)
            print(f"\n--- Run: {run_dir.name} ---")
            _generate_visuals_for_active_run()
        print(f"\nCreating cross-run comparison from {len(run_dirs)} run(s)...")
        create_cross_run_comparison(run_dirs, RESULTS_DIR / "cross_run_analysis")
        return

    _generate_visuals_for_active_run()


if __name__ == "__main__":
    main()
