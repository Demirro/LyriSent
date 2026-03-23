import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create output directory for figures
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def load_metrics_data():
    """Load all metrics summary files."""
    methods = ['NaiveBayes', 'BERT', 'OpenAI']
    metrics_data = {}
    
    for method in methods:
        file_path = RESULTS_DIR / f'metrics_summary_{method.lower()}.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            metrics_data[method] = df
        else:
            print(f"Warning: {file_path} not found")
    
    return metrics_data


def load_comparison_summary():
    file_path = RESULTS_DIR / 'methods_comparison_summary.csv'
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0)
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


def plot_method_comparison(summary_df, save_path=None):
    """
    Figure 1: Overall method comparison (Accuracy and F1-Score).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = summary_df.index.tolist()
    x_pos = np.arange(len(methods))
    width = 0.6
    
    # Accuracy comparison
    accuracies = summary_df['avg_accuracy'].values
    acc_stds = summary_df['std_accuracy'].values
    
    bars1 = ax1.bar(x_pos, accuracies, width, yerr=acc_stds, 
                    capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Method', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(a) Average Accuracy Comparison', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=0)
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
    
    bars2 = ax2.bar(x_pos, f1_scores, width, alpha=0.8, 
                    edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('Weighted F1-Score', fontweight='bold')
    ax2.set_title('(b) Average F1-Score Comparison', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=0)
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
    metrics_to_plot = ['Accuracy', 'Precision_Pos', 'Recall_Pos', 'F1_Pos']
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
                                 columns=methods)
        
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
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
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
        bars = ax.bar(x + offset, accuracies, width, label=method, 
                     alpha=0.8, edgecolor='black', linewidth=0.8)
        
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
    predictions_file = RESULTS_DIR / 'rada_all_predictions.csv'
    if not predictions_file.exists():
        print("Warning: Full predictions file not found. Skipping confusion matrices.")
        return
    
    df = pd.read_csv(predictions_file)
    emotions = infer_emotions(metrics_data, predictions_df=df)
    methods = ['NB', 'BERT', 'OpenAI']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
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
            ax.set_title(f'{method}', fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method}', fontweight='bold')
    
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
    emotions_all = infer_emotions(metrics_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for method, color in zip(methods, colors):
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
            ax.scatter(recalls, precisions, s=150, alpha=0.7, 
                      label=method, color=color, edgecolors='black', linewidth=1.5)
            
            # Add emotion labels
            for i, emotion in enumerate(emotions):
                ax.annotate(emotion, (recalls[i], precisions[i]), 
                          fontsize=8, alpha=0.8)
    
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
    metrics_to_plot = ['Accuracy', 'F1_Pos', 'F1_Weighted']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        
        for method in methods:
            df = metrics_data[method]
            if metric in df.columns:
                values = df[metric].dropna().values
                data_to_plot.append(values)
                labels.append(method)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            
            ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
            ax.set_title(f'Distribution of {metric.replace("_", " ")}', 
                        fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1])
    
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
    print("=" * 60)
    print("Generating Academic Visualizations for LyriSent_Bert")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading metrics data...")
    metrics_data = load_metrics_data()
    summary_df = load_comparison_summary()
    
    if not metrics_data:
        print("Error: No metrics data found. Please run testing.py first.")
        return
    
    print(f"   Loaded metrics for: {', '.join(metrics_data.keys())}")
    emotions = infer_emotions(metrics_data)
    print(f"   Inferred evaluated emotions: {emotions}")
    
    # Generate all figures
    print("\n2. Generating visualizations...")
    
    # Figure 1: Method comparison
    print("   Creating method comparison chart...")
    plot_method_comparison(
        summary_df, 
        save_path=FIGURES_DIR / '01_method_comparison.png'
    )
    
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
    create_summary_table(
        summary_df,
        save_path=str(FIGURES_DIR / 'summary_table.tex')
    )
    
    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
