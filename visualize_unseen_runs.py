import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import EMOTION_LABELS, OPENAI_PROMPTING_MODES, RESULTS_DIR


def _list_unseen_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.endswith("_unseen")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _resolve_run_dir(run_dir_arg: str) -> Path:
    if run_dir_arg:
        candidate = Path(run_dir_arg)
        if not candidate.is_absolute():
            candidate = RESULTS_DIR / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Run directory not found: {candidate}")
        return candidate

    unseen_runs = _list_unseen_runs()
    if not unseen_runs:
        raise FileNotFoundError("No unseen run folders found in comparison_results.")
    return unseen_runs[0]


def _method_columns(method_name: str) -> list[str]:
    return [f"{method_name}_{emotion}" for emotion in EMOTION_LABELS]


def _build_method_map(df: pd.DataFrame) -> dict[str, list[str]]:
    method_map: dict[str, list[str]] = {
        "NB": [col for col in _method_columns("NB") if col in df.columns],
        "BERT": [col for col in _method_columns("BERT") if col in df.columns],
    }
    for mode in OPENAI_PROMPTING_MODES:
        prefix = f"OpenAI_{mode}"
        cols = [f"{prefix}_{emotion}" for emotion in EMOTION_LABELS if f"{prefix}_{emotion}" in df.columns]
        if cols:
            method_map[prefix] = cols
    return method_map


def _song_label(row: pd.Series) -> str:
    song = str(row.get("Song", "Unknown Song"))
    artist = str(row.get("Artist", "")).strip()
    return f"{song} - {artist}" if artist else song


def _song_long_df(df: pd.DataFrame, row_idx: int, method_map: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for method, cols in method_map.items():
        for emotion in EMOTION_LABELS:
            col = f"{method}_{emotion}"
            if col not in cols:
                continue
            rows.append(
                {
                    "Method": method,
                    "Emotion": emotion,
                    "Score": pd.to_numeric(df.iloc[row_idx][col], errors="coerce"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Emotion"] = pd.Categorical(out["Emotion"], categories=EMOTION_LABELS, ordered=True)
    return out.sort_values(["Emotion", "Method"])


def _safe_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:80] or "song"


def _plot_single_song_emotions(song_df: pd.DataFrame, label: str, out_path: Path) -> None:
    if song_df.empty:
        return
    plt.figure(figsize=(12, 5.5))
    ax = sns.barplot(data=song_df, x="Emotion", y="Score", hue="Method", errorbar=None)
    ax.set_title(f"Per-song emotion outputs: {label}")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Model output")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_single_song_heatmap(song_df: pd.DataFrame, label: str, out_path: Path) -> None:
    if song_df.empty:
        return
    pivot = song_df.pivot(index="Method", columns="Emotion", values="Score")
    pivot = pivot.reindex(columns=EMOTION_LABELS)
    plt.figure(figsize=(10, max(2.8, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
    plt.title(f"Per-song emotion heatmap: {label}")
    plt.xlabel("Emotion")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def visualize_unseen_run(run_dir: Path, style: str = "heatmap") -> None:
    predictions_path = run_dir / "unseen_song_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing file: {predictions_path}")

    figures_dir = run_dir / "figures_unseen"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    method_map = _build_method_map(df)
    if not method_map:
        raise ValueError("No recognized prediction columns found in unseen_song_predictions.csv")

    long_rows = []
    for i in range(len(df)):
        label = _song_label(df.iloc[i])
        song_df = _song_long_df(df, i, method_map)
        if song_df.empty:
            continue
        song_df = song_df.assign(Song=label)
        long_rows.append(song_df)
        out_name = f"{i + 1:02d}_{_safe_filename(label)}_per_song_{style}.png"
        out_path = figures_dir / out_name
        if style == "bars":
            _plot_single_song_emotions(song_df, label, out_path)
        else:
            _plot_single_song_heatmap(song_df, label, out_path)

    print(f"Visualized unseen run: {run_dir}")
    print(f"Saved files in: {figures_dir}")
    if long_rows:
        combined_long = pd.concat(long_rows, ignore_index=True)
        long_csv = figures_dir / "per_song_emotions_long.csv"
        combined_long.to_csv(long_csv, index=False)
        print(f"- {long_csv.name}")
    print(f"- one per-song {style} plot per track")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize unseen-song runs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run folder name under comparison_results or absolute path. Defaults to latest *_unseen run.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="heatmap",
        choices=["heatmap", "bars"],
        help="Visualization style for per-song plots. Default: heatmap",
    )
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    visualize_unseen_run(run_dir, style=args.style)


if __name__ == "__main__":
    main()
