import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, average_precision_score
from bert_class import predict_emotions_bert, bert_load_success, EMOTION_LABELS as BERT_EMOTION_LABELS
from nb_class import predict_emotions, preprocess_lyrics, emotions_list as NB_EMOTION_LABELS
from GPT.OpenAI import check_sentiment_openai_with_meta
import utils
import numpy as np
import io
import csv
import os
import json
import time
from typing import Optional

from config import (
    RADA_ANNOTATION_CSV,
    SENTIMENT_COMPARISON_CSV,
    EMOTION_LABELS,
    RESULTS_DIR
)

RANDOM_STATE = 42
PROBABILITY_METHODS = {"NB", "BERT"}
GPT_USE_CLEANED_LYRICS = True
THRESHOLD_VAL_FRACTION = 0.5
MODEL_PRICING_PER_1K_USD = {
    "gpt-4": {"input": 0.03, "output": 0.06},
}


def _ordered_intersection(order, candidates: set[str]) -> list[str]:
    return [x for x in order if x in candidates]


def _truth_emotions_from_df(df_truth: pd.DataFrame) -> list[str]:
    truth_cols_lower = {c.lower(): c for c in df_truth.columns}
    available = {c.lower() for c in df_truth.columns}
    # Prefer config order; filter to truth columns that exist (case-insensitive)
    ordered = [e for e in EMOTION_LABELS if e.lower() in available]
    # Return canonical casing (from config), but we will map to actual column names later.
    return ordered


def split_eval_indices(results_df: pd.DataFrame, *, val_fraction: float = THRESHOLD_VAL_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into threshold-tuning and final-evaluation subsets."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx_all = results_df.index.to_numpy()
    rng.shuffle(idx_all)
    split = int(len(idx_all) * val_fraction)
    idx_val = idx_all[:split]
    idx_test = idx_all[split:]
    return idx_val, idx_test


def _estimate_gpt_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1K_USD.get(str(model_name), None)
    if not pricing:
        return float("nan")
    in_rate = float(pricing.get("input", 0.0))
    out_rate = float(pricing.get("output", 0.0))
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate


def save_runtime_and_cost_logs(events: list[dict], output_dir: str) -> None:
    if not events:
        return

    events_df = pd.DataFrame(events)
    events_path = os.path.join(output_dir, "runtime_cost_events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"Saved runtime/cost events to {events_path}")

    summary_rows = []
    for method, g in events_df.groupby("method"):
        summary_rows.append({
            "method": method,
            "n_calls": int(len(g)),
            "total_elapsed_sec": float(g["elapsed_sec"].sum()),
            "avg_elapsed_sec": float(g["elapsed_sec"].mean()),
            "total_prompt_tokens": int(g["prompt_tokens"].sum()),
            "total_completion_tokens": int(g["completion_tokens"].sum()),
            "total_tokens": int(g["total_tokens"].sum()),
            "estimated_total_cost_usd": float(g["estimated_cost_usd"].sum(skipna=True)),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="method")
    summary_path = os.path.join(output_dir, "runtime_cost_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved runtime/cost summary to {summary_path}")

# 1. Load data
df_rada = pd.read_csv(RADA_ANNOTATION_CSV)
df_predefined = pd.read_csv(SENTIMENT_COMPARISON_CSV)

print("\nStructure of df_rada:")
print(df_rada.info())
print("\nFirst few rows of df_rada:")
print(df_rada.head())
print("\nColumns in df_rada:")
print(df_rada.columns.tolist())

# Initialize dictionaries to store results
nb_results = {}
bert_results = {}
openai_results = {}

# Determine evaluation emotion set driven by truth (and ordered)
eval_emotions = _truth_emotions_from_df(df_rada)
if not eval_emotions:
    raise ValueError(
        "No emotion columns found in truth dataframe. "
        f"Expected something like: {EMOTION_LABELS}. Got columns: {df_rada.columns.tolist()}"
    )
print(f"Evaluation emotions (truth-driven, ordered): {eval_emotions}")


# 2. Test methods on RadaNewAnnotation
def test_methods_on_rada(df, runtime_events: list[dict]):
    results_data = []

    for index, row in df.iterrows():
        lyrics = row['text']
        song_name = row['Song']
        print(f"\nProcessing song {index + 1}/{len(df)}: {song_name}")

        # Store the results in a dictionary for this song
        song_results = {
            'index': index,
            'song_name': song_name
        }

        # Naive Bayes
        try:
            nb_start = time.perf_counter()
            preprocessed_lyrics = preprocess_lyrics(lyrics)
            nb_predictions = predict_emotions(preprocessed_lyrics)
            nb_elapsed = time.perf_counter() - nb_start
            nb_results[song_name] = nb_predictions
            runtime_events.append({
                "song_name": song_name,
                "method": "NaiveBayes",
                "elapsed_sec": nb_elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "model": "",
                "response_format": "",
            })
            # Add NB results to song_results
            for emotion in eval_emotions:
                song_results[f'NB_{emotion}'] = nb_predictions.get(emotion, np.nan)
        except Exception as e:
            print(f"Error in NB prediction for {song_name}: {e}")
            nb_results[song_name] = {emotion: np.nan for emotion in eval_emotions}
            for emotion in eval_emotions:
                song_results[f'NB_{emotion}'] = np.nan

        # BERT
        if bert_load_success:
            try:
                bert_start = time.perf_counter()
                bert_predictions = predict_emotions_bert(lyrics)
                bert_elapsed = time.perf_counter() - bert_start
                bert_results[song_name] = bert_predictions
                runtime_events.append({
                    "song_name": song_name,
                    "method": "BERT",
                    "elapsed_sec": bert_elapsed,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "model": "",
                    "response_format": "",
                })
                # Add BERT results to song_results
                for emotion in eval_emotions:
                    song_results[f'BERT_{emotion}'] = bert_predictions.get(emotion, np.nan)
            except Exception as e:
                print(f"Error in BERT prediction for {song_name}: {e}")
                bert_results[song_name] = {emotion: np.nan for emotion in eval_emotions}
                for emotion in eval_emotions:
                    song_results[f'BERT_{emotion}'] = np.nan
        else:
            bert_results[song_name] = {emotion: np.nan for emotion in eval_emotions}
            for emotion in eval_emotions:
                song_results[f'BERT_{emotion}'] = np.nan

        # OpenAI
        try:
            gpt_lyrics = utils.clean_lyrics(lyrics) if GPT_USE_CLEANED_LYRICS else lyrics
            openai_start = time.perf_counter()
            openai_payload = check_sentiment_openai_with_meta({'lyrics': gpt_lyrics})
            openai_elapsed = time.perf_counter() - openai_start
            openai_output = openai_payload.get("content", "")
            usage = openai_payload.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)
            model_name = str(openai_payload.get("model", "unknown"))
            estimated_cost_usd = _estimate_gpt_cost_usd(model_name, prompt_tokens, completion_tokens)
            runtime_events.append({
                "song_name": song_name,
                "method": "OpenAI",
                "elapsed_sec": openai_elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "model": model_name,
                "response_format": openai_payload.get("format", "unknown"),
            })
            print(f"Raw OpenAI Output length: {len(openai_output) if openai_output else 0} characters")

            openai_predictions = parse_openai_output(openai_output, song_name)
            openai_results[song_name] = openai_predictions
            song_results["OpenAI_prompt_tokens"] = prompt_tokens
            song_results["OpenAI_completion_tokens"] = completion_tokens
            song_results["OpenAI_total_tokens"] = total_tokens
            song_results["OpenAI_estimated_cost_usd"] = estimated_cost_usd

            # Add OpenAI results to song_results
            for emotion in eval_emotions:
                song_results[f'OpenAI_{emotion}'] = openai_predictions.get(emotion, np.nan)

        except Exception as e:
            print(f"Error in OpenAI prediction for {song_name}: {e}")
            openai_results[song_name] = {emotion: np.nan for emotion in eval_emotions}
            for emotion in eval_emotions:
                song_results[f'OpenAI_{emotion}'] = np.nan

        results_data.append(song_results)

    return nb_results, bert_results, openai_results, results_data


def parse_openai_output(openai_output, song_name):
    """Parse OpenAI output more robustly"""
    openai_predictions = {}

    if not openai_output:
        print(f"No OpenAI output for {song_name}")
        return {emotion: np.nan for emotion in eval_emotions}

    # 1) Try JSON first (preferred format)
    try:
        obj = json.loads(openai_output)
        emotions_obj = obj.get("emotions", {})
        if isinstance(emotions_obj, dict) and emotions_obj:
            for emotion in EMOTION_LABELS:
                v = emotions_obj.get(emotion, None)
                if v in (0, 1, "0", "1"):
                    openai_predictions[emotion] = int(v)
                else:
                    openai_predictions[emotion] = np.nan
            return openai_predictions
    except Exception:
        pass

    gpt_emotions_order = EMOTION_LABELS
    expected_gpt_cols = 3 + len(gpt_emotions_order)

    # Try to find the data line
    raw_lines = openai_output.strip().splitlines()
    data_line_string = None

    for i, line in enumerate(raw_lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Skip header lines
        if any(keyword in stripped_line for keyword in ["Song Name", "Artists", "Lyrics", "Joy,Trust"]):
            continue

        # Try to parse as CSV
        try:
            string_io_line = io.StringIO(stripped_line)
            reader_line = csv.reader(string_io_line)
            parsed_row = next(reader_line)

            # Check if this could be a data line
            if len(parsed_row) >= expected_gpt_cols - 2:
                # Additional check: see if the emotion columns contain 0s and 1s
                emotion_values = parsed_row[3:3 + len(gpt_emotions_order)]
                if all(val.strip() in ['0', '1', ''] for val in emotion_values):
                    data_line_string = stripped_line
                    break
        except Exception:
            continue

    if data_line_string:
        try:
            string_io = io.StringIO(data_line_string)
            reader = csv.reader(string_io)
            gpt_sentiment_list_raw = next(reader)

            # Extract emotion values
            for i, emotion in enumerate(gpt_emotions_order):
                if 3 + i < len(gpt_sentiment_list_raw):
                    value_str = gpt_sentiment_list_raw[3 + i].strip().strip('"\'')
                    if value_str in {'0', '1'}:
                        openai_predictions[emotion] = int(value_str)
                    else:
                        openai_predictions[emotion] = np.nan
                else:
                    openai_predictions[emotion] = np.nan
        except Exception as e:
            print(f"Error parsing data line for {song_name}: {e}")
            return {emotion: np.nan for emotion in eval_emotions}
    else:
        print(f"Could not find data line in OpenAI output for {song_name}")
        return {emotion: np.nan for emotion in eval_emotions}

    return openai_predictions


# Run the tests
print("\nTesting methods on RadaNewAnnotation dataset...")
runtime_events = []
nb_results, bert_results, openai_results, results_data = test_methods_on_rada(df_rada, runtime_events)

# Create a comprehensive results DataFrame
results_df = pd.DataFrame(results_data)
results_df = results_df.set_index('index')  # Use the original index from df_rada

# Now add the ground truth labels from df_rada
print("\nAdding ground truth labels from df_rada...")
rada_cols_lower = {col.lower(): col for col in df_rada.columns}
for emotion in eval_emotions:
    # Find the corresponding column in df_rada (case-insensitive)
    if emotion.lower() in rada_cols_lower:
        actual_col = rada_cols_lower[emotion.lower()]
        results_df[f'Truth_{emotion}'] = df_rada[actual_col].values
        print(f"Added truth column for {emotion} from df_rada column '{actual_col}'")
    else:
        print(f"Warning: Could not find column for emotion '{emotion}' in df_rada")
        results_df[f'Truth_{emotion}'] = np.nan

# Save the comprehensive results
output_dir = RESULTS_DIR
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(os.path.join(output_dir, 'rada_all_predictions.csv'))
print(f"\nSaved all predictions to {os.path.join(output_dir, 'rada_all_predictions.csv')}")
save_runtime_and_cost_logs(runtime_events, output_dir)


# 3. Performance evaluation function
def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Simple grid search; fast enough for small datasets.
    grid = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def tune_thresholds(results_df: pd.DataFrame, method_prefix: str, emotion_labels: list[str]):
    """
    Tune per-emotion thresholds on a holdout split of the Rada dataset.
    Note: this is for more rigorous reporting than a fixed 0.5 threshold.
    """
    thresholds = {}
    for emotion in emotion_labels:
        pred_col = f"{method_prefix}_{emotion}"
        truth_col = f"Truth_{emotion}"
        if pred_col not in results_df.columns or truth_col not in results_df.columns:
            continue

        df_sub = results_df[[pred_col, truth_col]].dropna()
        if len(df_sub) < 5:
            thresholds[emotion] = 0.5
            continue

        y_true = df_sub[truth_col].astype(int).to_numpy()
        y_prob = df_sub[pred_col].astype(float).to_numpy()
        thresholds[emotion] = _optimal_threshold(y_true, y_prob)

    return thresholds


def evaluate_performance(results_df, method_prefix, emotion_labels, *, thresholds: Optional[dict] = None):
    """Evaluate performance of a method against ground truth."""
    all_metrics = {}

    for emotion in emotion_labels:
        pred_col = f'{method_prefix}_{emotion}'
        truth_col = f'Truth_{emotion}'

        if pred_col not in results_df.columns or truth_col not in results_df.columns:
            print(f"Missing columns for {method_prefix} - {emotion}")
            continue

        # Get predictions and truth, dropping NaN values
        mask = results_df[[pred_col, truth_col]].notna().all(axis=1)
        predictions = results_df.loc[mask, pred_col]
        truth = results_df.loc[mask, truth_col]

        if len(predictions) < 2:
            print(f"Not enough data for {method_prefix} - {emotion}")
            continue

        # Convert to binary predictions based on threshold
        if method_prefix in PROBABILITY_METHODS:
            t = 0.5 if not thresholds else float(thresholds.get(emotion, 0.5))
            pred_binary = (predictions.astype(float) >= t).astype(int)
        else:
            # OpenAI already returns 0/1
            pred_binary = predictions.astype(int)

        truth_binary = truth.astype(int)

        try:
            # Calculate metrics
            accuracy = accuracy_score(truth_binary, pred_binary)
            report = classification_report(truth_binary, pred_binary, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(truth_binary, pred_binary)
            f1_pos = report.get('1', {}).get('f1-score', np.nan)

            # Optional probability metrics for NB/BERT
            auc_roc = np.nan
            pr_auc = np.nan
            if method_prefix in PROBABILITY_METHODS:
                try:
                    auc_roc = roc_auc_score(truth_binary, predictions.astype(float))
                except ValueError:
                    auc_roc = np.nan
                try:
                    pr_auc = average_precision_score(truth_binary, predictions.astype(float))
                except ValueError:
                    pr_auc = np.nan

            all_metrics[emotion] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'n_samples': len(predictions),
                'f1_pos': f1_pos,
                'auc_roc': auc_roc,
                'pr_auc': pr_auc
            }

            print(f"{method_prefix} - {emotion}: Accuracy = {accuracy:.3f}, N = {len(predictions)}")

        except Exception as e:
            print(f"Error calculating metrics for {method_prefix} - {emotion}: {e}")

    return all_metrics


# Evaluate each method
print("\n" + "=" * 50)
print("PERFORMANCE EVALUATION")
print("=" * 50)

tune = True
idx_threshold_val, idx_final_test = split_eval_indices(results_df, val_fraction=THRESHOLD_VAL_FRACTION)
results_df_threshold_val = results_df.loc[idx_threshold_val].copy()
results_df_final_test = results_df.loc[idx_final_test].copy()
print(f"Threshold tuning split size: {len(results_df_threshold_val)}")
print(f"Final evaluation split size: {len(results_df_final_test)}")

nb_thresholds = tune_thresholds(results_df_threshold_val, 'NB', eval_emotions) if tune else None
bert_thresholds = tune_thresholds(results_df_threshold_val, 'BERT', eval_emotions) if (tune and bert_load_success) else None

thresholds_payload = {
    "random_state": RANDOM_STATE,
    "threshold_val_fraction": THRESHOLD_VAL_FRACTION,
    "nb_thresholds": nb_thresholds or {},
    "bert_thresholds": bert_thresholds or {},
}
thresholds_path = os.path.join(output_dir, "thresholds_used.json")
with open(thresholds_path, "w", encoding="utf-8") as f:
    json.dump(thresholds_payload, f, indent=2)
print(f"Saved thresholds to {thresholds_path}")

nb_metrics = evaluate_performance(results_df_final_test, 'NB', eval_emotions, thresholds=nb_thresholds)
bert_metrics = evaluate_performance(results_df_final_test, 'BERT', eval_emotions, thresholds=bert_thresholds) if bert_load_success else {}
openai_metrics = evaluate_performance(results_df_final_test, 'OpenAI', eval_emotions)


# Save detailed metrics
def save_detailed_metrics(metrics, method_name, output_dir):
    """Save detailed metrics to a CSV file"""
    if not metrics:
        print(f"No metrics to save for {method_name}")
        return

    # Create a summary DataFrame
    summary_data = []
    for emotion, emotion_metrics in metrics.items():
        row = {
            'Emotion': emotion,
            'Accuracy': emotion_metrics['accuracy'],
            'N_Samples': emotion_metrics['n_samples'],
            'AUC_ROC': emotion_metrics.get('auc_roc', np.nan),
            'PR_AUC': emotion_metrics.get('pr_auc', np.nan)
        }

        # Add classification report metrics
        if 'classification_report' in emotion_metrics:
            report = emotion_metrics['classification_report']
            if '1' in report:  # Positive class
                row['Precision_Pos'] = report['1']['precision']
                row['Recall_Pos'] = report['1']['recall']
                row['F1_Pos'] = report['1']['f1-score']
            if '0' in report:  # Negative class
                row['Precision_Neg'] = report['0']['precision']
                row['Recall_Neg'] = report['0']['recall']
                row['F1_Neg'] = report['0']['f1-score']
            if 'weighted avg' in report:
                row['F1_Weighted'] = report['weighted avg']['f1-score']

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    filename = os.path.join(output_dir, f'metrics_summary_{method_name.lower()}.csv')
    summary_df.to_csv(filename, index=False)
    print(f"Saved {method_name} metrics summary to {filename}")


# Save all metrics
save_detailed_metrics(nb_metrics, 'NaiveBayes', output_dir)
if bert_load_success:
    save_detailed_metrics(bert_metrics, 'BERT', output_dir)
save_detailed_metrics(openai_metrics, 'OpenAI', output_dir)

# 4. Create a comparison summary
print("\n" + "=" * 50)
print("CREATING COMPARISON SUMMARY")
print("=" * 50)


# Calculate average metrics across all emotions for each method
def calculate_average_metrics(metrics):
    if not metrics:
        return {}

    accuracies = [m['accuracy'] for m in metrics.values()]
    f1_scores = []
    f1_pos_scores = []
    aucs = []
    pr_aucs = []
    for m in metrics.values():
        if 'classification_report' in m and 'weighted avg' in m['classification_report']:
            f1_scores.append(m['classification_report']['weighted avg']['f1-score'])
        if 'f1_pos' in m and m['f1_pos'] is not None:
            f1_pos_scores.append(m['f1_pos'])
        if 'auc_roc' in m and not np.isnan(m['auc_roc']):
            aucs.append(m['auc_roc'])
        if 'pr_auc' in m and not np.isnan(m['pr_auc']):
            pr_aucs.append(m['pr_auc'])

    return {
        'avg_accuracy': np.mean(accuracies) if accuracies else np.nan,
        'std_accuracy': np.std(accuracies) if accuracies else np.nan,
        'avg_f1_weighted': np.mean(f1_scores) if f1_scores else np.nan,
        'avg_f1_pos': np.mean(f1_pos_scores) if f1_pos_scores else np.nan,
        'avg_auc_roc': np.mean(aucs) if aucs else np.nan,
        'avg_pr_auc': np.mean(pr_aucs) if pr_aucs else np.nan,
        'n_emotions': len(metrics)
    }


# Create summary
summary = {
    'NaiveBayes': calculate_average_metrics(nb_metrics),
    'BERT': calculate_average_metrics(bert_metrics) if bert_load_success else {},
    'OpenAI': calculate_average_metrics(openai_metrics)
}

# Save summary
summary_df = pd.DataFrame(summary).T
summary_df.to_csv(os.path.join(output_dir, 'methods_comparison_summary.csv'))
print("\nMethods Comparison Summary:")
print(summary_df)

print("\n" + "=" * 50)
print("All evaluations completed successfully!")
print(f"Results saved in '{output_dir}' directory")
print("=" * 50)