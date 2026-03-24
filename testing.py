import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, average_precision_score
from bert_class import predict_emotions_bert, bert_load_success, EMOTION_LABELS as BERT_EMOTION_LABELS
from nb_class import predict_emotions, preprocess_lyrics, emotions_list as NB_EMOTION_LABELS
from GPT.OpenAI import check_sentiment_openai
import utils
import numpy as np
import io
import csv
import os
import json
import time
import re
from datetime import datetime
from typing import Optional
from sklearn.model_selection import train_test_split

from config import (
    RADA_ANNOTATION_CSV,
    SENTIMENT_COMPARISON_CSV,
    EMOTION_LABELS,
    RESULTS_DIR,
    OPENAI_MODEL,
    OPENAI_PROMPTING_MODES,
)

RANDOM_STATE = 42
PROBABILITY_METHODS = {"NB", "BERT"}
USE_CLEANED_LYRICS_FOR_ALL_METHODS = True
THRESHOLD_TUNE_FRACTION = 0.5


def _ordered_intersection(order, candidates: set[str]) -> list[str]:
    return [x for x in order if x in candidates]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-").lower() or "unknown"


def _truth_emotions_from_df(df_truth: pd.DataFrame) -> list[str]:
    truth_cols_lower = {c.lower(): c for c in df_truth.columns}
    available = {c.lower() for c in df_truth.columns}
    # Keep config order and only use available truth columns.
    ordered = [e for e in EMOTION_LABELS if e.lower() in available]
    # Keep canonical names from config.
    return ordered

# Load data
df_rada = pd.read_csv(RADA_ANNOTATION_CSV)
df_predefined = pd.read_csv(SENTIMENT_COMPARISON_CSV)

print("\nStructure of df_rada:")
print(df_rada.info())
print("\nFirst few rows of df_rada:")
print(df_rada.head())
print("\nColumns in df_rada:")
print(df_rada.columns.tolist())

# Result containers
nb_results = {}
bert_results = {}
openai_results = {}
runtime_events = []
openai_parse_failures = []
OPENAI_MODES = tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)

# Evaluation labels from truth columns
eval_emotions = _truth_emotions_from_df(df_rada)
if not eval_emotions:
    raise ValueError(
        "No emotion columns found in truth dataframe. "
        f"Expected something like: {EMOTION_LABELS}. Got columns: {df_rada.columns.tolist()}"
    )
print(f"Evaluation emotions (truth-driven, ordered): {eval_emotions}")


# Run methods on RadaNewAnnotation
def test_methods_on_rada(df):
    results_data = []
    method_timers = {"NB": 0.0, "BERT": 0.0}
    openai_mode_keys = [f"OpenAI_{mode}" for mode in OPENAI_MODES]
    for mode_key in openai_mode_keys:
        method_timers[mode_key] = 0.0
    gpt_usage_by_mode = {
        mode: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "model": OPENAI_MODEL,
            "prompting_mode": mode,
            "structured_output_supported": False,
            "structured_output_requested_any": False,
            "json_success_count": 0,
            "parse_fail_count": 0,
            "avg_attempts": 0.0,
            "_attempts_sum": 0,
        }
        for mode in OPENAI_MODES
    }

    for index, row in df.iterrows():
        lyrics = row['text']
        lyrics_for_models = utils.clean_lyrics(lyrics) if USE_CLEANED_LYRICS_FOR_ALL_METHODS else lyrics
        song_name = row['Song']
        print(f"\nProcessing song {index + 1}/{len(df)}: {song_name}")

        # Per-song output row
        song_results = {
            'index': index,
            'song_name': song_name
        }

        # Naive Bayes
        try:
            t0 = time.perf_counter()
            preprocessed_lyrics = preprocess_lyrics(lyrics_for_models)
            nb_predictions = predict_emotions(preprocessed_lyrics)
            method_timers["NB"] += time.perf_counter() - t0
            nb_results[song_name] = nb_predictions
            # Write NB outputs
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
                t0 = time.perf_counter()
                bert_predictions = predict_emotions_bert(lyrics_for_models)
                method_timers["BERT"] += time.perf_counter() - t0
                bert_results[song_name] = bert_predictions
                # Write BERT outputs
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

        # OpenAI variants
        for mode in OPENAI_MODES:
            mode_method_key = f"OpenAI_{mode}"
            mode_col_prefix = f"OpenAI_{mode}"
            mode_usage = gpt_usage_by_mode[mode]
            try:
                t0 = time.perf_counter()
                openai_output, openai_meta = check_sentiment_openai(
                    {'lyrics': lyrics_for_models},
                    return_metadata=True,
                    prompting_mode=mode,
                )
                openai_elapsed = time.perf_counter() - t0
                method_timers[mode_method_key] += openai_elapsed
                print(
                    f"Raw OpenAI Output length [{mode}]: "
                    f"{len(openai_output) if openai_output else 0} characters"
                )

                openai_predictions = parse_openai_output(openai_output, f"{song_name} [{mode}]")
                openai_results.setdefault(mode, {})[song_name] = openai_predictions
                mode_usage["prompt_tokens"] += int(openai_meta.get("prompt_tokens", 0) or 0)
                mode_usage["completion_tokens"] += int(openai_meta.get("completion_tokens", 0) or 0)
                mode_usage["total_tokens"] += int(openai_meta.get("total_tokens", 0) or 0)
                mode_usage["estimated_cost_usd"] += float(openai_meta.get("estimated_cost_usd", 0.0) or 0.0)
                runtime_events.append({
                    "method": "OpenAI",
                    "method_variant": mode_method_key,
                    "song_name": song_name,
                    "elapsed_sec": openai_elapsed,
                    "model": str(openai_meta.get("model", OPENAI_MODEL)),
                    "prompting_mode": mode,
                    "prompt_tokens": int(openai_meta.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(openai_meta.get("completion_tokens", 0) or 0),
                    "total_tokens": int(openai_meta.get("total_tokens", 0) or 0),
                    "estimated_cost_usd": float(openai_meta.get("estimated_cost_usd", 0.0) or 0.0),
                    "response_format": str(openai_meta.get("format", "")),
                    "attempts": int(openai_meta.get("attempts", 0) or 0),
                    "structured_output_requested": bool(openai_meta.get("structured_output_requested", False)),
                    "structured_output_supported": bool(openai_meta.get("structured_output_supported", False)),
                })
                mode_usage["structured_output_supported"] = bool(openai_meta.get("structured_output_supported", False))
                mode_usage["structured_output_requested_any"] = (
                    mode_usage["structured_output_requested_any"]
                    or bool(openai_meta.get("structured_output_requested", False))
                )
                mode_usage["json_success_count"] += int(str(openai_meta.get("format", "")) == "json")
                mode_usage["_attempts_sum"] += int(openai_meta.get("attempts", 0) or 0)

                # Log raw response when parsing fails.
                if all(pd.isna(openai_predictions.get(emotion, np.nan)) for emotion in eval_emotions):
                    preview = (openai_output or "")[:800].replace("\n", "\\n")
                    print(
                        f"[OpenAI parse fail] mode={mode} song={song_name} "
                        f"format={openai_meta.get('format', '')} len={len(openai_output or '')}"
                    )
                    print(f"[OpenAI parse fail] raw preview: {preview}")
                    openai_parse_failures.append({
                        "song_name": song_name,
                        "prompting_mode": mode,
                        "response_format": str(openai_meta.get("format", "")),
                        "response_length": int(len(openai_output or "")),
                        "attempts": int(openai_meta.get("attempts", 0) or 0),
                        "prompt_tokens": int(openai_meta.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(openai_meta.get("completion_tokens", 0) or 0),
                        "total_tokens": int(openai_meta.get("total_tokens", 0) or 0),
                        "estimated_cost_usd": float(openai_meta.get("estimated_cost_usd", 0.0) or 0.0),
                        "raw_output": openai_output or "",
                    })
                    mode_usage["parse_fail_count"] += 1

                # Write OpenAI outputs
                for emotion in eval_emotions:
                    song_results[f'{mode_col_prefix}_{emotion}'] = openai_predictions.get(emotion, np.nan)

            except Exception as e:
                print(f"Error in OpenAI prediction ({mode}) for {song_name}: {e}")
                openai_results.setdefault(mode, {})[song_name] = {emotion: np.nan for emotion in eval_emotions}
                for emotion in eval_emotions:
                    song_results[f'{mode_col_prefix}_{emotion}'] = np.nan

        results_data.append(song_results)

    if len(df) > 0:
        for mode in OPENAI_MODES:
            mode_usage = gpt_usage_by_mode[mode]
            mode_usage["avg_attempts"] = float(mode_usage["_attempts_sum"]) / float(len(df))
            mode_usage.pop("_attempts_sum", None)

    return nb_results, bert_results, openai_results, results_data, method_timers, gpt_usage_by_mode


def parse_openai_output(openai_output, song_name):
    """Parse OpenAI output more robustly"""
    openai_predictions = {}

    if not openai_output:
        print(f"No OpenAI output for {song_name}")
        return {emotion: np.nan for emotion in eval_emotions}

    # Handle responses wrapped as a quoted CSV blob.
    normalized_output = openai_output.strip()
    if (
        len(normalized_output) >= 2
        and normalized_output[0] == '"'
        and normalized_output[-1] == '"'
        and ("\n" in normalized_output or "\\n" in normalized_output)
    ):
        normalized_output = normalized_output[1:-1]
    if "\\n" in normalized_output and "\n" not in normalized_output:
        normalized_output = normalized_output.replace("\\n", "\n")

    def _extract_first_json_object(raw: str):
        start_idx = raw.find("{")
        while start_idx != -1:
            depth = 0
            for i in range(start_idx, len(raw)):
                ch = raw[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start_idx : i + 1]
                        try:
                            obj_local = json.loads(candidate)
                            if isinstance(obj_local, dict):
                                return obj_local
                        except Exception:
                            pass
                        break
            start_idx = raw.find("{", start_idx + 1)
        return None

    # Try JSON first.
    try:
        obj = json.loads(normalized_output)
    except Exception:
        obj = _extract_first_json_object(normalized_output)

    try:
        if not isinstance(obj, dict):
            raise ValueError("No JSON object found")
        emotions_obj = obj.get("emotions", {})
        if isinstance(emotions_obj, dict) and emotions_obj:
            for emotion in EMOTION_LABELS:
                v = emotions_obj.get(emotion, None)
                if v in (0, 1, "0", "1"):
                    openai_predictions[emotion] = int(v)
                elif v in (True, False):
                    openai_predictions[emotion] = int(bool(v))
                else:
                    openai_predictions[emotion] = np.nan
            return openai_predictions
    except Exception:
        pass

    # Then try a direct CSV row parse.
    try:
        rows = list(csv.reader(io.StringIO(normalized_output)))
        if rows:
            # Use the first row with enough fields and binary tail.
            for row in rows:
                if len(row) < len(EMOTION_LABELS):
                    continue
                tail = [str(x).strip().strip('"\'') for x in row[-len(EMOTION_LABELS):]]
                if all(v in {"0", "1"} for v in tail):
                    for i, emotion in enumerate(EMOTION_LABELS):
                        openai_predictions[emotion] = int(tail[i])
                    return openai_predictions
    except Exception:
        pass

    gpt_emotions_order = EMOTION_LABELS
    expected_gpt_cols = 3 + len(gpt_emotions_order)

    # Find the data line
    raw_lines = normalized_output.splitlines()
    data_line_string = None

    for i, line in enumerate(raw_lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Skip header lines
        if any(keyword in stripped_line for keyword in ["Song Name", "Artists", "Lyrics", "Joy,Trust"]):
            continue

        # Parse line as CSV
        try:
            string_io_line = io.StringIO(stripped_line)
            reader_line = csv.reader(string_io_line)
            parsed_row = next(reader_line)

            # Check if this looks like data
            if len(parsed_row) >= expected_gpt_cols - 2:
                # Emotion fields should be 0/1 values.
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


# Run tests
print("\nTesting methods on RadaNewAnnotation dataset...")
run_started_at = time.time()
nb_results, bert_results, openai_results, results_data, method_timers, gpt_usage_by_mode = test_methods_on_rada(df_rada)
run_timestamp = datetime.fromtimestamp(run_started_at).strftime("%Y%m%d_%H%M%S")
run_tag = f"{run_timestamp}_{_slugify(OPENAI_MODEL)}"
run_output_dir_name = run_tag

# Build results table
results_df = pd.DataFrame(results_data)
results_df = results_df.set_index('index')  # Keep source row index

# Add ground-truth labels from df_rada
print("\nAdding ground truth labels from df_rada...")
rada_cols_lower = {col.lower(): col for col in df_rada.columns}
for emotion in eval_emotions:
    # Match truth column case-insensitively.
    if emotion.lower() in rada_cols_lower:
        actual_col = rada_cols_lower[emotion.lower()]
        results_df[f'Truth_{emotion}'] = df_rada[actual_col].values
        print(f"Added truth column for {emotion} from df_rada column '{actual_col}'")
    else:
        print(f"Warning: Could not find column for emotion '{emotion}' in df_rada")
        results_df[f'Truth_{emotion}'] = np.nan

# Save predictions
os.makedirs(RESULTS_DIR, exist_ok=True)
output_dir = os.path.join(RESULTS_DIR, run_output_dir_name)
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(os.path.join(output_dir, 'rada_all_predictions.csv'))
print(f"\nSaved all predictions to {os.path.join(output_dir, 'rada_all_predictions.csv')}")

if openai_parse_failures:
    fail_path = os.path.join(output_dir, "openai_parse_failures.jsonl")
    with open(fail_path, "w", encoding="utf-8") as f:
        for row in openai_parse_failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved OpenAI parse failures to {fail_path}")


# Performance evaluation
def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Simple threshold grid search.
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

        # Drop rows with missing prediction or truth.
        mask = results_df[[pred_col, truth_col]].notna().all(axis=1)
        predictions = results_df.loc[mask, pred_col]
        truth = results_df.loc[mask, truth_col]

        if len(predictions) < 2:
            print(f"Not enough data for {method_prefix} - {emotion}")
            continue

        # Convert to binary predictions.
        if method_prefix in PROBABILITY_METHODS:
            t = 0.5 if not thresholds else float(thresholds.get(emotion, 0.5))
            pred_binary = (predictions.astype(float) >= t).astype(int)
        else:
            # OpenAI outputs are already binary.
            pred_binary = predictions.astype(int)

        truth_binary = truth.astype(int)

        try:
            # Compute metrics
            accuracy = accuracy_score(truth_binary, pred_binary)
            report = classification_report(truth_binary, pred_binary, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(truth_binary, pred_binary)
            f1_pos = report.get('1', {}).get('f1-score', np.nan)

            # Compute ROC-AUC/PR-AUC only for score-based methods.
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
# Tune thresholds on one split and evaluate on the other.
tune_df, eval_df = train_test_split(
    results_df,
    test_size=(1.0 - THRESHOLD_TUNE_FRACTION),
    random_state=RANDOM_STATE,
    shuffle=True,
)

nb_thresholds = tune_thresholds(tune_df, 'NB', eval_emotions) if tune else None
bert_thresholds = tune_thresholds(tune_df, 'BERT', eval_emotions) if (tune and bert_load_success) else None

nb_metrics = evaluate_performance(eval_df, 'NB', eval_emotions, thresholds=nb_thresholds)
bert_metrics = evaluate_performance(eval_df, 'BERT', eval_emotions, thresholds=bert_thresholds) if bert_load_success else {}
openai_metrics_by_mode = {
    mode: evaluate_performance(eval_df, f'OpenAI_{mode}', eval_emotions)
    for mode in OPENAI_MODES
}


# Save detailed metrics
def save_detailed_metrics(metrics, method_name, output_dir):
    """Save detailed metrics to a CSV file"""
    if not metrics:
        print(f"No metrics to save for {method_name}")
        return

    # Build summary table.
    summary_data = []
    for emotion, emotion_metrics in metrics.items():
        row = {
            'Emotion': emotion,
            'Accuracy': emotion_metrics['accuracy'],
            'N_Samples': emotion_metrics['n_samples'],
            'AUC_ROC': emotion_metrics.get('auc_roc', np.nan),
            'PR_AUC': emotion_metrics.get('pr_auc', np.nan)
        }

        # Add classification-report metrics.
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
for mode in OPENAI_MODES:
    save_detailed_metrics(openai_metrics_by_mode.get(mode, {}), f'OpenAI_{mode}', output_dir)

# Create comparison summary
print("\n" + "=" * 50)
print("CREATING COMPARISON SUMMARY")
print("=" * 50)


# Average metrics across emotions
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


# Build summary
summary = {
    'NaiveBayes': calculate_average_metrics(nb_metrics),
    'BERT': calculate_average_metrics(bert_metrics) if bert_load_success else {},
}
for mode in OPENAI_MODES:
    summary[f'OpenAI_{mode}'] = calculate_average_metrics(openai_metrics_by_mode.get(mode, {}))

summary['NaiveBayes'].update({
    "display_name": "NaiveBayes",
    "run_tag": run_tag,
    "auc_pr_applicable": True,
    "auc_pr_note": "Probability-based method: ROC-AUC/PR-AUC computed from continuous scores.",
    "openai_model": "",
    "openai_structured_output_requested": False,
    "openai_structured_output_supported": False,
    "openai_json_success_rate": np.nan,
    "openai_parse_fail_rate": np.nan,
    "openai_avg_attempts": np.nan,
})
if summary.get('BERT'):
    summary['BERT'].update({
        "display_name": "BERT",
        "run_tag": run_tag,
        "auc_pr_applicable": True,
        "auc_pr_note": "Probability-based method: ROC-AUC/PR-AUC computed from continuous scores.",
        "openai_model": "",
        "openai_structured_output_requested": False,
        "openai_structured_output_supported": False,
        "openai_json_success_rate": np.nan,
        "openai_parse_fail_rate": np.nan,
        "openai_avg_attempts": np.nan,
    })
for mode in OPENAI_MODES:
    mode_usage = gpt_usage_by_mode.get(mode, {})
    mode_key = f'OpenAI_{mode}'
    summary[mode_key].update({
        "display_name": f"OpenAI ({OPENAI_MODEL}, {mode})",
        "run_tag": run_tag,
        "prompting_mode": mode,
        "auc_pr_applicable": False,
        "auc_pr_note": "Binary outputs only; ROC-AUC/PR-AUC intentionally left NaN to avoid misleading comparisons.",
        "openai_model": OPENAI_MODEL,
        "openai_structured_output_requested": bool(mode_usage.get("structured_output_requested_any", False)),
        "openai_structured_output_supported": bool(mode_usage.get("structured_output_supported", False)),
        "openai_json_success_rate": (float(mode_usage.get("json_success_count", 0)) / float(len(results_df))) if len(results_df) else np.nan,
        "openai_parse_fail_rate": (float(mode_usage.get("parse_fail_count", 0)) / float(len(results_df))) if len(results_df) else np.nan,
        "openai_avg_attempts": float(mode_usage.get("avg_attempts", np.nan)),
    })

# Save summary
summary_df = pd.DataFrame(summary).T
summary_df.to_csv(os.path.join(output_dir, 'methods_comparison_summary.csv'))
print("\nMethods Comparison Summary:")
print(summary_df)

# Save runtime and cost data.
runtime_totals = [
    {"method": "NaiveBayes", "elapsed_sec": method_timers["NB"], "n_samples": len(results_df)},
    {"method": "BERT", "elapsed_sec": method_timers["BERT"], "n_samples": len(results_df)},
]
for mode in OPENAI_MODES:
    mode_usage = gpt_usage_by_mode.get(mode, {})
    runtime_totals.append(
        {
            "method": "OpenAI",
            "method_variant": f"OpenAI_{mode}",
            "method_display": f"OpenAI ({OPENAI_MODEL}, {mode})",
            "elapsed_sec": method_timers.get(f"OpenAI_{mode}", np.nan),
            "n_samples": len(results_df),
            "model": OPENAI_MODEL,
            "prompting_mode": mode,
            "structured_output_requested": bool(mode_usage.get("structured_output_requested_any", False)),
            "structured_output_supported": bool(mode_usage.get("structured_output_supported", False)),
            "json_success_count": int(mode_usage.get("json_success_count", 0)),
            "parse_fail_count": int(mode_usage.get("parse_fail_count", 0)),
            "json_success_rate": (float(mode_usage.get("json_success_count", 0)) / float(len(results_df))) if len(results_df) else np.nan,
            "parse_fail_rate": (float(mode_usage.get("parse_fail_count", 0)) / float(len(results_df))) if len(results_df) else np.nan,
            "avg_attempts": float(mode_usage.get("avg_attempts", np.nan)),
            "prompt_tokens": int(mode_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(mode_usage.get("completion_tokens", 0)),
            "total_tokens": int(mode_usage.get("total_tokens", 0)),
            "estimated_cost_usd": float(mode_usage.get("estimated_cost_usd", 0.0)),
        }
    )
runtime_totals_df = pd.DataFrame(runtime_totals)
runtime_totals_df.to_csv(os.path.join(output_dir, 'runtime_cost_summary.csv'), index=False)

if runtime_events:
    runtime_events_df = pd.DataFrame(runtime_events)
    runtime_events_df.to_csv(os.path.join(output_dir, 'openai_usage_per_song.csv'), index=False)

run_finished_at = time.time()
run_metadata = {
    "run_tag": run_tag,
    "random_state": RANDOM_STATE,
    "openai_model": OPENAI_MODEL,
    "openai_prompting_modes": list(OPENAI_MODES),
    "openai_usage_by_mode": {
        mode: {
            "structured_output_requested": bool(gpt_usage_by_mode.get(mode, {}).get("structured_output_requested_any", False)),
            "structured_output_supported": bool(gpt_usage_by_mode.get(mode, {}).get("structured_output_supported", False)),
            "json_success_count": int(gpt_usage_by_mode.get(mode, {}).get("json_success_count", 0)),
            "parse_fail_count": int(gpt_usage_by_mode.get(mode, {}).get("parse_fail_count", 0)),
            "avg_attempts": float(gpt_usage_by_mode.get(mode, {}).get("avg_attempts", np.nan)),
            "prompt_tokens": int(gpt_usage_by_mode.get(mode, {}).get("prompt_tokens", 0)),
            "completion_tokens": int(gpt_usage_by_mode.get(mode, {}).get("completion_tokens", 0)),
            "total_tokens": int(gpt_usage_by_mode.get(mode, {}).get("total_tokens", 0)),
            "estimated_cost_usd": float(gpt_usage_by_mode.get(mode, {}).get("estimated_cost_usd", 0.0)),
        }
        for mode in OPENAI_MODES
    },
    "use_cleaned_lyrics_for_all_methods": USE_CLEANED_LYRICS_FOR_ALL_METHODS,
    "threshold_tuning_enabled": tune,
    "threshold_tune_fraction": THRESHOLD_TUNE_FRACTION,
    "tuning_rows": int(len(tune_df)),
    "evaluation_rows": int(len(eval_df)),
    "wall_clock_total_sec": float(run_finished_at - run_started_at),
}
with open(os.path.join(output_dir, "run_runtime_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2)

print("\n" + "=" * 50)
print("All evaluations completed successfully!")
print(f"Results saved in '{output_dir}' directory")
print("=" * 50)