import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bert_class import predict_emotions_bert, bert_load_success, EMOTION_LABELS as BERT_EMOTION_LABELS
from nb_class import predict_emotions, preprocess_lyrics, emotions_list as NB_EMOTION_LABELS
from GPT.OpenAI import check_sentiment_openai
import utils
import numpy as np
import io
import csv
import os

from config import (
    RADA_ANNOTATION_CSV,
    SENTIMENT_COMPARISON_CSV,
    EMOTION_LABELS,
    RESULTS_DIR
)

# Ensure all emotions are the same across all models
unified_emotion_labels = list(set(BERT_EMOTION_LABELS) & set(NB_EMOTION_LABELS))
print(f"Unified Emotion Labels: {unified_emotion_labels}")

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


# 2. Test methods on RadaNewAnnotation
def test_methods_on_rada(df):
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
            preprocessed_lyrics = preprocess_lyrics(lyrics)
            nb_predictions = predict_emotions(preprocessed_lyrics)
            nb_results[song_name] = nb_predictions
            # Add NB results to song_results
            for emotion in unified_emotion_labels:
                song_results[f'NB_{emotion}'] = nb_predictions.get(emotion, np.nan)
        except Exception as e:
            print(f"Error in NB prediction for {song_name}: {e}")
            nb_results[song_name] = {emotion: np.nan for emotion in unified_emotion_labels}
            for emotion in unified_emotion_labels:
                song_results[f'NB_{emotion}'] = np.nan

        # BERT
        if bert_load_success:
            try:
                bert_predictions = predict_emotions_bert(lyrics)
                bert_results[song_name] = bert_predictions
                # Add BERT results to song_results
                for emotion in unified_emotion_labels:
                    song_results[f'BERT_{emotion}'] = bert_predictions.get(emotion, np.nan)
            except Exception as e:
                print(f"Error in BERT prediction for {song_name}: {e}")
                bert_results[song_name] = {emotion: np.nan for emotion in unified_emotion_labels}
                for emotion in unified_emotion_labels:
                    song_results[f'BERT_{emotion}'] = np.nan
        else:
            bert_results[song_name] = {emotion: np.nan for emotion in unified_emotion_labels}
            for emotion in unified_emotion_labels:
                song_results[f'BERT_{emotion}'] = np.nan

        # OpenAI
        try:
            openai_output = check_sentiment_openai({'lyrics': lyrics})
            print(f"Raw OpenAI Output length: {len(openai_output) if openai_output else 0} characters")

            openai_predictions = parse_openai_output(openai_output, song_name)
            openai_results[song_name] = openai_predictions

            # Add OpenAI results to song_results
            for emotion in unified_emotion_labels:
                song_results[f'OpenAI_{emotion}'] = openai_predictions.get(emotion, np.nan)

        except Exception as e:
            print(f"Error in OpenAI prediction for {song_name}: {e}")
            openai_results[song_name] = {emotion: np.nan for emotion in unified_emotion_labels}
            for emotion in unified_emotion_labels:
                song_results[f'OpenAI_{emotion}'] = np.nan

        results_data.append(song_results)

    return nb_results, bert_results, openai_results, results_data


def parse_openai_output(openai_output, song_name):
    """Parse OpenAI output more robustly"""
    openai_predictions = {}

    if not openai_output:
        print(f"No OpenAI output for {song_name}")
        return {emotion: np.nan for emotion in unified_emotion_labels}

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
            return {emotion: np.nan for emotion in unified_emotion_labels}
    else:
        print(f"Could not find data line in OpenAI output for {song_name}")
        return {emotion: np.nan for emotion in unified_emotion_labels}

    return openai_predictions


# Run the tests
print("\nTesting methods on RadaNewAnnotation dataset...")
nb_results, bert_results, openai_results, results_data = test_methods_on_rada(df_rada)

# Create a comprehensive results DataFrame
results_df = pd.DataFrame(results_data)
results_df = results_df.set_index('index')  # Use the original index from df_rada

# Now add the ground truth labels from df_rada
print("\nAdding ground truth labels from df_rada...")
for emotion in unified_emotion_labels:
    # Find the corresponding column in df_rada (case-insensitive)
    rada_cols_lower = {col.lower(): col for col in df_rada.columns}
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


# 3. Performance evaluation function
def evaluate_performance(results_df, method_prefix, emotion_labels):
    """Evaluate performance of a method against ground truth"""
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
        if method_prefix in ['NB', 'BERT']:
            # These methods return probabilities
            pred_binary = (predictions >= 0.5).astype(int)
        else:
            # OpenAI already returns 0/1
            pred_binary = predictions.astype(int)

        truth_binary = truth.astype(int)

        try:
            # Calculate metrics
            accuracy = accuracy_score(truth_binary, pred_binary)
            report = classification_report(truth_binary, pred_binary, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(truth_binary, pred_binary)

            all_metrics[emotion] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'n_samples': len(predictions)
            }

            print(f"{method_prefix} - {emotion}: Accuracy = {accuracy:.3f}, N = {len(predictions)}")

        except Exception as e:
            print(f"Error calculating metrics for {method_prefix} - {emotion}: {e}")

    return all_metrics


# Evaluate each method
print("\n" + "=" * 50)
print("PERFORMANCE EVALUATION")
print("=" * 50)

nb_metrics = evaluate_performance(results_df, 'NB', unified_emotion_labels)
bert_metrics = evaluate_performance(results_df, 'BERT', unified_emotion_labels) if bert_load_success else {}
openai_metrics = evaluate_performance(results_df, 'OpenAI', unified_emotion_labels)


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
            'N_Samples': emotion_metrics['n_samples']
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
    for m in metrics.values():
        if 'classification_report' in m and 'weighted avg' in m['classification_report']:
            f1_scores.append(m['classification_report']['weighted avg']['f1-score'])

    return {
        'avg_accuracy': np.mean(accuracies) if accuracies else np.nan,
        'std_accuracy': np.std(accuracies) if accuracies else np.nan,
        'avg_f1_weighted': np.mean(f1_scores) if f1_scores else np.nan,
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