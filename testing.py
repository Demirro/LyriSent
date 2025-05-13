import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bert_class import predict_emotions_bert, bert_load_success, EMOTION_LABELS as BERT_EMOTION_LABELS
from nb_class import predict_emotions, preprocess_lyrics, emotions_list as NB_EMOTION_LABELS
from GPT.OpenAI import check_sentiment_openai
import utils
import numpy as np
import io
import csv
import os  # Import the os module

# Stelle sicher, dass alle Emotionen in allen Modellen gleich sind
unified_emotion_labels = list(set(BERT_EMOTION_LABELS) & set(NB_EMOTION_LABELS))
print(f"Unified Emotion Labels: {unified_emotion_labels}")

# 1. Laden der Daten
df_rada = pd.read_csv('data/RadaNewAnnotation (1).csv')
df_predefined = pd.read_csv('data/sentiment_comparison.csv')

print("\nStructure of df_rada:")
print(df_rada.info())
print("\nFirst few rows of df_rada:")
print(df_rada.head())

# Initialisiere Dictionaries zum Speichern der Ergebnisse
nb_results = {}
bert_results = {}
openai_results = {}
comparison_data = []


# 2. Testen der Methoden auf RadaNewAnnotation
def test_methods_on_rada(df):
    for index, row in df.iterrows():
        lyrics = row['text']
        song_name = row['Song']
        print(f"Processing song: {song_name}")

        # Naive Bayes
        preprocessed_lyrics = preprocess_lyrics(lyrics)
        nb_predictions = predict_emotions(preprocessed_lyrics)
        nb_results[song_name] = nb_predictions

        # BERT
        if bert_load_success:
            bert_predictions = predict_emotions_bert(lyrics)
            bert_results[song_name] = bert_predictions
        else:
            bert_results[song_name] = {emotion: None for emotion in unified_emotion_labels}

        # OpenAI
        openai_output = check_sentiment_openai({'lyrics': lyrics})
        print(f"Raw OpenAI Output:\n---\n{openai_output}\n---\n")  # Print raw output for inspection
        openai_predictions = {}  # Initialize to an empty dictionary
        if openai_output:
            gpt_sentiment_list_raw = []
            gpt_emotions_order = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
            expected_gpt_cols = 3 + len(gpt_emotions_order)
            data_line_string = None
            raw_lines = openai_output.strip().splitlines()

            for i, line in enumerate(raw_lines):
                stripped_line = line.strip()
                if not stripped_line:
                    print(f"Skipping empty line {i + 1}")
                    continue
                num_commas = stripped_line.count(',')
                print(f"Line {i + 1}: Commas = {num_commas}, Line: {stripped_line}")
                if num_commas < expected_gpt_cols - 3 or num_commas > expected_gpt_cols + 3:
                    print(f"Skipping line {i + 1} due to comma count.")
                    continue
                if any(keyword in stripped_line for keyword in ["Song Name", "Artists", "Lyrics", "Joy", "Trust"]):
                    print(f"Skipping line {i + 1} due to header keyword.")
                    continue
                try:
                    string_io_line = io.StringIO(stripped_line)
                    reader_line = csv.reader(string_io_line)  # Use default CSV reader
                    parsed_row_from_line = next(reader_line)
                    if len(parsed_row_from_line) >= expected_gpt_cols - 2 and len(parsed_row_from_line) <= expected_gpt_cols + 2:
                        data_line_string = stripped_line
                        print(f"Identified data line: {data_line_string}")
                        break
                except Exception as e:
                    print(f"Error parsing line {i + 1}: {e}")
                    continue

            if data_line_string:
                string_io = io.StringIO(data_line_string)
                reader = csv.reader(string_io)  # Use default CSV reader
                try:
                    gpt_sentiment_list_raw = next(reader)
                    if len(gpt_sentiment_list_raw) < expected_gpt_cols:
                        gpt_sentiment_list_raw.extend([''] * (expected_gpt_cols - len(gpt_sentiment_list_raw)))
                    elif len(gpt_sentiment_list_raw) > expected_gpt_cols:
                        gpt_sentiment_list_raw = gpt_sentiment_list_raw[:expected_gpt_cols]
                except Exception as e:
                    print(f"Error parsing data line: {e}")
                    gpt_sentiment_list_raw = [''] * expected_gpt_cols
            else:
                print("Could not identify data line in GPT output.")
                gpt_sentiment_list_raw = [''] * expected_gpt_cols

            gpt_emotion_values = {}
            if len(gpt_sentiment_list_raw) >= 3:
                gpt_emotions_order_expected = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger',
                                               'Anticipation']
                for i, emotion in enumerate(gpt_emotions_order_expected):
                    if 3 + i < len(gpt_sentiment_list_raw):
                        try:
                            value_str_raw = gpt_sentiment_list_raw[3 + i]
                            value_str = value_str_raw.strip().rstrip('"').rstrip("'")
                            if value_str in {'0', '1'}:
                                gpt_emotion_values[emotion] = int(value_str)
                            else:
                                gpt_emotion_values[emotion] = ''
                        except (ValueError, IndexError) as e:
                            print(f"Error processing GPT emotion value: {e}")
                            gpt_emotion_values[emotion] = ''
                    else:
                        print(f"Index out of bounds for GPT emotion: {emotion}")
                        gpt_emotion_values[emotion] = ''
            else:
                print(f"Parsed GPT list has less than 3 columns: {gpt_sentiment_list_raw}")
                for emotion in gpt_emotions_order:
                    gpt_emotion_values[emotion] = ''

            openai_predictions = gpt_emotion_values  # Assign the parsed values
        else:
            print("No OpenAI output received.")
            openai_predictions = {emotion: None for emotion in unified_emotion_labels}  # Ensure a dict is returned

        openai_results[song_name] = openai_predictions

    return nb_results, bert_results, openai_results


nb_results, bert_results, openai_results = test_methods_on_rada(df_rada)

# Konvertiere Ergebnisse in DataFrames für die weitere Analyse
nb_results_df = pd.DataFrame(nb_results).transpose()
bert_results_df = pd.DataFrame(bert_results).transpose()
openai_results_df = pd.DataFrame(openai_results).transpose()

# Fülle fehlende Spalten mit NaN, um die DataFrames kompatibel zu machen
for df in [nb_results_df, bert_results_df, openai_results_df]:
    for emotion in unified_emotion_labels:
        if emotion not in df.columns:
            df[emotion] = np.nan

# Erstelle einen DataFrame für den Vergleich
comparison_df_rada = pd.DataFrame(index=nb_results_df.index)

# Reset index of df_rada to ensure alignment
df_rada = df_rada.reset_index(drop=True)

# Iterate through the emotion columns in unified_emotion_labels
for emotion in unified_emotion_labels:
    # Check for the emotion column in df_rada (case-insensitive)
    available_columns = [col.lower() for col in df_rada.columns]
    if emotion.lower() in available_columns:
        rada_col_name = df_rada.columns[available_columns.index(emotion.lower())]
        print(f"Transferring data from df_rada column '{rada_col_name}' to comparison_df_rada for emotion '{emotion}'")
        comparison_df_rada[f'Standard_{emotion}'] = df_rada[rada_col_name]
    else:
        print(f"Warning: Emotion column '{emotion}' not found in df_rada")
        comparison_df_rada[f'Standard_{emotion}'] = np.nan

print("\nColumn names in comparison_df_rada:")
print(comparison_df_rada.columns)

# Stelle sicher, dass das Verzeichnis existiert
output_dir = 'comparison_results'
os.makedirs(output_dir, exist_ok=True)  # Erstelle das Verzeichnis, wenn es nicht existiert

comparison_df_rada.to_csv(os.path.join(output_dir, 'rada_predictions_with_standard.csv'))


# 3. Leistungsbewertung und Vergleich mit Standard
def evaluate_performance(results_df, standard_df, method_name):
    all_metrics = {}
    for emotion in unified_emotion_labels:
        if f'Standard_{emotion}' in standard_df.columns and emotion in results_df.columns:
            standard = standard_df[f'Standard_{emotion}'].dropna()
            predictions = results_df[emotion].dropna()

            print(f"\n--- Evaluating {method_name} - {emotion} ---")
            print(f"Standard data shape: {standard.shape}")
            print(f"Predictions data shape: {predictions.shape}")
            print(f"Standard non-NaN values: {standard.count()}")
            print(f"Predictions non-NaN values: {predictions.count()}")

            # Align indices to ensure correct comparison
            standard = standard.reindex(predictions.index)
            predictions = predictions.reindex(standard.index)

            # Check if we have enough data to evaluate
            if standard.count() > 1 and predictions.count() > 1:
                standard_list = standard.tolist()
                predictions_list = [1 if p >= 0.5 else 0 for p in predictions.tolist()]

                report = classification_report(standard_list, predictions_list, output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(standard_list, predictions_list)
                accuracy = accuracy_score(standard_list, predictions_list)

                all_metrics[emotion] = {
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'accuracy': accuracy
                }
            else:
                print(f"Not enough data for evaluation of {method_name} - {emotion}")
        else:
            print(f"Standard_{emotion} oder {emotion} nicht gefunden in DataFrames für {method_name}")
    return all_metrics


# Bewerte die Leistung jeder Methode
nb_metrics = evaluate_performance(nb_results_df, comparison_df_rada, 'Naive Bayes')
if bert_load_success:
    bert_metrics = evaluate_performance(bert_results_df, comparison_df_rada, 'BERT')
openai_metrics = evaluate_performance(openai_results_df, comparison_df_rada, 'OpenAI')


# Speichere die Leistungsmetriken
def save_metrics(metrics, filename):
    if metrics:
        all_data = {}
        for emotion, metric_values in metrics.items():
            all_data[emotion] = metric_values
        df_metrics = pd.DataFrame(all_data)
        df_metrics.to_csv(filename)
    else:
        print(f"Keine Metriken zum Speichern in {filename}")


# Stelle sicher, dass das Verzeichnis existiert
output_dir = 'comparison_results'
os.makedirs(output_dir, exist_ok=True)  # Erstelle das Verzeichnis, wenn es nicht existiert

save_metrics(nb_metrics, os.path.join(output_dir, 'rada_performance_metrics_nb.csv'))
if bert_load_success:
    save_metrics(bert_metrics, os.path.join(output_dir, 'rada_performance_metrics_bert.csv'))
save_metrics(openai_metrics, os.path.join(output_dir, 'rada_performance_metrics_openai.csv'))


# 4. Vergleich der Methoden untereinander (predefined list)
def compare_methods_agreement(nb_results, bert_results, openai_results, df_predefined):
    agreement_data = []
    nb_songs = set(nb_results.keys())
    bert_songs = set(bert_results.keys())
    openai_songs = set(openai_results.keys())
    all_songs = nb_songs.union(bert_songs).union(openai_songs)

    for song in all_songs:
        agreement = {'Song': song}

        for emotion in unified_emotion_labels:
            # Verwende .get(), um sicherzustellen, dass kein Fehler auftritt, wenn ein Song fehlt
            nb_emotion_data = nb_results.get(song, {})
            bert_emotion_data = bert_results.get(song, {})
            openai_emotion_data = openai_results.get(song, {})

            nb = nb_emotion_data.get(emotion)
            bert = bert_emotion_data.get(emotion)
            openai = openai_emotion_data.get(emotion)

            # Konvertiere Wahrscheinlichkeiten in binäre Werte (1 oder 0)
            nb_bin = 1 if nb >= 0.5 else 0 if nb is not None else None
            bert_bin = 1 if bert >= 0.5 else 0 if bert is not None else None
            # Konvertiere openai zu float, default zu 0.0 wenn leer
            openai_bin = 1 if float(openai or 0.0) >= 0.5 else 0 if openai is not None else None

            agreement[f'NB_{emotion}'] = nb_bin
            agreement[f'BERT_{emotion}'] = bert_bin
            agreement[f'OpenAI_{emotion}'] = openai_bin

            # Berechne die Übereinstimmung zwischen den Methoden (hier als binäre Übereinstimmung)
            valid_values = [val for val in [nb_bin, bert_bin, openai_bin] if val is not None]
            if valid_values:
                # Wenn alle Werte gleich sind, ist die Übereinstimmung 1, sonst 0
                all_same = all(x == valid_values[0] for x in valid_values)
                agreement[f'{emotion}_Agreement'] = 1 if all_same else 0
            else:
                agreement[f'{emotion}_Agreement'] = None
        agreement_data.append(agreement)
    return pd.DataFrame(agreement_data)


# Führe den Methodenvergleich durch
agreement_df = compare_methods_agreement(
    {song: {emotion: nb_results[song][emotion] for emotion in unified_emotion_labels if emotion in nb_results[song]} for
     song in nb_results.keys()},  # Iterate over keys
    {song: {emotion: bert_results.get(song, {}).get(emotion) for emotion in unified_emotion_labels if
             bert_results.get(song) and emotion in bert_results.get(song)} for song in bert_results.keys()} if bert_load_success else {},  # Iterate over keys
    {song: {emotion: openai_results.get(song, {}).get(emotion) for emotion in unified_emotion_labels if
             openai_results.get(song) and emotion in openai_results.get(song)} for song in openai_results.keys()},  # Iterate over keys
    df_predefined
)

# Stelle sicher, dass das Verzeichnis existiert
output_dir = 'comparison_results'
os.makedirs(output_dir, exist_ok=True)  # Erstelle das Verzeichnis, wenn es nicht existiert

agreement_df.to_csv(os.path.join(output_dir, 'method_agreement_metrics.csv'))

print("Alle Berechnungen und Vergleiche wurden abgeschlossen. Die Ergebnisse wurden in CSV-Dateien gespeichert.")