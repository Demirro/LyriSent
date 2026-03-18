# LyriSent_Bert/app.py
import csv
import io
import re
# import streamlit as st # Falls du Streamlit nutzt, einkommentieren
import os # Importiere os für Pfadoperationen

# Importiere deine lokalen Module
import nb_class
from GPT.OpenAI import check_sentiment_openai # Stelle sicher, dass diese Datei existiert
from genius import fetch_lyrics # Stelle sicher, dass diese Datei existiert und funktioniert
import utils # Stelle sicher, dass diese Datei existiert
import bert_class # Importiere die neue BERT-Klasse/Modul

from config import (
    SENTIMENT_COMPARISON_CSV,
    EMOTION_LABELS
)

# --- Helfer-Funktionen (können aus deiner Original app.py übernommen werden) ---

# Pfad zur CSV-Datei für die kombinierten Ergebnisse
output_csv_file_path = SENTIMENT_COMPARISON_CSV

# Lade existierende Daten aus der kombinierten CSV, falls vorhanden
existing_combined_data = []
# Definiere hier den erwarteten Header, um Spalten korrekt zuordnen zu können
# Wird beim ersten Schreiben erstellt, aber hilfreich beim Lesen existierender Daten
expected_header = [
    'Song', 'Artist', 'Lyrics',
    'Joy_GPT', 'Trust_GPT', 'Fear_GPT', 'Surprise_GPT', 'Sadness_GPT', 'Disgust_GPT', 'Anger_GPT', 'Anticipation_GPT',
    # Diese müssen zur Reihenfolge in nb_class.emotions_list passen
    'Joy_NB', 'Trust_NB', 'Fear_NB', 'Surprise_NB', 'Sadness_NB', 'Disgust_NB', 'Anger_NB', 'Anticipation_NB',
    # Diese müssen zur Reihenfolge in bert_class.EMOTION_LABELS passen
    'Joy_BERT', 'Trust_BERT', 'Fear_BERT', 'Surprise_BERT', 'Sadness_BERT', 'Disgust_BERT', 'Anger_BERT', 'Anticipation_BERT'
]
# Stelle sicher, dass die NB und BERT Spaltennamen zur tatsächlichen Reihenfolge in den predict Funktionen passen!
# Annahme: bert_class.EMOTION_LABELS ist identisch zu nb_class.emotions_list für einfache Zuordnung

try:
    # Prüfe, ob die Datei existiert und nicht leer ist
    if os.path.exists(output_csv_file_path) and os.path.getsize(output_csv_file_path) > 0:
        with open(output_csv_file_path, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header_row = next(reader) # Lese Header

            # Optional: Überprüfe oder re-mappe Spalten basierend auf dem Header,
            # falls sich die Reihenfolge oder Namen geändert haben könnten.
            # Für jetzt gehen wir davon aus, der Header ist wie erwartet beim Lesen.
            print(f"Loaded existing CSV with header: {header_row}")
            existing_combined_data = list(reader)
            print(f"Loaded {len(existing_combined_data)} existing rows.")

except FileNotFoundError:
    print(f"No existing CSV file found at {output_csv_file_path}. A new one will be created.")
    existing_combined_data = []
except Exception as e:
    print(f"An error occurred while loading existing CSV: {e}")
    existing_combined_data = []


def is_track_checked(track_to_check, artist_to_check, existing_data):
    """
    Überprüft, ob ein Track bereits in den existierenden Daten vorhanden ist.
    """
    if not existing_data:
        return False
    # Gehe durch die existierenden Zeilen
    for row in existing_data:
         # Annahme: Song Name ist in Spalte 0, Artist in Spalte 1
        if len(row) > 1 and row[0].casefold() == track_to_check.casefold() and row[1].casefold() == artist_to_check.casefold():
            #print(f"Track '{track_to_check}' by '{artist_to_check}' was already checked. It will be skipped.")
            return True
    return False

def _input(message, input_type=str):
    """
    Helferfunktion für Benutzereingaben.
    """
    while True:
        try:
            return input_type(input(message))
        except ValueError:
            print("Invalid input type. Please try again.")
        except EOFError:
             print("\nInput stream closed. Exiting.")
             exit() # Beende das Programm, wenn kein Input mehr möglich ist


# --- Hauptlogik ---

def main():
    songs_list = []
    prefab_check = input('Do you want to use the prefab list of songs (10 songs)? Yes (y/Y) or No (n/N)\n')

    if prefab_check.casefold() in {'yes','y'}:
        songs_list = [
            ('Childish Gambino','This is America'),
             ('brakence','deepfacke'),
             ('Peter Fox','Haus am See'),
             ('Feu! Chatterton', "J'ai tout mon temps"),
             ('Bruno Mars', 'Treasure'),
             ('Ed Sheeran', 'Shape Of You'),
             ('The Japanese House', 'Saw You In A Dream'),
             ('Tom Misch', 'Disco Yes'),
             ('Radiohead', 'Creep'),
             ('Jacob Collier', 'Hideaway'),
        ]
    elif prefab_check.casefold() in {'no','n'}:
        number_of_songs = _input('How many songs do you want to check: ', int)
        for i in range(number_of_songs):
            print(f'Song no. {i+1}')
            track_name = input('Track Name: ')
            artist = input('Artist: ')
            songs_list.append((artist, track_name))
            print('\n')
        print(f"Checking the following songs: {songs_list}")
    else:
        print('Invalid input. Please enter Yes (y/Y) or No (n/N)')
        return # Programm beenden, wenn Input ungültig ist


    # Liste zur Speicherung der neuen oder aktualisierten Daten
    # Wir arbeiten mit einer Kopie der existierenden Daten und fügen neue hinzu
    # oder aktualisieren, falls nötig.
    # Liste zur Speicherung der neuen oder aktualisierten Daten
    all_combined_data_rows = list(existing_combined_data)  # Erstelle eine veränderbare Kopie

    song_count = 0
    for artist, track in songs_list:
        track_already_processed = is_track_checked(track, artist, all_combined_data_rows)

        if track_already_processed:
            print(f"Skipping '{track}' by '{artist}' as it was already processed.")
            continue  # Gehe zum nächsten Song

        print(f"Processing '{track}' by '{artist}'...")

        result = fetch_lyrics(artist, track)
        if result and 'lyrics' in result and result['lyrics']:
            lyrics = result['lyrics']
            # Bereinige Lyrics für Modelle (NB, BERT) und sende sie auch an GPT
            cleaned_lyrics = utils.clean_lyrics(lyrics)

            # --- OpenAI Sentiment Analyse ---
            print("Fetching GPT response...")
            song_data_for_openai = {'lyrics': cleaned_lyrics}
            gpt_response_string = ''
            try:
                gpt_response_string = check_sentiment_openai(song_data_for_openai)
                print(f"GPT Response Received (raw):\n---\n{gpt_response_string}\n---")

            except Exception as e:
                print(f"An error occurred during OpenAI API call for '{track}': {e}")
                gpt_response_string = ''  # Setze auf leer im Fehlerfall

            # --- Noch robusteres Parsing der GPT-Antwort: Identifiziere die Datenzeile im Roh-String ---
            gpt_sentiment_list_raw = []
            gpt_emotions_order = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
            expected_gpt_cols = 3 + len(gpt_emotions_order)  # Song Name,Artists,Lyrics + 8 Emotionen

            data_line_string = None  # Speichert den String der identifizierten Datenzeile

            # Teile die rohe Antwort in individuelle Zeilen auf
            raw_lines = gpt_response_string.strip().splitlines()
            # print(f"GPT Parsing: Split raw response into {len(raw_lines)} lines.") # Debug

            # Gehe Zeile für Zeile durch, um die Datenzeile zu finden.
            # Die Datenzeile sollte eine plausible Anzahl von Kommas (für die Spalten) enthalten
            # und wahrscheinlich nicht die Header-Namen.
            header_keywords = ["Song Name", "Artists", "Lyrics", "Joy", "Trust"]  # Typische Wörter im Header

            for i, line in enumerate(raw_lines):
                stripped_line = line.strip()
                if not stripped_line:  # Überspringe leere Zeilen
                    continue

                # Heuristik 1: Eine plausible Datenzeile hat ungefähr die erwartete Anzahl von Kommas
                # Wir erwarten N Spalten, also N-1 Kommas. Geben wir einen kleinen Puffer.
                num_commas = stripped_line.count(',')
                if num_commas < expected_gpt_cols - 3 or num_commas > expected_gpt_cols + 3:  # Erwarte N-1 Kommas, Puffer +-2
                    # print(f"GPT Parsing: Line {i+1} skipped (comma count {num_commas} not plausible).") # Debug
                    continue

                # Heuristik 2: Eine Datenzeile sollte typischerweise nicht die Header-Keywords enthalten
                if any(keyword in stripped_line for keyword in header_keywords):
                    # print(f"GPT Parsing: Line {i+1} skipped (contains header keywords).") # Debug
                    continue

                # Heuristik 3: Versuche, diese einzelne Zeile als CSV zu parsen und prüfe die Spaltenanzahl
                try:
                    string_io_line = io.StringIO(stripped_line)
                    reader_line = csv.reader(string_io_line)
                    parsed_row_from_line = next(reader_line)

                    # Heuristik 4: Prüfe, ob die geparste Zeile eine plausible Spaltenanzahl hat
                    # Innerhalb eines engen Bereichs um die erwartete Anzahl
                    if len(parsed_row_from_line) >= expected_gpt_cols - 2 and len(
                            parsed_row_from_line) <= expected_gpt_cols + 2:
                        # Diese Zeile scheint eine plausible Datenzeile zu sein!
                        data_line_string = stripped_line  # Speichere den String dieser Zeile
                        print(
                            f"GPT Parsing: Identified plausible data line string (line {i + 1}, {len(parsed_row_from_line)} cols): {data_line_string}")  # Debug
                        break  # Datenzeile gefunden, Suche beenden

                    # else:
                    # print(f"GPT Parsing: Line {i+1} parsed into {len(parsed_row_from_line)} cols, outside plausible range.") # Debug

                except Exception as e:
                    # print(f"GPT Parsing: Could not parse line {i+1} as CSV: {stripped_line[:100]}... Error: {e}") # Debug
                    continue  # Diese Zeile konnte nicht als CSV geparst werden oder passte nicht

            # Jetzt, parse die identifizierte Datenzeile (falls gefunden)
            if data_line_string:
                string_io = io.StringIO(data_line_string)
                reader = csv.reader(string_io)
                try:
                    # Lies die EINE Datenzeile aus dem String der identifizierten Zeile
                    gpt_sentiment_list_raw = next(reader)

                    # Validiere und passe die Spaltenanzahl an (gleich wie vorher)
                    if len(gpt_sentiment_list_raw) < expected_gpt_cols:
                        print(
                            f"Warning: Parsed GPT data row for '{track}' has fewer columns ({len(gpt_sentiment_list_raw)}) than expected ({expected_gpt_cols}). Padding with empty strings.")
                        gpt_sentiment_list_raw.extend([''] * (expected_gpt_cols - len(gpt_sentiment_list_raw)))
                    elif len(gpt_sentiment_list_raw) > expected_gpt_cols:
                        print(
                            f"Warning: Parsed GPT data row for '{track}' has more columns ({len(gpt_sentiment_list_raw)}) als expected ({expected_gpt_cols}). Truncating.")
                        gpt_sentiment_list_raw = gpt_sentiment_list_raw[:expected_gpt_cols]

                except Exception as e:
                    print(
                        f"Severe error parsing identified data_line_string '{data_line_string[:100]}...' for '{track}': {e}")
                    gpt_sentiment_list_raw = [''] * expected_gpt_cols  # Setze auf leere Liste im Fehlerfall

            else:
                print(f"Warning: Could not identify a plausible data line string within the response for '{track}'.")
                gpt_sentiment_list_raw = [''] * expected_gpt_cols  # Keine Datenzeile gefunden

            # Extrahiere die Emotionswerte aus der geparsten Liste
            # Dieser Block bleibt gleich wie in der vorherigen verbesserten Version
            gpt_emotion_values = {}
            # Stelle sicher, dass gpt_sentiment_list_raw die minimale Länge für Song, Artist, Lyrics hat
            if len(gpt_sentiment_list_raw) >= 3:
                gpt_emotions_order_expected = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger',
                                               'Anticipation']
                for i, emotion in enumerate(gpt_emotions_order_expected):
                    # Überprüfe, ob der Index für die Emotion gültig ist
                    if 3 + i < len(gpt_sentiment_list_raw):
                        try:
                            # Hole den Wert-String aus der geparsten Liste
                            value_str_raw = gpt_sentiment_list_raw[3 + i]
                            # --- VERBESSERUNG HIER ---
                            # Entferne führende/nachfolgende Leerzeichen UND potenziell überzählige Anführungszeichen
                            value_str = value_str_raw.strip().rstrip('"').rstrip(
                                "'")  # rstrip('"') entfernt ein " am Ende, rstrip("'") entfernt ein ' am Ende

                            # Prüfe nun, ob der bereinigte String "0" oder "1" ist
                            if value_str in {'0', '1'}:
                                gpt_emotion_values[emotion] = int(value_str)
                            else:
                                # Wenn der Wert nach der Bereinigung nicht '0' oder '1' ist, logge es und setze auf leeren String
                                # print(f"Warning: GPT value for '{emotion}' in '{track}' is not '0' or '1' after cleaning: '{value_str}'. Raw: '{value_str_raw}'. Setting to ''.")
                                gpt_emotion_values[emotion] = ''  # Speichere einen leeren String für ungültige Werte
                        except (ValueError, IndexError) as e:
                            # Fehler beim Zugriff auf Index oder bei der Konvertierung (sollte seltener passieren)
                            print(
                                f"Error processing GPT emotion value for '{emotion}' in '{track}'. Raw value: '{gpt_sentiment_list_raw[3 + i] if 3 + i < len(gpt_sentiment_list_raw) else 'N/A'}'. Error: {e}")
                            gpt_emotion_values[emotion] = ''
                    else:
                        # Index ausserhalb der Grenzen der geparsten Liste
                        print(
                            f"Warning: Index out of bounds when extracting GPT emotion '{emotion}' for '{track}'. List length: {len(gpt_sentiment_list_raw)}.")
                        gpt_emotion_values[emotion] = ''
            else:
                # Die geparste Liste hat weniger als 3 Spalten
                print(
                    f"Warning: Parsed GPT list for '{track}' has less than 3 columns. Cannot extract emotions. List: {gpt_sentiment_list_raw}")
                for emotion in gpt_emotions_order:
                    gpt_emotion_values[emotion] = ''

            # *** Naive Bayes Sentiment Vorhersage ***
            print("Performing Naive Bayes prediction...")
            try:
                preprocessed_lyrics_nb = nb_class.preprocess_lyrics(cleaned_lyrics)
                emotion_predictions_nb = nb_class.predict_emotions(preprocessed_lyrics_nb)
                #print(f"Naive Bayes Predictions: {emotion_predictions_nb}")
            except Exception as e:
                print(f"An error occurred during Naive Bayes prediction for '{track}': {e}")
                emotion_predictions_nb = {emotion: None for emotion in nb_class.emotions_list}


            # *** BERT Sentiment Vorhersage ***
            print("Performing BERT prediction...")
            emotion_predictions_bert = {}
            if bert_class.bert_load_success: # Nur versuchen, wenn das Modell geladen wurde
                 try:
                     emotion_predictions_bert = bert_class.predict_emotions_bert(lyrics) # Kann bereinigte oder rohe Texte nehmen, je nach bert_class
                     #print(f"BERT Predictions: {emotion_predictions_bert}")
                 except Exception as e:
                     print(f"An error occurred during BERT prediction for '{track}': {e}")
                     emotion_predictions_bert = {emotion: None for emotion in bert_class.EMOTION_LABELS}
            else:
                 print("BERT model not loaded, skipping BERT prediction.")
                 emotion_predictions_bert = {emotion: None for emotion in bert_class.EMOTION_LABELS}


            # *** Kombinieren der Ergebnisse ***

            # Erstelle eine neue Zeile für die kombinierten Daten
            combined_row = [
                track,
                artist,
                cleaned_lyrics[:200] + '...' if len(cleaned_lyrics) > 200 else cleaned_lyrics, # Begrenze Lyrics Länge für CSV Anzeige
            ]

            # Füge GPT-Ergebnisse hinzu (basierend auf der erwarteten Reihenfolge im GPT Prompt)
            gpt_emotions_order = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
            for emotion in gpt_emotions_order:
                 combined_row.append(gpt_emotion_values.get(emotion, '')) # Nutze get(), um Fehler zu vermeiden, falls eine Emotion fehlt

            # Füge Naive Bayes Ergebnisse hinzu (basierend auf der Reihenfolge in nb_class.emotions_list)
            for emotion in nb_class.emotions_list:
                 combined_row.append(emotion_predictions_nb.get(emotion, None)) # Nutze get()

            # Füge BERT Ergebnisse hinzu (basierend auf der Reihenfolge in bert_class.EMOTION_LABELS)
            for emotion in bert_class.EMOTION_LABELS:
                 combined_row.append(emotion_predictions_bert.get(emotion, None)) # Nutze get()


            # Füge die neue Zeile zu den gesammelten Daten hinzu
            all_combined_data_rows.append(combined_row)

            song_count += 1
        else:
            print(f"Could not fetch lyrics for '{track}' by '{artist}' or lyrics were empty.")


    # *** Speichern der kombinierten Daten in einer neuen CSV-Datei ***
    print(f"\nSaving collected data to {output_csv_file_path}...")
    try:
        with open(output_csv_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Definiere den endgültigen Header
            final_header = [
                'Song', 'Artist', 'Lyrics',
                'Joy_GPT', 'Trust_GPT', 'Fear_GPT', 'Surprise_GPT', 'Sadness_GPT', 'Disgust_GPT', 'Anger_GPT', 'Anticipation_GPT'
            ]
            # Füge die Naive Bayes Emotionsspalten hinzu basierend auf nb_class.emotions_list
            nb_header_cols = [f"{emotion}_NB" for emotion in nb_class.emotions_list]
            final_header.extend(nb_header_cols)
            # Füge die BERT Emotionsspalten hinzu basierend auf bert_class.EMOTION_LABELS
            bert_header_cols = [f"{emotion}_BERT" for emotion in bert_class.EMOTION_LABELS]
            final_header.extend(bert_header_cols)


            writer.writerow(final_header)  # Schreibe die Kopfzeile
            writer.writerows(all_combined_data_rows) # Schreibe alle gesammelten Daten

    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

    print(f"Data has been saved to {output_csv_file_path}")
    print(f"Processed {song_count} new songs.")


if __name__ == "__main__":
    main()