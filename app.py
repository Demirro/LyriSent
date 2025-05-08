def main():
    # List of songs (artist, track)
    songs_list = [
    ]
    prefab_check = input('Do you want to use the prefab list of songs (10 songs)? Yes (y/Y) or No (n/N)\n')
    if prefab_check.casefold() in {'yes','y'}:
        songs_list = [
            ('Childish Gambino','This is America'),
            # ('brakence','deepfacke'),
            # ('Peter Fox','Haus am See'),
            # ('Feu! Chatterton', "J'ai tout mon temps"),
            # ('Bruno Mars', 'Treasure'),
            # ('Ed Sheeran', 'Shape Of You'),
            # ('The Japanese House', 'Saw You In A Dream'),
            # ('Tom Misch', 'Disco Yes'),
            # ('Radiohead', 'Creep'),
            # ('Jacob Collier', 'Hideaway'),
        ]
    elif prefab_check.casefold() in {'no','n'}:
        number_of_songs = _input('How many songs do you want to check: ', int)
        for i in range(number_of_songs):
            print('Song no. ' + str(i+1))
            track_name = input('Track Name: ')
            artist = input('Artist: ')
            songs_list.append((artist, track_name))
            print('\n')
        print(songs_list)
    else:
        print('Yes (y/Y) or No (n/N)')

    # Fetch lyrics for each song and get sentiment predictions
    processed_data = [] # Liste zur Speicherung der kombinierten Daten
    song_count = 0
    for artist, track in songs_list:
        if not is_track_checked(track): # Überprüfe, ob der Track bereits in der CSV ist
            result = fetch_lyrics(artist, track)
            if result:
                lyrics = result['lyrics']
                cleaned_lyrics = clean_lyrics(lyrics) # Nutze die clean_lyrics Funktion aus utils.py

                # *** OpenAI Sentiment Analyse ***
                gpt_response_string = ''
                # Überprüfen, ob bereits eine GPT-Antwort vorhanden ist
                for existing_track in existing_csv_data:
                    if existing_track and existing_track[0].casefold() == track.casefold() and existing_track[1].casefold() == artist.casefold():
                         # Annahme: Song Name ist in Spalte 0, Artist in Spalte 1
                        if len(existing_track) > 3: # Überprüfen, ob GPT-Antwort Spalte existiert (Spalte 2 ist Lyrics, Spalte 3 könnte GPT-Antwort sein, je nach CSV Struktur)
                            gpt_response_string = existing_track[3] # Annahme: GPT-Antwort ist in Spalte 3
                            print(f"Found existing GPT response for {track} by {artist}")
                            break # Gefunden und Schleife verlassen

                if not gpt_response_string: # Wenn keine bestehende GPT-Antwort gefunden wurde
                     print(f"Fetching GPT response for {track} by {artist}")
                     # Erstelle ein Dictionary, das die erwartete Struktur für check_sentiment_openai hat
                     song_data_for_openai = {'lyrics': cleaned_lyrics}
                     gpt_response_string = check_sentiment_openai(song_data_for_openai)
                     # Entferne doppelte Anführungszeichen, falls nötig
                     gpt_response_string = re.sub('""', '', gpt_response_string)
                     print(f"GPT Response: {gpt_response_string}")


                # Hier wandeln wir den GPT-Antwort-String in eine Liste um
                gpt_sentiment_list = []
                if gpt_response_string:
                    string_io = io.StringIO(gpt_response_string)
                    reader = csv.reader(string_io)
                    try:
                         # Lies die erste (und einzige) Zeile der GPT CSV Antwort (Header überspringen falls vorhanden)
                        rows = list(reader)
                        if len(rows) > 1: # Skip header if present
                             gpt_sentiment_list = rows[1]
                        elif len(rows) == 1: # Process if no header
                             gpt_sentiment_list = rows[0]

                         # Überprüfe, ob die gpt_sentiment_list die erwartete Länge hat (Song Name, Artists, Lyrics + 8 Emotionen)
                        expected_gpt_length = 3 + 8 # Song Name, Artists, Lyrics + 8 Emotionen
                        if len(gpt_sentiment_list) < expected_gpt_length:
                             print(f"Warning: GPT response for {track} by {artist} is shorter than expected. Expected {expected_gpt_length} columns, got {len(gpt_sentiment_list)}")
                             # Füge leere Werte hinzu, falls Spalten fehlen, um Fehler zu vermeiden
                             gpt_sentiment_list.extend([''] * (expected_gpt_length - len(gpt_sentiment_list)))

                    except Exception as e:
                        print(f"Error parsing GPT response for {track} by {artist}: {e}")
                        # Setze die Liste auf leere Werte bei Parsing-Fehler
                        gpt_sentiment_list = [''] * (3 + 8)


                # *** Naive Bayes Sentiment Vorhersage ***
                print(f"Performing Naive Bayes prediction for {track} by {artist}")
                preprocessed_lyrics_nb = nb_class.preprocess_lyrics(cleaned_lyrics)
                emotion_predictions_nb = nb_class.predict_emotions(preprocessed_lyrics_nb)
                print(f"Naive Bayes Predictions: {emotion_predictions_nb}")

                # Kombiniere die Daten
                # Stelle sicher, dass die Reihenfolge der Spalten mit dem Header übereinstimmt
                # Header: Song,Artist,Lyrics,Joy_GPT,Trust_GPT,Fear_GPT,Surprise_GPT,Sadness_GPT,Disgust_GPT,Anger_GPT,Anticipation_GPT,Joy_NB,Trust_NB,Fear_NB,Surprise_NB,Sadness_NB,Disgust_NB,Anger_NB,Anticipation_NB

                # Erstelle eine neue Zeile für die kombinierten Daten
                combined_row = [
                    track,
                    artist,
                    cleaned_lyrics[:50] + '...' if len(cleaned_lyrics) > 50 else cleaned_lyrics, # Begrenze Lyrics Länge
                ]

                # Füge GPT-Ergebnisse hinzu. Annahme: Reihenfolge der Emotionen im GPT-Output ist Joy,Trust,Fear,Surprise,Sadness,Disgust,Anger,Anticipation
                # Wir müssen sicherstellen, dass wir die richtigen Spalten aus gpt_sentiment_list extrahieren.
                # Basierend auf dem System-Prompt in OpenAI.py ist der GPT CSV Header: Song Name,Artists,Lyrics,Joy,Trust,Fear,Surprise,Sadness,Disgust,Anger,Anticipation
                # Das bedeutet, die Emotionen beginnen ab Index 3 in der gpt_sentiment_list.
                gpt_emotions_order = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
                gpt_emotion_values = {}
                if len(gpt_sentiment_list) >= 3 + len(gpt_emotions_order):
                    for i, emotion in enumerate(gpt_emotions_order):
                        try:
                            # Extrahiere den Wert und konvertiere zu int (0 oder 1)
                            gpt_emotion_values[emotion] = int(gpt_sentiment_list[3 + i])
                        except (ValueError, IndexError):
                            gpt_emotion_values[emotion] = '' # Handle fehlende oder ungültige Werte

                # Füge die extrahierten GPT-Emotionswerte hinzu
                for emotion in gpt_emotions_order:
                    combined_row.append(gpt_emotion_values.get(emotion, ''))


                # Füge Naive Bayes Ergebnisse hinzu. emotion_predictions_nb ist ein Dictionary {emotion: probability}
                nb_emotions_order = emotion_predictions_nb.keys() # Die Reihenfolge der Emotionen aus Naive Bayes
                for emotion in nb_emotions_order:
                    combined_row.append(emotion_predictions_nb.get(emotion, '')) # Füge die Wahrscheinlichkeit hinzu

                processed_data.append(combined_row) # Füge die kombinierte Zeile zur Liste hinzu

                song_count += 1

    # *** Speichern der kombinierten Daten in einer neuen CSV-Datei ***
    output_csv_file_path = 'data/sentiment_comparison.csv' # Neuer Dateiname für die kombinierte Ausgabe
    try:
        with open(output_csv_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Neuer Header mit NB und GPT Unterscheidung
            header = ['Song', 'Artist', 'Lyrics', 'Joy_GPT', 'Trust_GPT', 'Fear_GPT', 'Surprise_GPT', 'Sadness_GPT', 'Disgust_GPT', 'Anger_GPT', 'Anticipation_GPT']
            # Füge die Naive Bayes Emotionsspalten hinzu basierend auf den Schlüsseln im NB Ergebnis
            nb_emotions_order_header = [f"{emotion}_NB" for emotion in emotion_predictions_nb.keys()]
            header.extend(nb_emotions_order_header)

            writer.writerow(header)  # Schreibe die Kopfzeile
            # Schreibe die gesammelten Daten
            writer.writerows(processed_data)

    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

    print(f"Sentiment comparison data has been saved to {output_csv_file_path}")