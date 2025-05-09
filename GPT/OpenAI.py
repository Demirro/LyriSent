from os import environ

from openai import OpenAI

client = OpenAI()
from openai import OpenAI

client = OpenAI()


def check_sentiment_openai(song):
    lyrics = song['lyrics']
    # Example OpenAI Python library request
    MODEL = "gpt-4" # Oder welcher Modellname auch immer du verwendest
    response = client.chat.completions.create(model=MODEL,
                                              messages=[
                                                  {"role": "system",
                                                   # --- VERFEINERTES PROMPT ---
                                                   "content": "You are a tone analyzer for lyrics. Your ONLY output is the sentiment analysis in CSV format. Provide ONLY a header row followed by ONE data row for the analyzed lyrics. Systematically analyze the mood of the song lyrics according to Plutchiks 8 core emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation. Give the results in a tabular form using csv with the table header: Song Name,Artists,Lyrics,Joy,Trust,Fear,Surprise,Sadness,Disgust,Anger,Anticipation. Limit the output of the Lyrics column to only 50 symbols and add ... . Wrap the lyrics in single quotes. If a song displays any of the emotions it should get a 1 in the given column, if not a 0. Think about annotating as if you were the result or agreement of multiple hundreds of annotators. ABSOLUTELY no other text, explanation, or formatting outside of the CSV header and data row. Do NOT include markdown ```csv ``` formatting."},
                                                  {"role": "user",
                                                   "content": "Analyze the following lyrics: \n" + lyrics},
                                              ],
                                              temperature=0 # Niedrige Temperatur für konsistentere Ausgabe
                                              # Optional: response_format parameter (prüfen, ob dein Modell es unterstützt)
                                              # response_format={"type": "json_object"} # Wenn du JSON statt CSV anfordern möchtest (erfordert dann Anpassung der Parsing-Logik in app.py)
                                             )
    # Stelle sicher, dass wir den Text der Nachricht erhalten
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        return response.choices[0].message.content
    else:
        print("Warning: OpenAI API returned no message content.")
        return "" # Leeren String zurückgeben, wenn keine Antwort erhalten wurde
