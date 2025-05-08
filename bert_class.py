# LyriSent_Bert/bert_class.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# --- Konfiguration ---
# Pfad, unter dem dein trainiertes BERT-Modell gespeichert ist
# Beispiel: './trained_bert_model'
BERT_MODEL_PATH = './trained_bert_model'
# Name des vortrainierten Modells, das für das Fine-Tuning verwendet wurde
# Beispiel: 'bert-base-uncased'
PRETRAINED_MODEL_NAME = 'bert-base-uncased'

# Die Reihenfolge der Emotionen, wie sie vom trainierten BERT-Modell ausgegeben wird
# STELL SICHER, dass diese Reihenfolge mit der Reihenfolge übereinstimmt,
# in der du deine Labels beim BERT-Training kodiert hast!
# Diese Liste sollte die gleichen Emotionen wie in nb_class.py enthalten
EMOTION_LABELS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']

# Gerät für Berechnungen (CPU oder GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Laden des Modells und Tokenizers ---
tokenizer = None
model = None
bert_load_success = False

try:
    # Lade den Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    # Lade das trainierte Modell
    # Konfiguriere num_labels entsprechend der Anzahl deiner Emotionen
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=len(EMOTION_LABELS))
    model.to(DEVICE)
    model.eval() # Setze das Modell in den Evaluationsmodus
    bert_load_success = True
    print(f"Successfully loaded BERT model from {BERT_MODEL_PATH}")
except OSError:
    print(f"Warning: BERT model not found at {BERT_MODEL_PATH}. BERT predictions will not be available.")
except Exception as e:
    print(f"An error occurred while loading the BERT model: {e}")


# --- Vorhersage Funktion ---

def predict_emotions_bert(lyrics):
    """
    Macht Emotionsvorhersagen mit dem trainierten BERT-Modell.
    Gibt ein Dictionary mit Emotionswahrscheinlichkeiten zurück.
    """
    if not bert_load_success:
        return {emotion: None for emotion in EMOTION_LABELS} # Gib None zurück, wenn das Modell nicht geladen wurde

    # Bereinigung der Lyrics könnte hier oder vorher in app.py erfolgen
    # Wir gehen davon aus, dass hier bereits bereinigte oder rohe Texte kommen,
    # die der Tokenizer verarbeiten kann. Starke Bereinigung (wie in utils.py)
    # ist für BERT oft weniger nötig, da es Subword-Tokenizer nutzt.
    # Lass uns hier die utils.clean_lyrics verwenden für Konsistenz,
    # obwohl BERT auch mit mehr Satzzeichen umgehen kann.
    import utils
    cleaned_lyrics = utils.clean_lyrics(lyrics)


    # Tokenisieren des Textes
    # max_length sollte der Länge entsprechen, die auch beim Training verwendet wurde
    # padding='max_length' stellt sicher, dass alle Sequenzen gleich lang sind
    # truncation=True schneidet längere Sequenzen ab
    encoding = tokenizer(
        cleaned_lyrics,
        max_length=128, # Passe dies an die Länge an, die du beim Training verwendet hast
        padding='max_length',
        truncation=True,
        return_tensors='pt' # Gibt PyTorch Tensoren zurück
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    predictions = {}
    with torch.no_grad(): # Keine Gradientenberechnung im Inferenzmodus
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Die Ausgabe des Modells sind Logits. Wende Sigmoid an, um Wahrscheinlichkeiten zu erhalten.
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy() # Sigmoid, Squeeze (entfernt Batch-Dimension), auf CPU, zu numpy

    # Ordne die Wahrscheinlichkeiten den Emotions-Labels zu
    # Stelle sicher, dass probabilities und EMOTION_LABELS in der richtigen Reihenfolge sind
    for i, emotion in enumerate(EMOTION_LABELS):
        # Runde die Wahrscheinlichkeit auf zwei Dezimalstellen
        rounded_probability = round(probabilities[i], 2)
        predictions[emotion] = rounded_probability

    return predictions

# Beispiel Aufruf (kann zum Testen dieser Datei genutzt werden)
if __name__ == '__main__':
    # Dieses Beispiel funktioniert nur, wenn ein Modell unter BERT_MODEL_PATH existiert
    if bert_load_success:
        test_lyrics = "This song makes me feel incredibly happy and full of joy!"
        bert_preds = predict_emotions_bert(test_lyrics)
        print(f"BERT predictions for '{test_lyrics}': {bert_preds}")

        test_lyrics_sad = "Feeling down and lonely, nothing seems right today."
        bert_preds_sad = predict_emotions_bert(test_lyrics_sad)
        print(f"BERT predictions for '{test_lyrics_sad}': {bert_preds_sad}")
    else:
        print("BERT model not loaded, cannot run example.")