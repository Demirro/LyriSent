# LyriSent_Bert/nb_class.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score

import utils

# --- Laden und Trainieren des Naive Bayes Modells ---
# (Dieser Teil bleibt wie in deiner Originaldatei,
# da das Training beim Skriptstart erfolgt)

nb_classifiers = {}

# Stelle sicher, dass der Pfad zu deiner CSV-Datei korrekt ist
df = pd.read_csv('data/EmotionWheelFinal (1).csv')

df['cleaned_lyrics'] = df['Lyrics'].apply(utils.clean_lyrics)

# Load the NRC Hashtag Emotion Lexicon
# Stelle sicher, dass der Pfad zu deinem Lexikon korrekt ist
lexicon_path = "data/NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
lexicon_df = pd.read_csv(lexicon_path, delimiter='\t', header=None, names=['emotion', 'word', 'score'])

lexicon_words = set(lexicon_df['word'].str.lower().str.replace(r'#', '', regex=True))
filtered_lexicon_words = {word for word in lexicon_words if isinstance(word, str)}

vectorizer = CountVectorizer(vocabulary=filtered_lexicon_words)

# Fit and transform the cleaned lyrics (Best Practice: Fit nur auf Trainingsdaten,
# aber für feste Vokabulargrösse hier akzeptabel)
lyrics_bow = vectorizer.fit_transform(df['cleaned_lyrics'])

X = lyrics_bow

# List of emotion columns in the dataset
# Passe die Spaltenindizes ggf. an deine tatsächliche CSV an
emotion_columns = df.columns[4:-1]
# Wir definieren hier die Liste der Emotionen, die wir verwenden
emotions_list = [col for col in emotion_columns if col != 'Unnamed: 11'] # Filter 'Unnamed: 11'


# Training and storing each emotion classifier
for emotion in emotions_list:
    y = df[emotion]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb_classifier = MultinomialNB()

    # Train the classifier
    nb_classifier.fit(X_train, y_train)

    # Store the classifier
    nb_classifiers[emotion] = nb_classifier

    # Predict on the test set (optional, for evaluation)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print(f"NB - Emotion: {emotion}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")


# --- Vorhersage Funktionen ---

def preprocess_lyrics(new_lyrics):
    """
    Bereinigt und vektorisiert neue Songtexte für das Naive Bayes Modell.
    """
    cleaned_lyrics = utils.clean_lyrics(new_lyrics)
    # Transform the lyrics using the previously defined vectorizer
    transformed_lyrics = vectorizer.transform([cleaned_lyrics])
    return transformed_lyrics

def predict_emotions(preprocessed_lyrics):
    """
    Macht Emotionsvorhersagen mit den trainierten Naive Bayes Klassifikatoren
    und gibt die Wahrscheinlichkeiten gerundet zurück.
    """
    emotion_scores = {}
    for emotion in emotions_list:
        classifier = nb_classifiers.get(emotion)
        if classifier:
            # predict_proba gibt [Wahrscheinlichkeit_Klasse_0, Wahrscheinlichkeit_Klasse_1] zurück
            # Wir wollen die Wahrscheinlichkeit für die positive Klasse (1)
            probability = classifier.predict_proba(preprocessed_lyrics)[0][1]
            # Runde die Wahrscheinlichkeit auf zwei Dezimalstellen
            rounded_probability = round(probability, 2)
            emotion_scores[emotion] = rounded_probability
        else:
            emotion_scores[emotion] = None # Oder 0.0, je nach gewünschter Behandlung

    return emotion_scores

# Füge hier ggf. weitere Hilfsfunktionen für NB hinzu