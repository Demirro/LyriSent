import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score

import utils
nb_classifiers = {}

df = pd.read_csv('data/EmotionWheelFinal (1).csv')

df['cleaned_lyrics'] = df['Lyrics'].apply(utils.clean_lyrics)
# Load the NRC Hashtag Emotion Lexicon
lexicon_path = "data/NRC-Hashtag-Emotion-Lexicon-v0.2.txt"
lexicon_df = pd.read_csv(lexicon_path, delimiter='\t', header=None, names=['emotion', 'word', 'score'])

# Filter the lexicon to include only unique words
lexicon_words = set(lexicon_df['word'].str.lower().str.replace(r'#', '', regex=True))

# Filter lexicon words to ensure they are all strings
filtered_lexicon_words = {word for word in lexicon_words if isinstance(word, str)}

# Show some of the lexicon words
#print(list(lexicon_words)[:10])

# Update the CountVectorizer with the filtered vocabulary
vectorizer = CountVectorizer(vocabulary=filtered_lexicon_words)

# Fit and transform the cleaned lyrics
lyrics_bow = vectorizer.fit_transform(df['cleaned_lyrics'])

# Convert to array and create a DataFrame to see the result
lyrics_bow_df = pd.DataFrame(lyrics_bow.toarray(), columns=vectorizer.get_feature_names_out())

#print(lyrics_bow_df.head())

# Initialize a dictionary to store model results
model_results = {}

X = lyrics_bow_df

# List of emotion columns in the dataset
emotion_columns = df.columns[4:-1]  # Assuming the last column is an extra unnamed column

# Training and storing each emotion classifier
for emotion in emotion_columns[:-1]:  # Excluding 'Unnamed: 11'
    # Prepare labels for the current emotion
    y = df[emotion]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Naive Bayes classifier
    nb_classifier = MultinomialNB()

    # Train the classifier
    nb_classifier.fit(X_train, y_train)

    # Store the classifier
    nb_classifiers[emotion] = nb_classifier

    # Predict on the test set
    y_pred = nb_classifier.predict(X_test)

    # Calculate accuracy and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Using weighted to account for class imbalance

    # Store the results
    model_results[emotion] = {'Accuracy': accuracy, 'F1-Score': f1}

# Display the results for each emotion model
#print(model_results)

def preprocess_lyrics(new_lyrics):
    # Clean the lyrics
    cleaned_lyrics = utils.clean_lyrics(new_lyrics)
    # Transform the lyrics using the previously defined vectorizer
    transformed_lyrics = vectorizer.transform([cleaned_lyrics])
    return transformed_lyrics

def predict_emotions(preprocessed_lyrics):
    emotion_scores = {}
    # Annahme: emotion_columns[:-1] enthält die Liste der Emotionen
    # Wir verwenden eine Liste der Emotionen, um sicherzustellen, dass sie in einer konsistenten Reihenfolge sind
    # Du kannst diese Liste bei Bedarf anpassen, basierend auf den tatsächlichen Spaltennamen in deinem df.columns[4:-1]
    emotions_to_predict = [col for col in df.columns[4:-1] if col != 'Unnamed: 11'] # Filter 'Unnamed: 11'

    for emotion in emotions_to_predict:
        # Retrieve the classifier for the current emotion
        classifier = nb_classifiers.get(emotion) # Nutze .get() um Fehler zu vermeiden, falls ein Classifier fehlt
        if classifier:
            # Predict the probability of the emotion being present
            # predict_proba gibt ein Array von Arrays zurück, [0][1] ist die Wahrscheinlichkeit für die positive Klasse
            probability = classifier.predict_proba(preprocessed_lyrics)[0][1]
            # Runde die Wahrscheinlichkeit auf zwei Dezimalstellen
            rounded_probability = round(probability, 2)
            # Store the rounded probability with the corresponding emotion
            emotion_scores[emotion] = rounded_probability
        else:
            print(f"Warning: Classifier for emotion '{emotion}' not found.")
            emotion_scores[emotion] = None # Oder ein anderer Standardwert, falls kein Classifier gefunden wurde

    return emotion_scores
