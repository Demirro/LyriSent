# LyriSent_Bert/bert_class.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

from config import (
    BERT_MODEL_PATH, 
    PRETRAINED_MODEL_NAME, 
    EMOTION_LABELS, 
    MAX_SEQ_LENGTH
)

# Compute device (CPU or GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Load model and tokenizer
tokenizer = None
model = None
bert_load_success = False

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    # Load trained model with configured label count
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=len(EMOTION_LABELS))
    model.to(DEVICE)
    model.eval()  # Inference mode
    bert_load_success = True
    print(f"Successfully loaded BERT model from {BERT_MODEL_PATH}")
except OSError:
    print(f"Warning: BERT model not found at {BERT_MODEL_PATH}. BERT predictions will not be available.")
except Exception as e:
    print(f"An error occurred while loading the BERT model: {e}")


# Prediction function

def predict_emotions_bert(lyrics):
    """Predict emotion probabilities using the trained BERT model."""
    if not bert_load_success:
        return {emotion: None for emotion in EMOTION_LABELS}  # Model unavailable

    # Use the same cleaning step as training for consistency.
    import utils
    cleaned_lyrics = utils.clean_lyrics(lyrics)


    # Tokenize with the same max length used in training.
    encoding = tokenizer(
        cleaned_lyrics,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    predictions = {}
    with torch.no_grad():  # No gradients in inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Convert logits to probabilities with sigmoid.
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Map probabilities to labels in config order.
    for i, emotion in enumerate(EMOTION_LABELS):
        # Round for stable output formatting.
        rounded_probability = round(probabilities[i], 2)
        predictions[emotion] = rounded_probability

    return predictions

# Example call for local testing
if __name__ == '__main__':
    # Requires a trained model in BERT_MODEL_PATH.
    if bert_load_success:
        test_lyrics = "This song makes me feel incredibly happy and full of joy!"
        bert_preds = predict_emotions_bert(test_lyrics)
        print(f"BERT predictions for '{test_lyrics}': {bert_preds}")

        test_lyrics_sad = "Feeling down and lonely, nothing seems right today."
        bert_preds_sad = predict_emotions_bert(test_lyrics_sad)
        print(f"BERT predictions for '{test_lyrics_sad}': {bert_preds_sad}")
    else:
        print("BERT model not loaded, cannot run example.")