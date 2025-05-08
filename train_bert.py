# LyriSent_Bert/train_bert.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Adafactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import os
import time

# --- Konfiguration ---
# Pfad zu deinem Dataset
DATA_PATH = 'data/EmotionWheelFinal (1).csv'
# Pfad, unter dem das trainierte Modell gespeichert werden soll
SAVE_PATH = './trained_bert_model'
# Name des vortrainierten BERT-Modells von Hugging Face
PRETRAINED_MODEL_NAME = 'bert-base-uncased' # Ein gängiges, nicht casesensitives Modell
# Die Spalten in deiner CSV, die die Emotionen als Labels enthalten
# Passe die Indizes ggf. an deine tatsächliche CSV an
# Annahme: Spalten 4 bis vorletzte Spalte (ohne 'Unnamed: 11')
# Diese Liste MUSS die gleiche Reihenfolge und die gleichen Emotionen wie in bert_class.py haben!
EMOTION_COLUMNS_SLICE = slice(4, -1) # Slice für die Spaltenindizes
EXCLUDE_COLUMN = 'Unnamed: 11' # Spalte, die ausgeschlossen werden soll

# Trainingsparameter
MAX_SEQ_LENGTH = 512 # Maximale Länge der tokenisierten Sequenzen (BERT base typical: 512)
BATCH_SIZE = 16      # Kleinere Batch-Größe für GPU-Speicher
NUM_EPOCHS = 25       # Anzahl der Trainingsdurchläufe durch das Dataset
LEARNING_RATE = 2e-5 # Typische Lernrate für BERT Fine-Tuning
# Gewichtungsabnahme für den Optimierer
WEIGHT_DECAY = 0.01

# --- Gerät Setup ---
print(torch.cuda.is_available())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Daten laden und vorbereiten ---
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Identifiziere die Text- und Label-Spalten
TEXT_COLUMN = 'Lyrics' # Passe den Namen der Spalte mit den Songtexten an

# Dynamisch die tatsächlichen Emotionsspalten basierend auf Slice und Ausschluss ermitteln
all_potential_emotion_cols = df.columns[EMOTION_COLUMNS_SLICE].tolist()
EMOTION_LABELS = [col for col in all_potential_emotion_cols if col != EXCLUDE_COLUMN]

print(f"Identified {len(EMOTION_LABELS)} emotion labels: {EMOTION_LABELS}")
if not EMOTION_LABELS:
    raise ValueError("No emotion columns found based on the provided slice and exclusion.")

# Extrahiere Texte und Labels
texts = df[TEXT_COLUMN].tolist()
# Extrahiere Label-Spalten und konvertiere zu NumPy Array (float für BCEWithLogitsLoss)
labels = df[EMOTION_LABELS].values.astype(float)

print(f"Loaded {len(texts)} texts and corresponding labels.")
# Überprüfe Dimensionen
# print(f"Shape of labels array: {labels.shape}")


# Daten aufteilen in Training, Validierung und Testsets
# Validation set ist wichtig, um Overfitting während des Trainings zu überwachen
# Test set wird am Ende einmalig für die finale Evaluation verwendet
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42) # stratify=labels entfernt
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # stratify=y_temp entfernt

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")


# --- Tokenisierung ---
print(f"Loading tokenizer: {PRETRAINED_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def tokenize_texts(tokenizer, texts, max_length):
    """Tokenisiert eine Liste von Texten."""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt' # Gibt PyTorch Tensoren zurück
    )
    return encodings['input_ids'], encodings['attention_mask']

# Tokenisiere die Datensätze
print("Tokenizing datasets...")
train_input_ids, train_attention_masks = tokenize_texts(tokenizer, X_train, MAX_SEQ_LENGTH)
val_input_ids, val_attention_masks = tokenize_texts(tokenizer, X_val, MAX_SEQ_LENGTH)
test_input_ids, test_attention_masks = tokenize_texts(tokenizer, X_test, MAX_SEQ_LENGTH)

# Konvertiere Labels zu PyTorch Tensoren
train_labels = torch.tensor(y_train, dtype=torch.float32)
val_labels = torch.tensor(y_val, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.float32)


# --- PyTorch Dataset und DataLoader ---
class EmotionDataset(Dataset):
    """Benutzerdefiniertes Dataset für Emotionstexte."""
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# Erstelle Datasets
train_dataset = EmotionDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = EmotionDataset(val_input_ids, val_attention_masks, val_labels)
test_dataset = EmotionDataset(test_input_ids, test_attention_masks, test_labels)

# Erstelle DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- Modell definieren ---
print(f"Loading BERT model: {PRETRAINED_MODEL_NAME} with {len(EMOTION_LABELS)} labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    # Füge dies hinzu, um das Modell für Multi-Label zu konfigurieren
    problem_type="multi_label_classification"
)
model.to(DEVICE)


# --- Optimierer und Verlustfunktion ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# --- NEUER CODE: Berechnung der positiven Gewichte für unausgeglichene Klassen ---
print("Calculating positive weights for BCEWithLogitsLoss...")
# y_train ist ein NumPy Array (oder Tensor, je nachdem, wie du es nach dem Split konvertiert hast)
# Stelle sicher, dass es ein NumPy Array der Form (Anzahl Trainingsbeispiele, Anzahl Emotionen) ist
if isinstance(y_train, torch.Tensor):
    y_train_np = y_train.cpu().numpy() # Konvertiere zu NumPy, falls es ein Tensor ist
else:
    y_train_np = y_train # Es ist bereits NumPy

num_samples = y_train_np.shape[0] # Anzahl der Trainingsbeispiele
num_emotions = y_train_np.shape[1] # Anzahl der Emotionen

# Zähle positive Beispiele pro Emotion im Trainingsset
# Summiere über die erste Achse (Beispiele)
positive_counts = y_train_np.sum(axis=0)

# Zähle negative Beispiele pro Emotion
negative_counts = num_samples - positive_counts

# Berechne die Gewichte für die positive Klasse
# Gewicht_i = Anzahl_negativer_Beispiele_i / Anzahl_positiver_Beispiele_i
# Füge einen kleinen Wert (1e-5) zum Nenner hinzu, um Division durch Null zu vermeiden, falls eine Emotion 0 positive Beispiele hat
pos_weights = negative_counts / (positive_counts + 1e-5)

# Konvertiere die berechneten Gewichte zu einem PyTorch Tensor und verschiebe es auf dasselbe Gerät wie das Modell
pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

print(f"Calculated positive weights per emotion: {pos_weights}")
# --- ENDE NEUER CODE ---


# BCEWithLogitsLoss ist gut für Multi-Label, da es Sigmoid und Binary Cross Entropy kombiniert
# --- MODIFIZIERTE ZEILE: Initialisierung der Verlustfunktion MIT den Gewichten ---
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)



# --- Trainings- und Evaluationsfunktionen ---
def train_epoch(model, data_loader, optimizer, loss_fn, device):
    """Trainiert das Modell für eine Epoche."""
    model.train() # Setze Modell in Trainingsmodus
    total_loss = 0
    start_time = time.time()

    for step, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Gradienten zurücksetzen

        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits # Rohe Ausgaben vor der Aktivierungsfunktion

        # Verlust berechnen
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward Pass und Optimierung
        loss.backward()
        optimizer.step()

        if step % 10 == 0: # Optional: Fortschritt anzeigen
            elapsed_time = time.time() - start_time
            print(f"  Step {step}/{len(data_loader)} Loss: {loss.item():.4f} Elapsed: {elapsed_time:.2f}s")
            start_time = time.time()


    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, loss_fn, device, emotion_labels):
    """Evaluiert das Modell auf einem Datensatz."""
    model.eval() # Setze Modell in Evaluationsmodus
    total_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad(): # Deaktiviere Gradientenberechnung
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Konkateniere alle Logits und Labels
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Wende Sigmoid auf die Logits an, um Wahrscheinlichkeiten zu erhalten
    all_probabilities = 1 / (1 + np.exp(-all_logits))

    # Wende einen Schwellenwert an (z.B. 0.5) um binäre Vorhersagen zu erhalten
    all_predictions = (all_probabilities > 0.5).astype(float)

    # Berechne Metriken
    accuracy = accuracy_score(all_labels, all_predictions) # Exact match accuracy (alle Labels müssen korrekt sein)
    # F1-Score per Label und gemittelt
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0) # Micro F1: totals up TP, FP, FN across all labels
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0) # Macro F1: average of F1 for each label
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) # Weighted F1: average of F1 for each label, weighted by support

    # Optional: AUC-ROC Score (funktioniert nur, wenn mindestens 2 Klassen vorhanden sind und nicht alle Labels 0 sind)
    # und Average Precision Score
    try:
        auc_roc_micro = roc_auc_score(all_labels, all_probabilities, average='micro')
        auc_roc_macro = roc_auc_score(all_labels, all_probabilities, average='macro')
        avg_precision_micro = average_precision_score(all_labels, all_probabilities, average='micro')
        avg_precision_macro = average_precision_score(all_labels, all_probabilities, average='macro')
    except ValueError:
        auc_roc_micro, auc_roc_macro, avg_precision_micro, avg_precision_macro = np.nan, np.nan, np.nan, np.nan
        print("Could not calculate AUC-ROC or Average Precision (likely only one class present in labels).")


    metrics = {
        'loss': avg_loss,
        'accuracy (exact match)': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_roc_micro': auc_roc_micro,
        'auc_roc_macro': auc_roc_macro,
        'average_precision_micro': avg_precision_micro,
        'average_precision_macro': avg_precision_macro,
    }

    # Optionale Ausgabe der F1-Scores pro Label
    f1_per_label = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    print("F1-Score per label:")
    for i, label in enumerate(emotion_labels):
        print(f"  {label}: {f1_per_label[i]:.4f}")


    return metrics


# --- Trainings-Loop ---
print("Starting training...")
best_val_f1_macro = -1 # Verfolge die beste Validierungsleistung

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Validierung
    val_metrics = evaluate(model, val_loader, loss_fn, DEVICE, EMOTION_LABELS)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Metrics:")
    for metric_name, metric_value in val_metrics.items():
        if metric_name != 'loss' and metric_value is not None:
             print(f"  {metric_name}: {metric_value:.4f}")


    # Modell speichern, wenn die Validierungsleistung besser ist (hier: macro F1)
    if val_metrics['f1_macro'] > best_val_f1_macro:
        best_val_f1_macro = val_metrics['f1_macro']
        # Erstelle das Speicherverzeichnis, falls es nicht existiert
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        # Speichere das Modell und den Tokenizer
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"Saved best model to {SAVE_PATH} with Validation Macro F1: {best_val_f1_macro:.4f}")

print("\nTraining finished.")

# --- Finale Evaluation auf dem Testset ---
print("\nEvaluating the best model on the test set...")
# Lade das beste Modell, das wir gerade gespeichert haben
if os.path.exists(SAVE_PATH):
     best_model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH, num_labels=len(EMOTION_LABELS))
     best_model.to(DEVICE)
     test_metrics = evaluate(best_model, test_loader, loss_fn, DEVICE, EMOTION_LABELS)
     print("\n--- Test Set Metrics ---")
     for metric_name, metric_value in test_metrics.items():
         if metric_value is not None:
              print(f"  {metric_name}: {metric_value:.4f}")
else:
    print(f"Best model not found at {SAVE_PATH}. Cannot perform test set evaluation.")


print("\nBERT training script finished.")