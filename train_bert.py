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
import json
import random
import hashlib
import platform
import sys
from datetime import datetime, timezone
import utils

from config import (
    EMOTION_WHEEL_CSV,
    MODEL_DIR as SAVE_PATH,
    PRETRAINED_MODEL_NAME,
    EMOTION_LABELS,
    EMOTION_COLUMNS_SLICE,
    EXCLUDE_COLUMN,
    TEXT_COLUMN,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)

# -------------------------
# Reproducibility utilities
# -------------------------
SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism: may reduce performance; some ops can still be nondeterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(save_dir: str, extra: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, "run_metadata.json")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device": str(DEVICE) if "DEVICE" in globals() else None,
        "transformers": None,
        "sklearn": None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": {
            "pretrained_model_name": PRETRAINED_MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "emotion_labels": EMOTION_LABELS,
            "text_column": TEXT_COLUMN,
        },
        **extra,
    }

    # Optional imports for versions (avoid hard failures)
    try:
        import transformers  # type: ignore

        payload["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import sklearn  # type: ignore

        payload["sklearn"] = sklearn.__version__
    except Exception:
        pass

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Device setup
set_global_seed(SEED)
print(torch.cuda.is_available())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Load and prepare data
print(f"Loading data from {EMOTION_WHEEL_CSV}...")
df = pd.read_csv(EMOTION_WHEEL_CSV)

# Validate that EMOTION_LABELS from config exist in the dataframe
missing_emotions = [emotion for emotion in EMOTION_LABELS if emotion not in df.columns]
if missing_emotions:
    raise ValueError(f"Missing emotion columns in CSV: {missing_emotions}. Expected: {EMOTION_LABELS}")

# Keep BERT training/inference consistent: both run on cleaned lyrics.
texts = df[TEXT_COLUMN].astype(str).apply(utils.clean_lyrics).tolist()
# Extract label columns as float for BCEWithLogitsLoss.
labels = df[EMOTION_LABELS].values.astype(float)

print(f"Loaded {len(texts)} texts and corresponding labels.")
# Check dimensions
# print(f"Shape of labels array: {labels.shape}")


# Split data into train/validation/test sets.
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)  # no stratify
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # no stratify

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")


# Tokenization
print(f"Loading tokenizer: {PRETRAINED_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def tokenize_texts(tokenizer, texts, max_length):
    """Tokenize a list of texts."""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'  # Return PyTorch tensors
    )
    return encodings['input_ids'], encodings['attention_mask']

# Tokenize datasets
print("Tokenizing datasets...")
train_input_ids, train_attention_masks = tokenize_texts(tokenizer, X_train, MAX_SEQ_LENGTH)
val_input_ids, val_attention_masks = tokenize_texts(tokenizer, X_val, MAX_SEQ_LENGTH)
test_input_ids, test_attention_masks = tokenize_texts(tokenizer, X_test, MAX_SEQ_LENGTH)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(y_train, dtype=torch.float32)
val_labels = torch.tensor(y_val, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.float32)


# PyTorch Dataset and DataLoader
class EmotionDataset(Dataset):
    """Custom dataset for emotion texts."""
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

# Build datasets
train_dataset = EmotionDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = EmotionDataset(val_input_ids, val_attention_masks, val_labels)
test_dataset = EmotionDataset(test_input_ids, test_attention_masks, test_labels)

# Build dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define model
print(f"Loading BERT model: {PRETRAINED_MODEL_NAME} with {len(EMOTION_LABELS)} labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    # Configure model for multi-label classification.
    problem_type="multi_label_classification"
)
model.to(DEVICE)


# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Compute positive weights for imbalanced classes.
print("Calculating positive weights for BCEWithLogitsLoss...")
# Ensure y_train is a NumPy array with shape (n_samples, n_emotions).
if isinstance(y_train, torch.Tensor):
    y_train_np = y_train.cpu().numpy()
else:
    y_train_np = y_train

num_samples = y_train_np.shape[0]
num_emotions = y_train_np.shape[1]

# Count positive samples per emotion.
positive_counts = y_train_np.sum(axis=0)

# Count negative samples per emotion.
negative_counts = num_samples - positive_counts

# Positive class weight = negatives / positives.
# Add epsilon to avoid division by zero.
pos_weights = negative_counts / (positive_counts + 1e-5)

# Move weights to model device.
pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

print(f"Calculated positive weights per emotion: {pos_weights}")
# BCEWithLogitsLoss with class weights for imbalance.
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)



# Training and evaluation helpers
def train_epoch(model, data_loader, optimizer, loss_fn, device):
    """Train the model for one epoch."""
    model.train()  # Training mode
    total_loss = 0
    start_time = time.time()

    for step, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        if step % 10 == 0:  # Optional progress logging
            elapsed_time = time.time() - start_time
            print(f"  Step {step}/{len(data_loader)} Loss: {loss.item():.4f} Elapsed: {elapsed_time:.2f}s")
            start_time = time.time()


    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, loss_fn, device, emotion_labels):
    """Evaluate the model on a dataset."""
    model.eval()  # Evaluation mode
    total_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
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

    # Concatenate all logits and labels.
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert logits to probabilities.
    all_probabilities = 1 / (1 + np.exp(-all_logits))

    # Apply threshold to get binary predictions.
    all_predictions = (all_probabilities > 0.5).astype(float)

    # Compute metrics.
    accuracy = accuracy_score(all_labels, all_predictions)  # Exact-match accuracy
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Optional AUC-ROC and average precision metrics.
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

    # Optional per-label F1 output.
    f1_per_label = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    print("F1-Score per label:")
    for i, label in enumerate(emotion_labels):
        print(f"  {label}: {f1_per_label[i]:.4f}")


    return metrics


# Training loop
print("Starting training...")
best_val_f1_macro = -1  # Track best validation score

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # Training
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation
    val_metrics = evaluate(model, val_loader, loss_fn, DEVICE, EMOTION_LABELS)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Metrics:")
    for metric_name, metric_value in val_metrics.items():
        if metric_name != 'loss' and metric_value is not None:
             print(f"  {metric_name}: {metric_value:.4f}")


    # Save model when validation macro F1 improves.
    if val_metrics['f1_macro'] > best_val_f1_macro:
        best_val_f1_macro = val_metrics['f1_macro']
        # Create save directory if needed.
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        # Save model and tokenizer.
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"Saved best model to {SAVE_PATH} with Validation Macro F1: {best_val_f1_macro:.4f}")

        # Save metadata for reproducibility
        try:
            dataset_hash = sha256_file(EMOTION_WHEEL_CSV)
        except Exception:
            dataset_hash = None
        write_run_metadata(
            str(SAVE_PATH),
            extra={
                "dataset": {"path": str(EMOTION_WHEEL_CSV), "sha256": dataset_hash},
                "split": {
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                    "random_state": 42,
                },
                "best_val_f1_macro": float(best_val_f1_macro),
            },
        )

print("\nTraining finished.")

# Final evaluation on the test set
print("\nEvaluating the best model on the test set...")
# Load the saved best model.
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