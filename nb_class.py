from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import utils
from config import EMOTION_LABELS, EMOTION_WHEEL_CSV, NRC_LEXICON_TXT, TEXT_COLUMN


emotions_list = EMOTION_LABELS.copy()


@dataclass
class NBArtifacts:
    vectorizer: CountVectorizer
    classifiers: Dict[str, MultinomialNB]


_ARTIFACTS: Optional[NBArtifacts] = None


def _load_lexicon_vocabulary(path: Path) -> set[str]:
    lexicon_df = pd.read_csv(path, delimiter="\t", header=None, names=["emotion", "word", "score"])
    # Guard against mixed dtypes/NaN values: CountVectorizer vocabulary must be pure strings.
    cleaned_words = (
        lexicon_df["word"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"#", "", regex=True)
        .str.strip()
    )
    return {w for w in cleaned_words.tolist() if isinstance(w, str) and w and w != "nan"}


def train_nb(
    emotion_wheel_csv: Path = EMOTION_WHEEL_CSV,
    nrc_lexicon_txt: Path = NRC_LEXICON_TXT,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> NBArtifacts:
    """
    Train one MultinomialNB per emotion using an NRC-lexicon-fixed vocabulary.
    Training is explicit (not run at import time) for reproducibility.
    """
    df = pd.read_csv(emotion_wheel_csv)

    missing_emotions = [emotion for emotion in emotions_list if emotion not in df.columns]
    if missing_emotions:
        raise ValueError(f"Missing emotion columns in CSV: {missing_emotions}. Expected: {emotions_list}")
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing text column '{TEXT_COLUMN}' in CSV. Available columns: {df.columns.tolist()}")

    cleaned_texts = df[TEXT_COLUMN].astype(str).apply(utils.clean_lyrics)

    vocab = _load_lexicon_vocabulary(nrc_lexicon_txt)
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_all = vectorizer.transform(cleaned_texts)

    # One consistent split across all labels (multi-label rows stay together).
    idx = df.index.to_numpy()
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    X_train = X_all[idx_train]

    classifiers: Dict[str, MultinomialNB] = {}
    for emotion in emotions_list:
        y_train = df.loc[idx_train, emotion]
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        classifiers[emotion] = clf

    return NBArtifacts(vectorizer=vectorizer, classifiers=classifiers)


def save_nb_artifacts(artifacts: NBArtifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": artifacts.vectorizer, "classifiers": artifacts.classifiers}, path)


def load_nb_artifacts(path: Path) -> NBArtifacts:
    payload = joblib.load(path)
    return NBArtifacts(vectorizer=payload["vectorizer"], classifiers=payload["classifiers"])


def get_nb_artifacts() -> NBArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = train_nb()
    return _ARTIFACTS


def preprocess_lyrics(new_lyrics: str):
    """Clean + vectorize lyrics for the NB model."""
    artifacts = get_nb_artifacts()
    cleaned = utils.clean_lyrics(new_lyrics)
    return artifacts.vectorizer.transform([cleaned])


def predict_emotions(preprocessed_lyrics) -> Dict[str, float]:
    """Return per-emotion positive-class probabilities (rounded to 2 decimals)."""
    artifacts = get_nb_artifacts()
    scores: Dict[str, float] = {}
    for emotion in emotions_list:
        clf = artifacts.classifiers.get(emotion)
        if clf is None:
            scores[emotion] = float("nan")
            continue
        prob_pos = float(clf.predict_proba(preprocessed_lyrics)[0][1])
        scores[emotion] = round(prob_pos, 2)
    return scores