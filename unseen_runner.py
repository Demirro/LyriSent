# LyriSent_Bert/unseen_runner.py
"""Shared pipeline for NB + BERT + OpenAI on unseen lyrics (CLI and web)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import bert_class
import nb_class
import utils
from GPT.OpenAI import check_sentiment_openai
from config import EMOTION_LABELS, OPENAI_MODEL, OPENAI_PROMPTING_MODES, RESULTS_DIR
from genius import fetch_lyrics


def parse_openai_json_emotions(raw: str) -> dict[str, int | None]:
    try:
        obj = json.loads(raw)
    except Exception:
        return {emotion: None for emotion in EMOTION_LABELS}

    emotions_obj = obj.get("emotions", {})
    if not isinstance(emotions_obj, dict):
        return {emotion: None for emotion in EMOTION_LABELS}

    parsed: dict[str, int | None] = {}
    for emotion in EMOTION_LABELS:
        value = emotions_obj.get(emotion)
        if value in (0, 1, "0", "1", True, False):
            parsed[emotion] = int(bool(value)) if value in (True, False) else int(value)
        else:
            parsed[emotion] = None
    return parsed


def predict_openai_modes(cleaned_lyrics: str, *, openai_api_key: str | None = None) -> dict[str, dict[str, Any]]:
    mode_results: dict[str, dict[str, Any]] = {}
    modes = tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)

    for mode in modes:
        try:
            payload, meta = check_sentiment_openai(
                {"lyrics": cleaned_lyrics},
                return_metadata=True,
                prompting_mode=mode,
                openai_api_key=openai_api_key,
            )
            emotions = parse_openai_json_emotions(payload)
            mode_results[mode] = {
                "emotions": emotions,
                "metadata": meta,
                "raw": payload,
            }
        except Exception as exc:
            mode_results[mode] = {
                "emotions": {emotion: None for emotion in EMOTION_LABELS},
                "metadata": {
                    "model": OPENAI_MODEL,
                    "prompting_mode": mode,
                    "error": str(exc),
                },
                "raw": "",
            }
    return mode_results


def predict_nb(cleaned_lyrics: str) -> dict[str, float | None]:
    try:
        preprocessed = nb_class.preprocess_lyrics(cleaned_lyrics)
        predictions = nb_class.predict_emotions(preprocessed)
        return {emotion: predictions.get(emotion, None) for emotion in nb_class.emotions_list}
    except Exception:
        return {emotion: None for emotion in nb_class.emotions_list}


def predict_bert(cleaned_lyrics: str) -> dict[str, float | None]:
    if not bert_class.bert_load_success:
        return {emotion: None for emotion in bert_class.EMOTION_LABELS}

    try:
        predictions = bert_class.predict_emotions_bert(cleaned_lyrics)
        return {emotion: predictions.get(emotion, None) for emotion in bert_class.EMOTION_LABELS}
    except Exception:
        return {emotion: None for emotion in bert_class.EMOTION_LABELS}


def new_results_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(RESULTS_DIR) / f"{timestamp}_{OPENAI_MODEL}_unseen"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_row(
    artist: str,
    track: str,
    cleaned_lyrics: str,
    *,
    use_openai: bool = True,
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    nb_predictions = predict_nb(cleaned_lyrics)
    bert_predictions = predict_bert(cleaned_lyrics)
    openai_mode_results = predict_openai_modes(cleaned_lyrics, openai_api_key=openai_api_key) if use_openai else {}

    row: dict[str, Any] = {
        "Song": track,
        "Artist": artist,
        "Lyrics_excerpt": cleaned_lyrics[:200] + ("..." if len(cleaned_lyrics) > 200 else ""),
    }

    for emotion in EMOTION_LABELS:
        row[f"NB_{emotion}"] = nb_predictions.get(emotion, None)
    for emotion in EMOTION_LABELS:
        row[f"BERT_{emotion}"] = bert_predictions.get(emotion, None)

    modes = tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)
    if not use_openai:
        for mode in modes:
            for emotion in EMOTION_LABELS:
                row[f"OpenAI_{mode}_{emotion}"] = None
            row[f"OpenAI_{mode}_model"] = OPENAI_MODEL
            row[f"OpenAI_{mode}_attempts"] = None
            row[f"OpenAI_{mode}_prompt_tokens"] = None
            row[f"OpenAI_{mode}_completion_tokens"] = None
            row[f"OpenAI_{mode}_total_tokens"] = None
            row[f"OpenAI_{mode}_estimated_cost_usd"] = None
            row[f"OpenAI_{mode}_metadata"] = {"skipped": True}
    else:
        for mode, mode_data in openai_mode_results.items():
            emotions = mode_data.get("emotions", {})
            for emotion in EMOTION_LABELS:
                row[f"OpenAI_{mode}_{emotion}"] = emotions.get(emotion, None)

            meta = mode_data.get("metadata", {})
            row[f"OpenAI_{mode}_model"] = meta.get("model", OPENAI_MODEL)
            row[f"OpenAI_{mode}_attempts"] = meta.get("attempts", None)
            row[f"OpenAI_{mode}_prompt_tokens"] = meta.get("prompt_tokens", None)
            row[f"OpenAI_{mode}_completion_tokens"] = meta.get("completion_tokens", None)
            row[f"OpenAI_{mode}_total_tokens"] = meta.get("total_tokens", None)
            row[f"OpenAI_{mode}_estimated_cost_usd"] = meta.get("estimated_cost_usd", None)
            row[f"OpenAI_{mode}_metadata"] = meta

    return row


def save_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    prediction_path = output_dir / "unseen_song_predictions.csv"
    metadata_path = output_dir / "unseen_song_openai_metadata.jsonl"

    df.to_csv(prediction_path, index=False, encoding="utf-8")

    modes = tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for row in rows:
            base = {
                "song": row.get("Song"),
                "artist": row.get("Artist"),
            }
            for mode in modes:
                meta_key = f"OpenAI_{mode}_metadata"
                data = row.get(meta_key, {})
                handle.write(json.dumps({**base, "mode": mode, **data}, ensure_ascii=False) + "\n")


def process_songs(
    songs: list[tuple[str, str]],
    *,
    use_openai: bool = True,
    save: bool = True,
    output_dir: Path | None = None,
    genius_token: str | None = None,
    openai_api_key: str | None = None,
) -> tuple[list[dict[str, Any]], Path | None, list[dict[str, str]]]:
    """
    Fetch lyrics per (artist, track), run models, optionally write CSV under RESULTS_DIR.
    Returns (rows, output_dir or None if save=False, issues log).
    """
    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    for artist, track in songs:
        key = (artist.casefold().strip(), track.casefold().strip())
        if key in seen_pairs:
            issues.append({"artist": artist, "track": track, "reason": "duplicate"})
            continue
        seen_pairs.add(key)

        lyrics_payload = fetch_lyrics(artist, track, genius_token=genius_token)
        lyrics = (lyrics_payload or {}).get("lyrics", "")
        if not lyrics:
            issues.append({"artist": artist, "track": track, "reason": "no_lyrics"})
            continue

        cleaned = utils.clean_lyrics(lyrics)
        rows.append(
            build_row(artist, track, cleaned, use_openai=use_openai, openai_api_key=openai_api_key)
        )

    out: Path | None = None
    if save and rows:
        out = output_dir or new_results_output_dir()
        save_results(rows, out)

    return rows, out, issues


def process_pasted_lyrics(
    lyrics: str,
    *,
    title: str = "Custom lyrics",
    artist: str = "(pasted)",
    use_openai: bool = True,
    save: bool = False,
    output_dir: Path | None = None,
    openai_api_key: str | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    cleaned = utils.clean_lyrics(lyrics)
    if not cleaned.strip():
        return [], None

    row = build_row(artist, title, cleaned, use_openai=use_openai, openai_api_key=openai_api_key)
    rows = [row]
    out: Path | None = None
    if save:
        out = output_dir or new_results_output_dir()
        save_results(rows, out)
    return rows, out
