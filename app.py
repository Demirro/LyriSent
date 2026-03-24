# LyriSent_Bert/app.py
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


def _input(message: str, cast_type=str):
    while True:
        try:
            return cast_type(input(message))
        except ValueError:
            print("Invalid input type. Please try again.")
        except EOFError:
            print("\nInput stream closed. Exiting.")
            raise SystemExit(0)


def _collect_songs() -> list[tuple[str, str]]:
    songs_list: list[tuple[str, str]] = []
    prefab_check = input("Do you want to use the prefab list of songs (10 songs)? Yes (y/Y) or No (n/N)\n")

    if prefab_check.casefold() in {"yes", "y"}:
        songs_list = [
            ("Childish Gambino", "This is America"),
            ("brakence", "deepfacke"),
            ("Peter Fox", "Haus am See"),
            ("Feu! Chatterton", "J'ai tout mon temps"),
            ("Bruno Mars", "Treasure"),
            ("Ed Sheeran", "Shape Of You"),
            ("The Japanese House", "Saw You In A Dream"),
            ("Tom Misch", "Disco Yes"),
            ("Radiohead", "Creep"),
            ("Jacob Collier", "Hideaway"),
        ]
    elif prefab_check.casefold() in {"no", "n"}:
        number_of_songs = _input("How many songs do you want to check: ", int)
        for i in range(number_of_songs):
            print(f"Song no. {i + 1}")
            track_name = input("Track Name: ")
            artist = input("Artist: ")
            songs_list.append((artist, track_name))
            print("")
    else:
        print("Invalid input. Please enter Yes (y/Y) or No (n/N)")
        return []

    return songs_list


def _parse_openai_json_emotions(raw: str) -> dict[str, int | None]:
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


def _predict_openai_modes(cleaned_lyrics: str) -> dict[str, dict[str, Any]]:
    mode_results: dict[str, dict[str, Any]] = {}
    modes = tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)

    for mode in modes:
        try:
            payload, meta = check_sentiment_openai(
                {"lyrics": cleaned_lyrics},
                return_metadata=True,
                prompting_mode=mode,
            )
            emotions = _parse_openai_json_emotions(payload)
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


def _predict_nb(cleaned_lyrics: str) -> dict[str, float | None]:
    try:
        preprocessed = nb_class.preprocess_lyrics(cleaned_lyrics)
        predictions = nb_class.predict_emotions(preprocessed)
        return {emotion: predictions.get(emotion, None) for emotion in nb_class.emotions_list}
    except Exception as exc:
        print(f"Naive Bayes prediction failed: {exc}")
        return {emotion: None for emotion in nb_class.emotions_list}


def _predict_bert(cleaned_lyrics: str) -> dict[str, float | None]:
    if not bert_class.bert_load_success:
        print("BERT model not loaded; writing empty BERT predictions.")
        return {emotion: None for emotion in bert_class.EMOTION_LABELS}

    try:
        predictions = bert_class.predict_emotions_bert(cleaned_lyrics)
        return {emotion: predictions.get(emotion, None) for emotion in bert_class.EMOTION_LABELS}
    except Exception as exc:
        print(f"BERT prediction failed: {exc}")
        return {emotion: None for emotion in bert_class.EMOTION_LABELS}


def _results_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(RESULTS_DIR) / f"{timestamp}_{OPENAI_MODEL}_unseen"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        print("No rows to save.")
        return

    df = pd.DataFrame(rows)
    prediction_path = output_dir / "unseen_song_predictions.csv"
    metadata_path = output_dir / "unseen_song_openai_metadata.jsonl"

    df.to_csv(prediction_path, index=False, encoding="utf-8")

    with open(metadata_path, "w", encoding="utf-8") as handle:
        for row in rows:
            base = {
                "song": row.get("Song"),
                "artist": row.get("Artist"),
            }
            for mode in (tuple(m for m in OPENAI_PROMPTING_MODES if str(m).strip()) or ("zero_shot",)):
                meta_key = f"OpenAI_{mode}_metadata"
                data = row.get(meta_key, {})
                handle.write(json.dumps({**base, "mode": mode, **data}, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved OpenAI metadata to: {metadata_path}")


def main():
    print("LyriSent unseen-song test runner")
    print("This runs NB + BERT + OpenAI (all configured prompting modes) on songs not seen during training.")
    print("")

    songs = _collect_songs()
    if not songs:
        return

    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for artist, track in songs:
        key = (artist.casefold().strip(), track.casefold().strip())
        if key in seen_pairs:
            print(f"Skipping duplicate entry: '{track}' by '{artist}'")
            continue
        seen_pairs.add(key)

        print(f"\nProcessing '{track}' by '{artist}'...")
        lyrics_payload = fetch_lyrics(artist, track)
        lyrics = (lyrics_payload or {}).get("lyrics", "")
        if not lyrics:
            print("Could not fetch lyrics. Skipping.")
            continue

        cleaned_lyrics = utils.clean_lyrics(lyrics)

        nb_predictions = _predict_nb(cleaned_lyrics)
        bert_predictions = _predict_bert(cleaned_lyrics)
        openai_mode_results = _predict_openai_modes(cleaned_lyrics)

        row: dict[str, Any] = {
            "Song": track,
            "Artist": artist,
            "Lyrics_excerpt": cleaned_lyrics[:200] + ("..." if len(cleaned_lyrics) > 200 else ""),
        }

        for emotion in EMOTION_LABELS:
            row[f"NB_{emotion}"] = nb_predictions.get(emotion, None)
        for emotion in EMOTION_LABELS:
            row[f"BERT_{emotion}"] = bert_predictions.get(emotion, None)

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

        rows.append(row)
        print("Finished.")

    output_dir = _results_output_path()
    _save_results(rows, output_dir)
    print(f"Processed {len(rows)} songs.")


if __name__ == "__main__":
    main()