import json
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from config import EMOTION_LABELS, OPENAI_MODEL

load_dotenv()

client = OpenAI()
OPENAI_SEED = 42


def _system_prompt_json() -> str:
    emotions = ", ".join(EMOTION_LABELS)
    return (
        "You are a tone analyzer for lyrics.\n"
        "Return ONLY valid JSON.\n"
        'Schema: {"emotions": {"Joy":0|1, "Trust":0|1, ...}, "lyrics_excerpt": "<max 50 chars + ...>"}\n'
        f"Emotions: {emotions}\n"
        "Rules:\n"
        "- Output must be a single JSON object and nothing else.\n"
        "- Each emotion must be present and must be exactly 0 or 1.\n"
        "- Do not add extra keys besides 'emotions' and 'lyrics_excerpt'.\n"
        "- Think as if labels represent agreement of many annotators.\n"
    )


def _system_prompt_csv() -> str:
    return (
        "You are a tone analyzer for lyrics. Your ONLY output is the sentiment analysis in CSV format. "
        "Provide ONLY a header row followed by ONE data row for the analyzed lyrics. "
        "Systematically analyze the mood of the song lyrics according to Plutchiks 8 core emotions: "
        "Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation. "
        "Give the results in a tabular form using csv with the table header: "
        "Song Name,Artists,Lyrics,Joy,Trust,Fear,Surprise,Sadness,Disgust,Anger,Anticipation. "
        "Limit the output of the Lyrics column to only 50 symbols and add ... . "
        "Wrap the lyrics in single quotes. "
        "If a song displays any of the emotions it should get a 1 in the given column, if not a 0. "
        "Think about annotating as if you were the result or agreement of multiple hundreds of annotators. "
        "ABSOLUTELY no other text, explanation, or formatting outside of the CSV header and data row. "
        "Do NOT include markdown ```csv ``` formatting."
    )


def check_sentiment_openai(song: Dict[str, Any]) -> str:
    """
    Returns either JSON (preferred) or CSV (fallback) as a string.
    Callers should parse robustly (accepting JSON or CSV).
    """
    payload = check_sentiment_openai_with_meta(song)
    return payload.get("content", "")


def _extract_usage(response) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def check_sentiment_openai_with_meta(song: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns payload with response content plus metadata:
    {
      "content": str,
      "model": str,
      "format": "json" | "csv" | "none",
      "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    }
    """
    lyrics = song["lyrics"]

    # 1) Prefer JSON structured output.
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt_json()},
                {"role": "user", "content": "Analyze the following lyrics:\n" + lyrics},
            ],
            temperature=0,
            top_p=1,
            seed=OPENAI_SEED,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content if response.choices else ""
        if content:
            # quick validation that it's JSON
            json.loads(content)
            return {
                "content": content,
                "model": OPENAI_MODEL,
                "format": "json",
                "usage": _extract_usage(response),
            }
    except Exception:
        pass

    # 2) Fallback to CSV prompt (older behavior).
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt_csv()},
            {"role": "user", "content": "Analyze the following lyrics:\n" + lyrics},
        ],
        temperature=0,
        top_p=1,
        seed=OPENAI_SEED,
    )

    if response.choices and response.choices[0].message and response.choices[0].message.content:
        return {
            "content": response.choices[0].message.content,
            "model": OPENAI_MODEL,
            "format": "csv",
            "usage": _extract_usage(response),
        }

    print("Warning: OpenAI API returned no message content.")
    return {
        "content": "",
        "model": OPENAI_MODEL,
        "format": "none",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
