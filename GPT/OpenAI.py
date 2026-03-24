import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from config import EMOTION_LABELS, OPENAI_MODEL

load_dotenv()

client = OpenAI()
OPENAI_DEBUG_VERBOSE = os.getenv("OPENAI_DEBUG_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}

# Approximate USD pricing per 1k tokens.
MODEL_PRICING_USD_PER_1K = {
    "gpt-4": {"input": 0.03, "output": 0.06},
}


def _system_prompt_json() -> str:
    return (
        "You are a strict JSON API for multi-label emotion tagging.\n"
        "Output must be EXACTLY one JSON object with this exact shape and keys:\n"
        '{"emotions":{"Joy":0,"Trust":0,"Fear":0,"Surprise":0,"Sadness":0,"Disgust":0,"Anger":0,"Anticipation":0}}\n'
        "Hard constraints:\n"
        "- Output ONLY JSON. No markdown. No code fences. No explanation.\n"
        "- Keep all 8 emotion keys exactly as shown and in the same casing.\n"
        "- Each value must be integer 0 or 1 only.\n"
        "- Do not include any other keys.\n"
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


def _few_shot_examples_block() -> str:
    return (
        'Example 1:\n'
        'Lyrics: "Sunlight on my face, I trust the road ahead and smile through the day."\n'
        'Output: {"emotions":{"Joy":1,"Trust":1,"Fear":0,"Surprise":0,"Sadness":0,"Disgust":0,"Anger":0,"Anticipation":1}}\n\n'
        'Example 2:\n'
        'Lyrics: "I shake in the dark, every shadow feels like danger and my chest is tight."\n'
        'Output: {"emotions":{"Joy":0,"Trust":0,"Fear":1,"Surprise":0,"Sadness":1,"Disgust":0,"Anger":0,"Anticipation":0}}\n\n'
        'Example 3:\n'
        'Lyrics: "You lied again, I am sick of this and furious at what we have become."\n'
        'Output: {"emotions":{"Joy":0,"Trust":0,"Fear":0,"Surprise":0,"Sadness":1,"Disgust":1,"Anger":1,"Anticipation":0}}\n\n'
    )


def _build_user_prompts(lyrics: str, prompting_mode: str) -> list[str]:
    mode = (prompting_mode or "zero_shot").strip().lower()
    few_shot_prefix = _few_shot_examples_block() if mode == "few_shot" else ""
    analyze_header = (
        "Analyze the target lyrics and return strict JSON only.\n"
        "Required format:\n"
        '{"emotions":{"Joy":0,"Trust":0,"Fear":0,"Surprise":0,"Sadness":0,"Disgust":0,"Anger":0,"Anticipation":0}}\n'
    )
    retry_header = (
        "Retry. Your previous output was invalid.\n"
        "Return ONLY this JSON schema with integer 0/1 values and no extra keys:\n"
        '{"emotions":{"Joy":0,"Trust":0,"Fear":0,"Surprise":0,"Sadness":0,"Disgust":0,"Anger":0,"Anticipation":0}}\n'
    )
    target_suffix = "Target lyrics:\n" + lyrics
    return [
        analyze_header + ("\n" + few_shot_prefix if few_shot_prefix else "") + target_suffix,
        retry_header + ("\n" + few_shot_prefix if few_shot_prefix else "") + target_suffix,
    ]


def _structured_emotion_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "emotion_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "emotions": {
                        "type": "object",
                        "properties": {label: {"type": "integer", "enum": [0, 1]} for label in EMOTION_LABELS},
                        "required": EMOTION_LABELS,
                        "additionalProperties": False,
                    }
                },
                "required": ["emotions"],
                "additionalProperties": False,
            },
        },
    }


def _supports_structured_outputs(model_name: str) -> bool:
    """Return whether the model likely supports json_schema output."""
    name = (model_name or "").lower()
    return ("gpt-4o" in name) or name.startswith("gpt-5")


def _extract_first_json_object(raw: str) -> dict | None:
    """Extract the first valid JSON object from text."""
    if not raw:
        return None

    start_idx = raw.find("{")
    while start_idx != -1:
        depth = 0
        for i in range(start_idx, len(raw)):
            ch = raw[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start_idx : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
        start_idx = raw.find("{", start_idx + 1)
    return None


def _normalize_emotions_payload(obj: dict) -> dict | None:
    """Validate and normalize the emotions payload."""
    emotions = obj.get("emotions")
    if not isinstance(emotions, dict):
        return None

    normalized = {}
    for label in EMOTION_LABELS:
        v = emotions.get(label, None)
        if v in (1, "1", True):
            normalized[label] = 1
        elif v in (0, "0", False):
            normalized[label] = 0
        else:
            return None

    return {"emotions": normalized}


def estimate_openai_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (
        (float(prompt_tokens) / 1000.0) * float(pricing["input"])
        + (float(completion_tokens) / 1000.0) * float(pricing["output"])
    )


def check_sentiment_openai(
    song: Dict[str, Any],
    *,
    return_metadata: bool = False,
    prompting_mode: str = "zero_shot",
):
    """Return the model output as a string."""
    lyrics = song["lyrics"]

    metadata: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "prompting_mode": prompting_mode,
        "structured_output_requested": False,
        "structured_output_supported": _supports_structured_outputs(OPENAI_MODEL),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "format": "",
        "attempts": 0,
    }

    def _debug(msg: str) -> None:
        if OPENAI_DEBUG_VERBOSE:
            print(msg)

    # Try JSON twice, with a stricter retry prompt on attempt 2.
    user_prompts = _build_user_prompts(lyrics, prompting_mode)
    for attempt_idx, user_prompt in enumerate(user_prompts, start=1):
        metadata["attempts"] = attempt_idx
        _debug(f"[OpenAI debug] JSON attempt {attempt_idx}/{len(user_prompts)}")
        try:
            request_kwargs = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": _system_prompt_json()},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_completion_tokens": 120,
            }
            if _supports_structured_outputs(OPENAI_MODEL):
                request_kwargs["response_format"] = _structured_emotion_schema()
                metadata["structured_output_requested"] = True
                _debug("[OpenAI debug] Using structured outputs json_schema mode.")
            else:
                _debug("[OpenAI debug] Model likely lacks structured outputs support; using prompt-only JSON mode.")

            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as e:
                err = str(e)
                # Backward compatibility for models that expect `max_tokens`.
                if "max_completion_tokens" in err and "unsupported" in err:
                    _debug("[OpenAI debug] Model does not support max_completion_tokens. Retrying with max_tokens.")
                    request_kwargs.pop("max_completion_tokens", None)
                    request_kwargs["max_tokens"] = 120
                    response = client.chat.completions.create(**request_kwargs)
                else:
                    raise

            usage = getattr(response, "usage", None)
            if usage is not None:
                p = int(getattr(usage, "prompt_tokens", 0) or 0)
                c = int(getattr(usage, "completion_tokens", 0) or 0)
                t = int(getattr(usage, "total_tokens", 0) or 0)
                metadata["prompt_tokens"] += p
                metadata["completion_tokens"] += c
                metadata["total_tokens"] += t
                metadata["estimated_cost_usd"] = estimate_openai_cost_usd(
                    OPENAI_MODEL,
                    metadata["prompt_tokens"],
                    metadata["completion_tokens"],
                )

            content = response.choices[0].message.content if response.choices else ""
            if not content:
                _debug("[OpenAI debug] Empty response content on JSON attempt.")
                continue

            obj = None
            try:
                obj = json.loads(content)
            except Exception:
                obj = _extract_first_json_object(content)

            if not isinstance(obj, dict):
                _debug("[OpenAI debug] JSON parse failed; no JSON object found.")
                _debug(f"[OpenAI debug] User prompt:\n{user_prompt}")
                _debug(f"[OpenAI debug] Raw response:\n{content}")
                continue

            normalized = _normalize_emotions_payload(obj)
            if normalized is None:
                _debug("[OpenAI debug] JSON validation failed; schema/values invalid.")
                _debug(f"[OpenAI debug] User prompt:\n{user_prompt}")
                _debug(f"[OpenAI debug] Parsed object:\n{json.dumps(obj, ensure_ascii=False)}")
                continue

            if "lyrics_excerpt" in obj:
                normalized["lyrics_excerpt"] = obj["lyrics_excerpt"]

            payload = json.dumps(normalized, ensure_ascii=False)
            metadata["format"] = "json"
            _debug("[OpenAI debug] JSON attempt succeeded.")
            if return_metadata:
                return payload, metadata
            return payload
        except Exception as e:
            _debug(f"[OpenAI debug] Exception during JSON attempt: {e}")
            _debug(f"[OpenAI debug] User prompt:\n{user_prompt}")
            continue

    print("Warning: OpenAI API did not return valid JSON after retries.")
    if return_metadata:
        return "", metadata
    return ""
