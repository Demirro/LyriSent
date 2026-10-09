# LyriSent_Bert/web_app.py — Flask UI: browse results + run predictions
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from config import EMOTION_LABELS, PUBLIC_MAX_LYRICS_CHARS, PUBLIC_MAX_SONGS, PUBLIC_MODE, RESULTS_DIR
from unseen_runner import process_pasted_lyrics, process_songs

app = Flask(__name__)
app.config.setdefault("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "dev-change-me"))


def _is_local_request() -> bool:
    addr = (request.remote_addr or "").strip().lower()
    return addr in ("127.0.0.1", "::1", "localhost") or addr.startswith("127.")


def _json_safe(obj):
    """Recursively convert numpy scalars and other non-JSON-native values."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _shown_path(path) -> str:
    return "" if PUBLIC_MODE else str(path)


def _resolve_run_dir(run_name: str) -> Path | None:
    """Resolve a run folder that must be a direct child of RESULTS_DIR (name only, no subpaths)."""
    raw = str(run_name).strip()
    if not raw:
        return None
    name = Path(raw).name
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        return None
    base = RESULTS_DIR.resolve()
    candidate = (RESULTS_DIR / name).resolve()
    if not candidate.is_dir():
        return None
    parent = candidate.parent.resolve()
    if os.path.normcase(str(parent)) != os.path.normcase(str(base)):
        return None
    return candidate


def parse_song_lines(text: str) -> list[tuple[str, str]]:
    """One song per line: tab, |, em dash, en dash, or ' - ' between artist and title."""
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        artist, track = "", ""
        if "\t" in line:
            parts = line.split("\t", 1)
            artist, track = parts[0].strip(), parts[1].strip()
        elif "|" in line:
            parts = line.split("|", 1)
            artist, track = parts[0].strip(), parts[1].strip()
        elif " — " in line:
            parts = line.split(" — ", 1)
            artist, track = parts[0].strip(), parts[1].strip()
        elif " – " in line:
            parts = line.split(" – ", 1)
            artist, track = parts[0].strip(), parts[1].strip()
        elif " - " in line:
            parts = line.split(" - ", 1)
            artist, track = parts[0].strip(), parts[1].strip()
        if artist and track:
            out.append((artist, track))
    return out


def _read_csv_table(path: Path) -> dict | None:
    """Parse a CSV into headers + body rows; blank first-column header becomes 'method'."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return None
    headers = list(rows[0])
    if headers and str(headers[0]).strip() == "":
        headers[0] = "method"
    return {"headers": headers, "rows": rows[1:]}


def _eval_summary_payload(run_name: str) -> dict | None:
    run_dir = _resolve_run_dir(run_name)
    if not run_dir:
        return None
    safe = Path(str(run_name)).name
    methods = _read_csv_table(run_dir / "methods_comparison_summary.csv")
    runtime = _read_csv_table(run_dir / "runtime_cost_summary.csv")
    return {
        "run": safe,
        "path": _shown_path(run_dir.resolve()),
        "methods_comparison": methods,
        "runtime_cost": runtime,
    }


def _cross_run_payload() -> dict:
    """Read cross-run analysis artifacts from comparison_results/cross_run_analysis."""
    cross_dir = (RESULTS_DIR / "cross_run_analysis").resolve()
    if not cross_dir.is_dir():
        return {"path": _shown_path(cross_dir), "tables": [], "images": []}

    tables: list[dict] = []
    for p in sorted(cross_dir.glob("*.csv"), key=lambda x: x.name.lower()):
        parsed = _read_csv_table(p)
        if parsed is None:
            continue
        tables.append(
            {
                "name": p.name,
                "headers": parsed["headers"],
                "rows": parsed["rows"],
            }
        )

    images = [
        {"name": p.name, "url": url_for("serve_cross_run_file", filename=p.name)}
        for p in sorted(cross_dir.glob("*.png"), key=lambda x: x.name.lower())
    ]
    return {"path": _shown_path(cross_dir), "tables": tables, "images": images}


def _list_unseen_runs() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for p in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        csv_path = p / "unseen_song_predictions.csv"
        if csv_path.is_file():
            runs.append(
                {
                    "name": p.name,
                    "path": _shown_path(p.resolve()),
                    "type": "unseen",
                }
            )
    return runs


def _list_eval_runs() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for p in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        if (p / "methods_comparison_summary.csv").is_file():
            fig_dir = p / "figures"
            pngs = sorted([f.name for f in fig_dir.glob("*.png")]) if fig_dir.is_dir() else []
            runs.append(
                {
                    "name": p.name,
                    "path": _shown_path(p.resolve()),
                    "type": "eval",
                    "figure_files": pngs,
                    "has_figures": bool(pngs),
                }
            )
    return runs


def _row_to_chart_payload(row: dict) -> dict:
    nb = {e: row.get(f"NB_{e}") for e in EMOTION_LABELS}
    bert = {e: row.get(f"BERT_{e}") for e in EMOTION_LABELS}
    openai: dict[str, dict[str, object]] = {}
    prefix = "OpenAI_"
    for key, val in row.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if key.endswith("_metadata") or key.endswith("_model") or key.endswith("_attempts"):
            continue
        if key.endswith("_prompt_tokens") or key.endswith("_completion_tokens") or key.endswith("_total_tokens"):
            continue
        if key.endswith("_estimated_cost_usd"):
            continue
        for emo in EMOTION_LABELS:
            suf = f"_{emo}"
            if key.endswith(suf):
                mode = key[len(prefix) : -len(suf)]
                if mode:
                    openai.setdefault(mode, {})[emo] = val
                break
    return {
        "nb": nb,
        "bert": bert,
        "openai_by_mode": openai,
        "meta": {"song": row.get("Song"), "artist": row.get("Artist")},
    }


def _extract_keys(data: dict) -> tuple[str | None, str | None]:
    ot = (data.get("openai_api_key") or "").strip() or None
    gt = (data.get("genius_token") or "").strip() or None
    if PUBLIC_MODE:
        gt = None
    return ot, gt


def _public_request_error(lyrics: str, pairs: list, use_openai: bool, openai_key: str | None) -> str | None:
    if not PUBLIC_MODE:
        return None
    if use_openai and not openai_key:
        return "Enter your own OpenAI API key to include OpenAI."
    if len(pairs) > PUBLIC_MAX_SONGS:
        return f"At most {PUBLIC_MAX_SONGS} songs per request."
    if len(lyrics) > PUBLIC_MAX_LYRICS_CHARS:
        return f"Lyrics are limited to {PUBLIC_MAX_LYRICS_CHARS} characters."
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        unseen_runs=_list_unseen_runs(),
        eval_runs=_list_eval_runs(),
        emotion_labels=EMOTION_LABELS,
        public_mode=PUBLIC_MODE,
    )


@app.get("/api/runs")
def api_runs():
    return jsonify({"unseen": _list_unseen_runs(), "eval": _list_eval_runs()})


@app.get("/api/unseen_csv/<run_name>")
def api_unseen_csv(run_name):
    run_dir = _resolve_run_dir(run_name)
    if not run_dir:
        return jsonify({"error": "not found"}), 404
    csv_path = run_dir / "unseen_song_predictions.csv"
    if not csv_path.is_file():
        return jsonify({"error": "not found"}), 404
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    headers = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return jsonify(
        {
            "path": _shown_path(csv_path.resolve()),
            "headers": headers,
            "rows": body,
            "text": text,
        }
    )


def _eval_figures_json(run_name: str):
    """Shared logic for listing figure PNGs in a run folder."""
    run_dir = _resolve_run_dir(run_name)
    if not run_dir:
        return None
    fig_dir = run_dir / "figures"
    safe_run = Path(str(run_name)).name
    if not fig_dir.is_dir():
        return {"run": safe_run, "path": _shown_path(run_dir.resolve()), "files": []}
    files = []
    for f in sorted(fig_dir.glob("*.png"), key=lambda x: x.name.lower()):
        files.append(
            {
                "name": f.name,
                "url": url_for("serve_figure", run_name=safe_run, filename=f.name),
            }
        )
    return {"run": safe_run, "path": _shown_path(run_dir.resolve()), "files": files}


# Use <run_name> (not <path:run_name>): <path> is greedy and can prevent matching the trailing /figures.
@app.get("/api/eval_run/<run_name>/figures")
def api_eval_figures(run_name):
    payload = _eval_figures_json(run_name)
    if payload is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(payload)


@app.get("/api/eval_figures")
def api_eval_figures_query():
    run_name = (request.args.get("run") or "").strip()
    if not run_name:
        return jsonify({"error": "missing run query parameter"}), 400
    payload = _eval_figures_json(run_name)
    if payload is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(payload)


@app.get("/api/eval_run/<run_name>/summary")
def api_eval_summary(run_name):
    payload = _eval_summary_payload(run_name)
    if payload is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(payload)


@app.get("/api/eval_summary")
def api_eval_summary_query():
    run_name = (request.args.get("run") or "").strip()
    if not run_name:
        return jsonify({"error": "missing run query parameter"}), 400
    payload = _eval_summary_payload(run_name)
    if payload is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(payload)


@app.post("/api/open-folder")
def api_open_folder():
    if PUBLIC_MODE or not _is_local_request():
        return jsonify({"error": "only allowed from localhost"}), 403
    data = request.get_json(silent=True) or {}
    run_name = data.get("run_name") or data.get("path")
    if not run_name:
        return jsonify({"error": "run_name required"}), 400
    run_dir = _resolve_run_dir(str(run_name))
    if not run_dir:
        return jsonify({"error": "not found"}), 404
    path_str = str(run_dir)
    try:
        if sys.platform == "win32":
            os.startfile(path_str)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path_str], start_new_session=True)
        else:
            subprocess.Popen(["xdg-open", path_str], start_new_session=True)
    except OSError as e:
        return jsonify({"error": str(e), "path": path_str}), 500
    return jsonify({"ok": True, "path": path_str})


@app.get("/api/cross_run")
def api_cross_run():
    return jsonify(_cross_run_payload())


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    use_openai = bool(data.get("use_openai", False))
    openai_key, genius_tok = _extract_keys(data)

    lyrics = (data.get("lyrics") or "").strip()
    artist = (data.get("artist") or "").strip()
    track = (data.get("track") or "").strip()
    song_lines = (data.get("song_list") or "").strip()
    songs_json = data.get("songs")

    if songs_json and isinstance(songs_json, list):
        pairs: list[tuple[str, str]] = []
        for item in songs_json:
            if not isinstance(item, dict):
                continue
            a = (item.get("artist") or "").strip()
            t = (item.get("track") or "").strip()
            if a and t:
                pairs.append((a, t))
    elif song_lines:
        pairs = parse_song_lines(song_lines)
    else:
        pairs = []

    public_error = _public_request_error(lyrics, pairs, use_openai, openai_key)
    if public_error:
        return jsonify({"error": public_error}), 400

    if lyrics:
        rows, _ = process_pasted_lyrics(
            lyrics,
            title=data.get("title") or "Custom lyrics",
            artist=artist or "(pasted)",
            use_openai=use_openai,
            save=False,
            openai_api_key=openai_key,
        )
        return (
            jsonify(
                _json_safe(
                    {
                        "rows": rows,
                        "charts": [_row_to_chart_payload(r) for r in rows],
                        "issues": [],
                    }
                )
            )
            if rows
            else (jsonify({"error": "No lyrics after cleaning."}), 400)
        )

    if pairs:
        rows, _, issues = process_songs(
            pairs,
            use_openai=use_openai,
            save=False,
            genius_token=genius_tok,
            openai_api_key=openai_key,
        )
        if not rows and issues:
            return jsonify({"error": "No songs could be fetched.", "issues": issues, "rows": [], "charts": []}), 400
        return jsonify(
            _json_safe(
                {
                    "rows": rows,
                    "charts": [_row_to_chart_payload(r) for r in rows],
                    "issues": issues,
                }
            )
        )

    if artist and track:
        rows, _, issues = process_songs(
            [(artist, track)],
            use_openai=use_openai,
            save=False,
            genius_token=genius_tok,
            openai_api_key=openai_key,
        )
        if not rows:
            return (
                jsonify(
                    {
                        "error": "No lyrics available (fetch failed or empty).",
                        "issues": issues,
                        "rows": [],
                        "charts": [],
                    }
                ),
                400,
            )
        return jsonify(
            _json_safe(
                {
                    "rows": rows,
                    "charts": [_row_to_chart_payload(r) for r in rows],
                    "issues": issues,
                }
            )
        )

    return jsonify({"error": "Provide lyrics, artist+track, or a song list / songs[]."}), 400


@app.post("/api/predict_save")
def api_predict_save():
    if PUBLIC_MODE:
        return jsonify({"error": "Saving is disabled on the public site."}), 403
    data = request.get_json(silent=True) or {}
    use_openai = bool(data.get("use_openai", False))
    openai_key, genius_tok = _extract_keys(data)

    lyrics = (data.get("lyrics") or "").strip()
    artist = (data.get("artist") or "").strip()
    track = (data.get("track") or "").strip()
    song_lines = (data.get("song_list") or "").strip()
    songs_json = data.get("songs")

    if songs_json and isinstance(songs_json, list):
        pairs = []
        for item in songs_json:
            if not isinstance(item, dict):
                continue
            a = (item.get("artist") or "").strip()
            t = (item.get("track") or "").strip()
            if a and t:
                pairs.append((a, t))
    elif song_lines:
        pairs = parse_song_lines(song_lines)
    else:
        pairs = []

    if lyrics:
        rows, out = process_pasted_lyrics(
            lyrics,
            title=data.get("title") or "Custom lyrics",
            artist=artist or "(pasted)",
            use_openai=use_openai,
            save=True,
            openai_api_key=openai_key,
        )
        if not rows or not out:
            return jsonify({"error": "Nothing saved (empty lyrics)."}), 400
        return jsonify(
            _json_safe(
                {
                    "saved_to": str(out.resolve()),
                    "rows": rows,
                    "charts": [_row_to_chart_payload(r) for r in rows],
                    "issues": [],
                }
            )
        )

    if pairs:
        rows, out, issues = process_songs(
            pairs,
            use_openai=use_openai,
            save=True,
            genius_token=genius_tok,
            openai_api_key=openai_key,
        )
        if not rows or not out:
            return jsonify({"error": "Nothing saved (all fetches failed).", "issues": issues}), 400
        return jsonify(
            _json_safe(
                {
                    "saved_to": str(out.resolve()),
                    "rows": rows,
                    "charts": [_row_to_chart_payload(r) for r in rows],
                    "issues": issues,
                }
            )
        )

    if artist and track:
        rows, out, issues = process_songs(
            [(artist, track)],
            use_openai=use_openai,
            save=True,
            genius_token=genius_tok,
            openai_api_key=openai_key,
        )
        if not rows or not out:
            return jsonify({"error": "Nothing saved.", "issues": issues}), 400
        return jsonify(
            _json_safe(
                {
                    "saved_to": str(out.resolve()),
                    "rows": rows,
                    "charts": [_row_to_chart_payload(r) for r in rows],
                    "issues": issues,
                }
            )
        )

    return jsonify({"error": "Provide lyrics, artist+track, or a song list / songs[]."}), 400


@app.route("/figures/<run_name>/<filename>")
def serve_figure(run_name, filename):
    safe_run = Path(run_name).name
    safe_file = Path(filename).name
    fig_dir = RESULTS_DIR / safe_run / "figures"
    if not fig_dir.is_dir():
        return "Not found", 404
    return send_from_directory(fig_dir, safe_file)


@app.route("/cross_run/<filename>")
def serve_cross_run_file(filename):
    safe_file = Path(filename).name
    cross_dir = RESULTS_DIR / "cross_run_analysis"
    if not cross_dir.is_dir():
        return "Not found", 404
    return send_from_directory(cross_dir, safe_file)


if __name__ == "__main__":
    host = os.getenv("LYRISENT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("LYRISENT_WEB_PORT", "5000"))
    debug = os.getenv("LYRISENT_WEB_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
