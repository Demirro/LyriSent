---
title: LyriSent Bert
emoji: 📚
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

## Local project (CLI + optional web UI)

- **Environment:** copy `.env` with `GENIUS_TOKEN`, and `OPENAI_API_KEY` if you use OpenAI. Install deps: `pip install -r requirements.txt`.
- **Unseen songs (CLI):** `python app.py` — interactive Genius list or custom artist/track; writes a timestamped folder under `comparison_results/` with `unseen_song_predictions.csv`.
- **Figures from an eval run:** `python visualize_results.py` (optionally `--run-dir <folder_name>`).
- **Web UI (Flask):** `python web_app.py` then open http://127.0.0.1:5000 — optional Genius/OpenAI keys in the page (or `.env`); single lyrics, single artist/track, or **batch** lines (`Artist - Track`, `|`, or tab); figure gallery with keyboard navigation; CSV as a scrollable table; **Open folder** / **Copy path** (open only works on localhost).

Note: the YAML header above targets Hugging Face Spaces (`sdk: streamlit`, `app_file: app.py`). This repo’s `app.py` is the **CLI** unseen runner; for a Streamlit Space you would point `app_file` at a separate Streamlit entry script.
