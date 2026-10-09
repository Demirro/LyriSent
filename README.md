---
title: LyriSent Bert
emoji: 📚
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

## Local project (CLI + optional web UI)

- **Environment:** copy `.env` with `GENIUS_TOKEN`, and `OPENAI_API_KEY` if you use OpenAI. Install deps: `pip install -r requirements.txt`.
- **Unseen songs (CLI):** `python app.py` — interactive Genius list or custom artist/track; writes a timestamped folder under `comparison_results/` with `unseen_song_predictions.csv`.
- **Figures from an eval run:** `python visualize_results.py` (optionally `--run-dir <folder_name>`).
- **Web UI (Flask):** `python web_app.py` then open http://127.0.0.1:5000 — optional Genius/OpenAI keys in the page (or `.env`); single lyrics, single artist/track, or **batch** lines (`Artist - Track`, `|`, or tab); figure gallery with keyboard navigation; CSV as a scrollable table; **Open folder** / **Copy path** (open only works on localhost).

## Deployment (public web app)

The YAML header above runs the Space as a Docker app (`Dockerfile`, port 7860). The image starts `web_app.py` under gunicorn in public mode (`LYRISENT_PUBLIC=1`):

- OpenAI is only used with a key the visitor enters; the server never falls back to its own key.
- Saving runs and opening folders are disabled, folder paths are hidden.
- Batch requests are limited to `LYRISENT_PUBLIC_MAX_SONGS` songs (default 5), pasted lyrics to `LYRISENT_PUBLIC_MAX_LYRICS_CHARS` characters (default 20000).

The fine-tuned weights are not part of the repository. If `trained_bert_model/model.safetensors` is missing, BERT is loaded from the Hugging Face model repo named in `LYRISENT_BERT_REPO`.

Space settings:

- Variable `LYRISENT_BERT_REPO`, e.g. `Demirro/lyrisent-bert`
- Secret `HF_TOKEN` (read access) if that model repo is private
- Secret `GENIUS_TOKEN` for fetching lyrics by artist and title

Any other Docker host works the same way: `docker build -t lyrisent .` and `docker run -p 7860:7860 -e LYRISENT_BERT_REPO=... -e GENIUS_TOKEN=... lyrisent`.
