---
title: LyriSent Bert
emoji: 📚
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# LyriSent

Multi-label emotion classification for song lyrics, comparing a lexicon-based Naive Bayes baseline, a fine-tuned BERT transformer, and prompt-based GPT classification.

This is a university research project in Digital Humanities. The focus is on method comparison, evaluation design and reproducibility rather than raw performance numbers.

## 1. Project Overview

LyriSent classifies **multiple co-existing emotions** in song lyrics using Plutchik's eight emotion categories:

Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation

### Methods

- **Naive Bayes (NB):** trains 8 independent binary classifiers (one per emotion) on a vocabulary restricted to NRC Hashtag Emotion Lexicon terms. Each classifier returns a probability; thresholds are tuned separately in the evaluation pipeline.

- **BERT:** `bert-base-uncased` fine-tuned end-to-end on the annotated lyric dataset. Uses `BCEWithLogitsLoss` with per-emotion positive class weights (negative/positive count ratio) to handle label imbalance. Training runs for 25 epochs; the best checkpoint is saved when validation macro F1 improves. There is no early stopping, so all epochs run.

- **GPT (`gpt-4o` / `gpt-5.4`):** classifies lyrics via structured JSON output with `temperature=0` and all randomness penalties set to zero for maximum consistency. Supports zero-shot and few-shot modes; few-shot provides three labeled lyric examples covering distinct emotional profiles. If the first response fails to parse, a stricter retry prompt is sent automatically (fallback for older system). Parse failures are logged to `openai_parse_failures.jsonl` for audit.

The three approaches are compared on predictive performance, reproducibility and operational cost.

## 2. Dataset

**Training data** (`EmotionWheelFinal (1).csv`): multi-label annotations for song lyrics with Plutchik's eight emotions:

> Edmonds, D., & Sedoc, J. (2021). *Multi-Emotion Classification for Song Lyrics*. WASSA 2021, 221–235.  
> https://aclanthology.org/2021.wassa-1.24/

**Evaluation data** (`RadaNewAnnotation (1).csv`): 100 songs annotated with Ekman's six basic emotions:

> Mihalcea, R., & Strapparava, C. (2012). *Lyrics, Music, and Emotions*. EMNLP-CoNLL 2012, 590–599.

Evaluation runs split the 100 songs into separate threshold-tuning and evaluation subsets. Because the evaluation set covers six of the eight emotions (Anger, Disgust, Fear, Joy, Sadness, Surprise), all evaluation metrics are computed over these six; Trust and Anticipation are predicted but not scored.

Additional resources:
- NRC Hashtag Emotion Lexicon (Naive Bayes scoring)
- Genius API (unseen lyrics retrieval)

## 3. Repository Structure

- `config.py`: central configuration (paths, labels, hyperparameters)
- `utils.py`: shared preprocessing (lyrics cleaning)
- `nb_class.py`: Naive Bayes implementation
- `train_bert.py` / `bert_class.py`: BERT training and inference
- `GPT/OpenAI.py`: GPT interface with structured output handling
- `genius.py`: Genius API wrapper for lyrics fetching
- `testing.py`: evaluation pipeline
- `visualize_results.py`: result visualization and cross-run aggregation
- `app.py`: CLI demo for unseen songs
- `unseen_runner.py`: unseen song prediction logic (used by both `app.py` and `web_app.py`)
- `visualize_unseen_runs.py`: per-song emotion plots for unseen runs
- `backfill_runtime_cost_usd.py`: recalculates GPT cost estimates using updated pricing tables
- `web_app.py`: web interface (local, or public via Docker)
- `Dockerfile`: container for the public web app

## 4. Configuration (`config.py`)

All hardcoded values live in `config.py`. Changing a value there changes it for every module that imports it.

**Paths** (relative to project root):

| Variable | Default |
|---|---|
| `DATA_DIR` | `data/` |
| `MODEL_DIR` | `trained_bert_model/` |
| `RESULTS_DIR` | `comparison_results/` |

**Data files** expected in `data/`:

- `EmotionWheelFinal (1).csv`: training data (lyrics with all 8 emotion labels) for Naive Bayes and BERT
- `RadaNewAnnotation (1).csv`: 100-song evaluation set (lyrics with 6 emotion labels)
- `NRC-Hashtag-Emotion-Lexicon-v0.2.txt`: lexicon for Naive Bayes

**Emotion labels** (the order is fixed and shared across all modules):

```python
EMOTION_LABELS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
```

**BERT hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max sequence length | 512 |
| Batch size | 16 |
| Epochs | 25 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |

**GPT settings:**

| Parameter | Value |
|---|---|
| Default model | `gpt-5.4` |
| Prompting modes | `zero_shot`, `few_shot` |

To switch GPT models (e.g. to `gpt-4o`), change `OPENAI_MODEL` in `config.py`. The run tag in the output folder name reflects the model used, so results from different models are kept separate automatically.

## 5. Setup and Usage

### 5.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (`OPENAI_API_KEY` is only needed for the GPT comparison):
```
OPENAI_API_KEY=...
GENIUS_TOKEN=...
```

### 5.2 Training (BERT)

```bash
python train_bert.py
```

The dataset is split 70% train / 15% validation / 15% test. The best model checkpoint is saved to `trained_bert_model/` whenever validation macro F1 improves over the previous best. Training metadata written alongside includes timestamp, platform/CUDA info, split sizes, and a SHA256 hash of the dataset file for reproducibility verification.

### 5.3 Evaluation

```bash
python testing.py
```

Outputs written to `comparison_results/<run_tag>/`:

- `rada_all_predictions.csv`
- `metrics_summary_naivebayes.csv`
- `metrics_summary_bert.csv`
- `metrics_summary_openai_zero_shot.csv`
- `metrics_summary_openai_few_shot.csv`
- `methods_comparison_summary.csv`
- `runtime_cost_summary.csv`
- `openai_usage_per_song.csv`
- `run_runtime_metadata.json`
- `openai_parse_failures.jsonl`: raw response preview for any GPT outputs that failed to parse

### 5.4 Visualization

```bash
python visualize_results.py
```

Generates six figures (method comparison bar charts, per-emotion heatmap, confusion matrices, precision-recall scatter, metric distributions) in `comparison_results/figures/`. Also scans all run folders and writes a cross-run aggregate to `comparison_results/cross_run_analysis/`, which is useful for checking result stability across repeated runs.

### 5.5 CLI Demo (Unseen Songs)

```bash
python app.py
```

Fetches lyrics via the Genius API, runs all three methods, and writes results to a timestamped folder in `comparison_results/`.

### 5.6 Web App (Optional)

```bash
python web_app.py
```

Local interface at `http://127.0.0.1:5000`. Supports single lyric input, artist/track lookup, and batch processing. Includes a figure gallery and CSV inspection.

### 5.7 Public Deployment (Docker)

The YAML header at the top of this file runs the Hugging Face Space as a Docker app. The same image runs on any Docker host:

```bash
docker build -t lyrisent .
docker run -d --name lyrisent --restart unless-stopped -p 127.0.0.1:7860:7860 --env-file lyrisent.env lyrisent
```

The image starts `web_app.py` under gunicorn in public mode (`LYRISENT_PUBLIC=1`):

- GPT is only used with an API key the visitor enters; the server never falls back to its own key.
- Saving runs and opening folders are disabled, and server paths are hidden.
- Batch requests are limited to `LYRISENT_PUBLIC_MAX_SONGS` songs (default 5), pasted lyrics to `LYRISENT_PUBLIC_MAX_LYRICS_CHARS` characters (default 20000).

The fine-tuned weights are not part of the repository. Place `model.safetensors` in `trained_bert_model/` before building, or set `LYRISENT_BERT_REPO` to a Hugging Face model repo (plus `HF_TOKEN` if it is private). `GENIUS_TOKEN` enables artist/track lookup.

## 6. How the Project Developed

The project started as an interactive demo (`app.py`) to check that the whole chain worked (API calls, prediction output, basic plumbing). The Naive Bayes baseline and GPT integration were added next, then BERT. In the later stages the pipeline shifted toward reproducibility: training, threshold tuning, and evaluation were separated; a fixed random seed was enforced throughout; and runtime and cost tracking were added.

## 7. Evaluation Design

### Threshold tuning

The 100-song evaluation set is split 50/50 into a tuning subset and an evaluation subset. For each emotion independently, the decision threshold that maximises F1 on the tuning subset is selected via grid search over 19 evenly spaced values between 0.05 and 0.95. Final metrics are then computed on the separate evaluation subset. GPT outputs are already binary and skip this step.

### Preprocessing

`utils.py` applies the same cleaning pipeline to every lyric before any method sees it:

1. Replace `<br>` tags with newlines
2. Strip non-lyric lines: section headers (`[Verse]`, `[Chorus]`, etc.), contributor counts, "Read More", "Translations"
3. Remove punctuation (non-word, non-whitespace characters), preserving newlines
4. Lowercase and collapse whitespace

Because all three methods get the same cleaned text, differences in preprocessing do not affect the comparison.

### Metrics

- F1 (micro, macro, weighted)
- ROC-AUC and PR-AUC (NB and BERT only). GPT produces binary outputs, so these metrics are left blank for GPT.
- Per-label accuracy (averaged across the 6 evaluated emotions)
- Exact match accuracy (all 6 evaluated labels correct, which is naturally low for multi-label tasks)
- Runtime per method, OpenAI token usage, and estimated cost per run

## 8. Results

Averages over repeated runs (100 songs, fixed seed 42). GPT was tested with both `gpt-4o` and `gpt-5.4` in zero-shot and few-shot modes.

| Method | Per-label acc. | F1 (weighted) | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Naive Bayes | 0.623 | 0.560 | 0.535 | 0.460 |
| BERT | 0.657 | 0.608 | 0.683 | 0.584 |
| GPT-4o, zero-shot | 0.753 | 0.716 | n/a | n/a |
| GPT-4o, few-shot | 0.738 | 0.700 | n/a | n/a |
| GPT-5.4, zero-shot | 0.743 | 0.722 | n/a | n/a |
| GPT-5.4, few-shot | 0.768 | 0.750 | n/a | n/a |

Per-label accuracy is the average binary accuracy across the 6 evaluated emotion labels, not strict exact-match (exact match for BERT is ~6%, which is expected given the difficulty of predicting all labels correctly at once).

**Runtimes on 100 songs:** NB ~0.3 s, BERT ~16 s, GPT ~140 s. GPT-5.4 few-shot costs approximately $0.27 per run at current pricing.

Per-label observations for BERT: strongest on Joy and Trust, weakest on Surprise and Fear.

## 9. Limitations

**Metric asymmetry.** NB and BERT produce continuous probability scores; GPT outputs binary labels. ROC-AUC and PR-AUC are not computed for GPT to avoid a false comparison.

**GPT output instability.** Structured output mode achieves 100% parse success on `gpt-4o` and `gpt-5.4`, but earlier model versions were inconsistent even with explicit JSON schemas, requiring fallback parsing and producing missing values in earlier runs.

**Reproducibility.** The local pipeline is fully deterministic. External API behavior (latency, cost, model updates) is not.

**Dataset.** Annotation noise, class imbalance across the eight emotions, and lyric-specific language are inherent to the task. Results reflect these constraints. The evaluation set lacks Trust and Anticipation labels, so performance on those two emotions is not measured.

**Class imbalance handling.** BERT uses per-emotion positive class weights during training, but the train/val/test split is not stratified. For minority emotions this means split composition can vary across runs, which contributes to variance in per-label scores.

**Cost estimation.** The in-pipeline cost estimator only has hardcoded prices for `gpt-4`; all other models (including `gpt-4o` and `gpt-5.4`) default to $0.00 during a run. The cost figures in the results CSVs were recalculated after the fact using `backfill_runtime_cost_usd.py` with current API pricing.

## 10. Future Work

**Higher priority:**
- Add k-fold or repeated-split evaluation (currently a single fixed split)
- Error analysis: which emotion combinations or song types are hardest per method?
- Calibration analysis for BERT

**Lower priority:**
- Unit tests for the parsing and evaluation pipeline
- Probability or confidence outputs from GPT for a fairer AUC comparison

## 11. Reproducibility

```bash
python train_bert.py
python testing.py
python visualize_results.py
```

Requirements: valid `.env` file, dataset files in `data/`, fixed random seed (default: 42).

## 12. Citation

```
Edmonds, Darren, and João Sedoc. 2021. "Multi-Emotion Classification for Song Lyrics."
In Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity,
Sentiment and Social Media Analysis, 221–235. Online: Association for Computational Linguistics.
https://aclanthology.org/2021.wassa-1.24/

Mihalcea, Rada, and Carlo Strapparava. 2012. "Lyrics, Music, and Emotions."
In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing
and Computational Natural Language Learning, 590–599. Association for Computational Linguistics.
```

## 13. Data and Licensing

- Lyric emotion annotations: Edmonds & Sedoc (2021) for training, Mihalcea & Strapparava (2012) for evaluation
- NRC Hashtag Emotion Lexicon
- The data files in `data/` contain song lyrics from the source datasets, and saved unseen-song runs contain short lyric excerpts. Lyrics are copyrighted and remain with their rights holders.
- Lyrics for unseen songs are retrieved via the Genius API at runtime
- OpenAI API usage is subject to OpenAI terms; token counts and costs are logged per run
