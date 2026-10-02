# AI-Powered Content Summarization

[![CI/CD](https://github.com/<your-username>/ai-content-summarizer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/<your-username>/ai-content-summarizer/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.18-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed)

An end-to-end, production-style NLP system that turns long conversations into short **abstractive summaries**. It
fine-tunes Google's **PEGASUS** on the **SAMSum** dialogue dataset and serves it through a **FastAPI** REST API and
web UI. It is packaged with **Docker**, tested and shipped by **GitHub Actions**, and designed for deployment on
**AWS** (ECR + EC2).

```text
Hannah: Hey, do you have Betty's number?          ┐
Amanda: Lemme check                               │
Amanda: Sorry, can't find it.                     │   POST /summarize
Amanda: Ask Larry                                 ├─────────────────►  "<model summary — add a real
Amanda: He called her last time we were at ...    │                     example after training>"
Hannah: I'd rather you texted him                 │
Amanda: Just text him 🙂                          ┘
```

---

## Problem Statement

Teams, support desks and messaging apps produce huge volumes of chat. Reading every thread to find *who decided
what* is slow. Generic extractive summarizers fail on dialogue: the information is spread across speakers, the
language is informal, and a good summary must be **rewritten** in the third person ("Amanda will text Larry").
In this dataset, **78% of the word pairs in human summaries never appear in the conversation** (EDA §9), so copying
sentences cannot work.

## Solution

A sequence-to-sequence Transformer (PEGASUS, ~570M parameters), fine-tuned on 14.6k human-summarised conversations.
It is wrapped in a reproducible ML pipeline and a production-style inference service:

* a **5-stage training pipeline** (ingestion → validation → transformation → training → evaluation) driven by YAML config,
* a **model registry** with automatic, evidence-based promotion,
* a **REST API + web UI** with input validation, consistent error handling and structured logs,
* **CI/CD** that lints, tests, runs the whole pipeline on a tiny model, builds and tests the Docker image, and
  (once AWS is configured) pushes to ECR and deploys to EC2.

## Key Features

| | |
|---|---|
| 🧹 **Data quality gate** | Validation found and removed **93 train rows that leaked into validation/test** and 2 junk references, and kept ~460 dialogues with multiple human summaries on purpose. Report: `artifacts/reports/validation_report.json` |
| 🔬 **Evidence-based config** | 512-token input limit chosen because 99.3% of dialogues fit (EDA). Attachment tags rewritten because PEGASUS's vocabulary has no `<` character |
| 🔁 **No train/serve skew** | One `TextCleaner` and one `SummaryGenerator` are shared by training, evaluation and the API |
| ⚙️ **Config-driven, typed** | Every hyperparameter and decoding setting lives in `config/params.yaml`; bad values fail at start-up |
| 🧪 **Smoke mode** | `SMOKE_TEST=1` runs the full pipeline with a 1.6M-parameter model in ~20 s, on a laptop or in CI |
| 📦 **Model versioning** | Each run records git commit, data fingerprint, hyperparameters and library versions; the registry promotes a model only if it beats production on identical test rows |
| 📊 **Honest evaluation** | ROUGE-1/2/L/Lsum with 95% bootstrap CIs, BLEU, latency p50/p95, compression ratio, repetition rate; a 12-configuration decoding experiment |
| 🚀 **Production API** | Pydantic validation, 422/503/500 error mapping, request IDs, JSON logs, health/readiness, rate limiting via nginx |
| 🐳 **Lean, secure image** | Multi-stage build, CPU-only PyTorch, non-root user, health check, no model baked in |
| 🔐 **Keyless CI/CD** | GitHub Actions → AWS via OIDC (no stored AWS keys), deploy via SSM (no SSH) |

## Architecture

```
                 ┌─────────────────────────── Training (GPU: Colab / Kaggle) ──────────────────────────┐
 Hugging Face ──►│ 1 Ingestion → 2 Validation → 3 Transformation → 4 Training → 5 Evaluation          │
 (SAMSum CSVs)   │   raw CSVs      report+gate    clean+tokenize    Seq2SeqTrainer   ROUGE, CI, latency  │
                 └────────────────────────────────────────────────────┬───────────────────────────────┘
                                                                      ▼
                                                artifacts/models/<run_id> + registry.json (production)
                                                                      │
 Browser (/ui) ──► nginx (rate limit) ──► FastAPI ──► PredictionPipeline ──► SummaryGenerator (PEGASUS)
                                          │  validate → clean → tokenize → generate → post-process
                                          └► {summary, lengths, compression_ratio, processing_time, …}

 GitHub push ──► Actions: lint → tests → smoke pipeline → docker build+test ──► ECR ──► EC2 (SSM deploy)
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Deep learning / NLP | PyTorch 2.14, Hugging Face Transformers 5.18, Datasets, SentencePiece |
| Evaluation | rouge-score, SacreBLEU, NumPy bootstrap |
| Data / EDA | pandas, Matplotlib, Jupyter |
| Serving | FastAPI, Uvicorn, Pydantic v2, pydantic-settings; HTML/CSS/JS frontend |
| MLOps | YAML config, file-based model registry, structured logging, pytest (167 tests), Ruff |
| Infrastructure | Docker (multi-stage), Docker Compose, nginx, GitHub Actions, AWS ECR / EC2 / SSM / IAM-OIDC / CloudWatch / S3 |

## Dataset

**[SAMSum](https://huggingface.co/datasets/knkarthick/samsum)**: ~16k messenger-style conversations, each with a
human-written abstractive summary (official split 14,731 / 818 / 819).

| Finding (from [notebooks/EDA.ipynb](notebooks/EDA.ipynb)) | Consequence |
|---|---|
| 93 train dialogues also appear in validation/test | Removed from train (prevents inflated scores) |
| Median dialogue 73 words / 106 tokens; 99.3% ≤ 512 tokens | `max_input_length: 512` |
| Median summary 18 words; 98.8% ≤ 64 tokens | `max_target_length: 64` |
| Summaries ~4× shorter than dialogues (median 3.9×) | Compression ratio reported by the API |
| 97% of summaries name a speaker | No lowercasing; speaker prefixes kept |
| 34% of summary words and 78% of bigrams are not in the dialogue | Abstractive model required |

⚠️ SAMSum is licensed **CC BY-NC-ND 4.0** (non-commercial). This project is for research and portfolio use.

## Model

* **Base:** [`google/pegasus-xsum`](https://huggingface.co/google/pegasus-xsum). PEGASUS was pre-trained with *gap
  sentence generation* (regenerate removed important sentences), which is close to summarization itself. The XSum
  checkpoint already writes one-sentence summaries, matching SAMSum's short targets.
* **Fine-tuning:** `Seq2SeqTrainer`, 2 epochs, LR 5e-5, effective batch 16 (2 × 8 gradient accumulation),
  Adafactor, gradient checkpointing, early stopping on validation loss. **No fp16** (PEGASUS overflows to NaN):
  bf16 on Ampere+ GPUs, fp32 otherwise.
* **Decoding (`optimized` profile):** beam search (4 beams), length penalty 0.8, no repeated trigrams,
  10–64 tokens. It is compared against greedy decoding in the experiments below.

## ML Pipeline

```bash
python -m summarizer.pipeline.training_pipeline                       # all 5 stages
python -m summarizer.pipeline.training_pipeline --stages ingestion validation
python -m summarizer.pipeline.training_pipeline --stages training --run-id <id> --resume
SMOKE_TEST=1 python -m summarizer.pipeline.training_pipeline          # tiny model, ~20 s
```

| Stage | Class | Output |
|---|---|---|
| 1 Ingestion | `DataIngestion` | `data/raw/*.csv`, `ingestion_stats.json` (row counts, SHA-256, dataset commit) |
| 2 Validation | `DataValidation` | `data/validated/*.csv`, `validation_report.json`; **stops the pipeline** on schema errors or a >5% drop rate |
| 3 Transformation | `DataTransformation` | Tokenized Arrow dataset + `transformation_meta.json` (the trainer refuses mismatched tokenizers) |
| 4 Training | `ModelTrainer` | `artifacts/models/<run_id>/` + registry entry (candidate) |
| 5 Evaluation | `ModelEvaluator` | Metrics, per-row samples, history per model/device; promotes if better than production |

## API Documentation

Interactive docs at **`/docs`** (Swagger) and **`/redoc`**.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info and links |
| `/health` | GET | 200 if the model is loaded, 503 otherwise |
| `/config` | GET | Input limits and generation settings |
| `/summarize` | POST | Summarise one text |
| `/summarize/batch` | POST | Summarise up to `MAX_BATCH_SIZE` texts |
| `/ui/` | GET | Web interface |

```bash
curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" \
  -d '{"text": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I will bring you some tomorrow :-)"}'
```

```json
{
  "summary": "…",
  "original_length": 101,
  "summary_length": 52,
  "compression_ratio": 1.94,
  "processing_time_seconds": 1.12,
  "original_word_count": 18,
  "summary_word_count": 9,
  "input_tokens": 31,
  "truncated": false,
  "model_version": "20261005-101500 (v1)"
}
```
*(The values above illustrate the response shape; they are not a measured result.)*

Errors always look like `{"error": "invalid_input", "detail": "Text is too short to summarise: 2 words (minimum 5)", "request_id": "…"}`.
Status codes: **422** for invalid input, **503** when the model isn't loaded, **500** when generation fails.

## Installation

**Linux / macOS**
```bash
git clone https://github.com/<your-username>/ai-content-summarizer.git && cd ai-content-summarizer
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/<your-username>/ai-content-summarizer.git; cd ai-content-summarizer
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .
copy .env.example .env
```

## Running Locally

```bash
make test          # 167 tests (pytest -m "not slow" skips the tiny-model downloads)
make data          # download + validate SAMSum
make smoke         # whole training pipeline with a tiny model (proves the code path)
make api           # API + UI on http://localhost:8000  (needs ~3 GB free RAM for PEGASUS)
make api-demo      # same UI/API with a 1.6M-param random model (any laptop; gibberish output)
```
Windows: run the command each Make target wraps, e.g. `python -m uvicorn api.main:app --port 8000` (see `Makefile`).

## Docker

```bash
docker build -t ai-content-summarizer .
docker run -p 8000:8000 ai-content-summarizer                       # serves google/pegasus-xsum by default
docker run -p 8000:8000 -e MODEL_SOURCE=path \
  -e MODEL_NAME_OR_PATH=<your-hf-username>/pegasus-samsum ai-content-summarizer
docker compose up --build                                           # + persistent model cache volume
docker compose --profile proxy up -d                                # + nginx on :80 with rate limiting
```
The first start downloads the model (~2.3 GB) into the cache volume. The health check reports healthy once it's
loaded.

## Training

Fine-tuning needs a GPU. [notebooks/train_on_colab.ipynb](notebooks/train_on_colab.ipynb) runs on a free Colab or
Kaggle T4. It clones the repo, runs the pipeline with checkpoints on Google Drive (resumable), evaluates the
zero-shot baseline and the fine-tuned model, runs the decoding experiments, measures CPU latency, and packages the
reports.

```bash
make models                                                   # list model versions
python -m summarizer.components.model_registry promote <id>   # manual promotion / rollback
```

## Evaluation

```bash
python -m summarizer.pipeline.evaluation_pipeline --model google/pegasus-xsum --no-promote   # zero-shot baseline
python -m summarizer.pipeline.evaluation_pipeline --model <run_id>                           # fine-tuned
python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --experiments             # decoding sweep
python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --device cpu -n 20        # CPU latency
```

| Metric | Meaning |
|---|---|
| ROUGE-1 / ROUGE-2 | Overlap of words / word pairs with the human summary (F1) |
| ROUGE-L / ROUGE-Lsum | Longest common subsequence (word order), whole text / per sentence |
| BLEU | Precision-oriented n-gram overlap (secondary) |
| 95% CI | Bootstrap interval on the mean; overlapping intervals = no clear winner |
| Latency p50 / p95 | Seconds per single request |
| Compression ratio | Input length ÷ summary length |
| Repeated-trigram rate | Share of summaries repeating a 3-word phrase |

## Results

> ⚠️ **Pending GPU run. No numbers have been measured yet, and none are invented here.** Fill these tables from
> `artifacts/reports/evaluations/*.json` and `artifacts/reports/generation_experiments.md` after running the Colab
> notebook (sections 5–8).

| Model (200 seeded test dialogues) | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BLEU | CPU p50 latency |
|---|---|---|---|---|---|---|
| `google/pegasus-xsum`, zero-shot | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | — |
| Fine-tuned on SAMSum, greedy decoding | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Fine-tuned on SAMSum, `optimized` decoding | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Decoding experiments:** paste `artifacts/reports/generation_experiments.md` here.

**Examples (original → generated → reference):** pick 3 rows from `artifacts/reports/evaluation_samples.csv`,
including one failure, and explain it.

## AWS Deployment

Single EC2 instance (t3.large) running Docker Compose: nginx (rate limiting) → FastAPI/PEGASUS. Images in ECR,
logs in CloudWatch. GitHub Actions deploys via OIDC + SSM, with no stored AWS keys and no SSH.
Step-by-step commands, IAM policies, cost notes, alternatives (ECS, App Runner, SageMaker, Lambda), scaling and
teardown are in **[docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md)**.

> 🔑 Not deployed yet: deployment requires your AWS account. The Dockerfile, Compose files, workflow, scripts and
> IAM policies are statically validated (hadolint, actionlint, shellcheck, Compose-spec schema).

## Screenshots

> Add after serving the fine-tuned model: `docs/images/ui.png` (web UI with a real summary),
> `docs/images/swagger.png` (`/docs`), `docs/images/actions.png` (green CI run).

## Project Structure

```
ai-content-summarizer/
├── api/                     FastAPI app (main.py) + Pydantic schemas
├── config/                  config.yaml (paths, sources) + params.yaml (hyperparameters, decoding, experiments)
├── deploy/                  nginx.conf, EC2 compose override, deploy.sh, ec2-user-data.sh, iam/*.json
├── docs/                    ARCHITECTURE, AWS_DEPLOYMENT, INTERVIEW_PREP, RESUME_BULLETS
├── frontend/                index.html, styles.css, app.js (served at /ui/)
├── notebooks/               EDA.ipynb (executed), train_on_colab.ipynb (GPU launcher)
├── src/summarizer/
│   ├── components/          data_ingestion, data_validation, text_cleaning, data_transformation,
│   │                        model_trainer, model_registry, summary_generator, model_evaluation
│   ├── pipeline/            training_pipeline, evaluation_pipeline, prediction_pipeline
│   ├── config/              ConfigurationManager (YAML) + AppSettings (env vars)
│   ├── entity/              typed per-stage config dataclasses
│   ├── utils/               io helpers, text stats, metrics
│   ├── logger/              text/JSON logging
│   └── exception/           custom exception hierarchy
├── tests/                   167 pytest tests (unit, API, tiny-model integration)
├── artifacts/reports/       committed reports and EDA figures (models are git-ignored)
├── Dockerfile, .dockerignore, docker-compose.yml
├── .github/workflows/ci-cd.yml
├── pyproject.toml, requirements.txt, requirements-train.txt, requirements-dev.txt, Makefile
└── .env.example
```

## Business Use Cases

* **Customer support:** summarise long chat or ticket threads for hand-offs and QA review.
* **Team messaging:** "catch me up" digests of Slack/Teams channels.
* **CRM / sales:** turn call transcripts or chat logs into account notes.
* **Healthcare / legal intake:** first-draft summaries of patient or client conversations (with human review and
  appropriately licensed training data).
* **Moderation and analytics:** compress conversations before topic or sentiment analysis.

## Limitations

* Trained on short, informal English chats; long documents (>512 tokens) are truncated, and other domains are not
  evaluated.
* Abstractive models can **hallucinate** or attribute actions to the wrong speaker; ROUGE does not catch this
  reliably.
* CPU inference is slow relative to GPU (measure it: `--device cpu`); one generation runs at a time per container.
* No authentication or HTTPS out of the box (nginx rate limiting only).
* SAMSum's non-commercial license prevents commercial use of models trained on it.

## Future Improvements

* Human and factual-consistency evaluation (NLI/QA-based) alongside ROUGE; full 819-row test set.
* Compare with FLAN-T5 and BART-large; parameter-efficient fine-tuning (LoRA).
* int8 quantization / ONNX Runtime for faster CPU inference; dynamic batching.
* Long-document support via chunking and hierarchical summarization.
* API keys, HTTPS (Caddy or ALB + ACM), drift monitoring dashboards, MLflow tracking.

## Author

**Charan Teja**: [LinkedIn](https://www.linkedin.com/in/<your-profile>) · [GitHub](https://github.com/<your-username>) · [Email](mailto:<your-email>)

Interview notes for this project: [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md)
