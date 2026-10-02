# Architecture & Roadmap

This is the design reference for the project. Each phase updates the **Status** column of the roadmap.

Labels used throughout:

| Label | Meaning |
|---|---|
| ✅ **Implemented** | Code exists in this repo and its tests pass |
| 🧪 **Implemented, not run** | Code exists but needs hardware or infrastructure this machine doesn't have (GPU, Docker, AWS) |
| 💡 **Recommended** | Advice you should follow, not code |
| ➕ **Optional** | Nice-to-have extension |
| 🔑 **Requires AWS credentials** | Can only be done with your own AWS account |

---

## 1. System architecture

Two separate flows share one package (`src/summarizer`).

### Offline: training pipeline (run on a GPU: Colab, Kaggle or a GPU EC2 instance)

```
 Hugging Face Hub (knkarthick/samsum CSVs)
        │
        ▼
 ┌──────────────────┐   data/raw/{train,validation,test}.csv
 │ 1 DataIngestion  │──► artifacts/reports/ingestion_stats.json
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   data/validated/*.csv  (bad rows removed)
 │ 2 DataValidation │──► artifacts/reports/validation_report.json
 └────────┬─────────┘    (pipeline stops if a blocking check fails)
          ▼
 ┌──────────────────────┐ data/processed/  (tokenized Arrow dataset:
 │ 3 DataTransformation │  input_ids, attention_mask, labels)
 └────────┬─────────────┘
          ▼
 ┌──────────────────┐   artifacts/models/<run_id>/  (model + tokenizer +
 │ 4 ModelTrainer   │   metadata.json: params, git commit, data hash)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐   evaluation_metrics.json, evaluation_samples.csv,
 │ 5 ModelEvaluator │   generation_experiments.csv
 └────────┬─────────┘
          ▼
 artifacts/models/registry.json  ← records every run and marks one "production"
```

### Online: inference service (CPU is fine)

```
 Browser (frontend/ static HTML+JS)
        │  POST /summarize  {"text": "..."}
        ▼
 FastAPI (api/main.py) ── Pydantic validation, error→HTTP mapping, logging
        │
        ▼
 PredictionPipeline ── validate → clean → tokenize → model.generate → post-process
        │                                     │
        │                   model loaded once at start-up from the registry
        ▼                   (or MODEL_NAME_OR_PATH)
 {"summary", "original_length", "summary_length",
  "compression_ratio", "processing_time_seconds"}
```

### Deployment (🔑 needs your AWS account)

```
 GitHub push ─► GitHub Actions: lint → test → docker build ─► Amazon ECR
                                                                │ docker pull
                                                                ▼
 User ─► http://<EC2 public IP>:80 ─► EC2 (Docker) ─► FastAPI + frontend + PEGASUS
                                       │
                                       └─► CloudWatch Logs (container logs, JSON)
 Fine-tuned model weights ─► S3 bucket ─► pulled by the container at start-up
```

---

## 2. Technology stack

Versions are the ones installed and tested in Phase 1 (October 2026). `requirements*.txt` pins them exactly.

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 (3.10+ supported) | ML ecosystem standard |
| Deep learning | PyTorch 2.x (CPU build locally, CUDA on GPU hosts) | What Transformers runs on |
| Models | Hugging Face Transformers 5.x | Pretrained PEGASUS, `Seq2SeqTrainer`, `generate()` |
| Data | `datasets`, pandas | Arrow-backed tokenized datasets and tabular validation |
| Metrics | `evaluate` + `rouge-score`, `sacrebleu` | Standard ROUGE and BLEU implementations |
| Config | YAML (pipeline) + `pydantic-settings` (runtime env vars) | Tune experiments without code changes; one Docker image for every environment |
| API | FastAPI + Uvicorn + Pydantic v2 | Async, auto Swagger docs, typed validation |
| Frontend | Plain HTML/CSS/JS served by FastAPI | No build step and no second container to deploy |
| Testing | pytest, httpx (`TestClient`), pytest-cov | Unit and API tests |
| Linting | Ruff | One fast tool replaces flake8 + isort + pyupgrade |
| Containers | Docker (multi-stage, CPU-only torch) | Reproducible runtime; small image |
| CI/CD | GitHub Actions | Free for public repos; native secrets |
| Cloud | AWS ECR + EC2 (+ S3, CloudWatch) | Cheapest setup that is still realistic (see Phase 6) |

> 💡 **Transformers v5 note.** Many PEGASUS tutorials online use APIs that v5 removed or renamed: `evaluation_strategy=`, `tokenizer.as_target_tokenizer()`, `Trainer(tokenizer=...)`. This project uses the current equivalents: `eval_strategy=`, `tokenizer(text_target=...)` and `processing_class=`.

---

## 3. Folder structure

Changes from the suggested layout, and why:

* **`src/summarizer/` instead of `src/` as a package.** Naming a package `src` collides with other projects and breaks imports once the project is installed. With the src-layout and `pip install -e .`, tests, notebooks and the API all `import summarizer`.
* **`pyproject.toml` instead of `setup.py`.** It is the current standard (PEP 621), and it also holds Ruff and pytest settings.
* **Split requirements.** The API image needs neither `datasets` nor `evaluate`. Keeping them out of the image keeps it smaller.
* **`config/params.yaml` separate from `config/config.yaml`.** Paths stay separate from tunable hyperparameters.
* **`entity/`** holds typed dataclasses for each stage's config, and **`data/validated/`** sits between `raw` and `processed`.

```
ai-content-summarizer/
├── config/
│   ├── config.yaml              # paths, dataset id, registry location        ✅ P1
│   └── params.yaml              # hyperparameters, generation profiles, smoke  ✅ P1
├── src/summarizer/
│   ├── constants.py             # PROJECT_ROOT, config file paths              ✅ P1
│   ├── logger/                  # text/JSON logging, rotating file             ✅ P1
│   ├── exception/               # custom exception hierarchy                   ✅ P1
│   ├── utils/common.py          # yaml/json io, seeding, device selection      ✅ P1
│   ├── utils/text_stats.py      # word counts + length summaries               ✅ P2
│   ├── utils/metrics.py         # ROUGE, BLEU, bootstrap CI, repetition        ✅ P4
│   ├── entity/config_entity.py  # typed per-stage configs with validation      ✅ P1
│   ├── config/
│   │   ├── configuration.py     # ConfigurationManager (YAML → dataclasses)    ✅ P1
│   │   └── settings.py          # AppSettings (env vars → API runtime)         ✅ P1
│   ├── components/
│   │   ├── data_ingestion.py    # DataIngestion                                ✅ P2
│   │   ├── data_validation.py   # DataValidation                               ✅ P2
│   │   ├── text_cleaning.py     # shared cleaning (training == inference)      ✅ P3
│   │   ├── data_transformation.py # DataTransformation                         ✅ P3
│   │   ├── model_trainer.py     # ModelTrainer (smoke-run on CPU only)         🧪 P3
│   │   ├── model_registry.py    # version registry + CLI                       ✅ P3
│   │   ├── summary_generator.py # one generation path for evaluation AND the API ✅ P4
│   │   └── model_evaluation.py  # ModelEvaluator, experiments, auto-promotion   ✅ P4
│   └── pipeline/
│       ├── training_pipeline.py # stages 1–5                                   ✅ P4
│       ├── evaluation_pipeline.py # evaluate any model / run experiments (CLI)  ✅ P4
│       └── prediction_pipeline.py # PredictionPipeline                         ✅ P5
├── api/main.py, api/schemas.py # FastAPI app + Pydantic schemas               ✅ P5
├── frontend/                    # index.html, styles.css, app.js (served /ui/) ✅ P5
├── notebooks/
│   ├── EDA.ipynb                # executed; figures → artifacts/reports/figures ✅ P2
│   └── train_on_colab.ipynb     # GPU training launcher (not executed: no GPU) 🧪 P3
├── tests/                       # pytest suite                                 ✅ P1 (grows each phase)
├── docs/
│   ├── ARCHITECTURE.md          # this file                                    ✅ P1
│   ├── AWS_DEPLOYMENT.md        # step-by-step AWS commands (not executed)     🔑 P6
│   ├── INTERVIEW_PREP.md        # Q&A                                          ✅ P6
│   └── RESUME_BULLETS.md        # bullets + where each number comes from       ✅ P6
├── data/{raw,validated,processed}/   # git-ignored, recreated by `make data`
├── artifacts/{models,reports}/       # models git-ignored; small reports committed
├── Dockerfile, docker-compose.yml, .dockerignore  # hadolint / schema-valid   🧪 P6
├── deploy/                      # nginx, compose override, scripts, IAM JSON   🔑 P6
├── .github/workflows/ci-cd.yml  # actionlint + shellcheck clean               🧪 P6
├── pyproject.toml, requirements*.txt, Makefile                                 ✅ P1
├── .env.example, .gitignore                                                    ✅ P1
└── README.md                    # full recruiter README                        ✅ P6
```

---

## 4. Development roadmap

| Phase | Scope | Status |
|---|---|---|
| **P1 Setup & config** | Package skeleton, YAML configs, ConfigurationManager, AppSettings, logger, exceptions, utils, tooling, tests | ✅ Done |
| **P2 Data** | DataIngestion, DataValidation (+ report), EDA notebook with plots and written insights | ✅ Done (run on real SAMSum) |
| **P3 Transformation & training** | Text cleaning (incl. attachment-tag normalisation found in EDA), tokenization, label prep, ModelTrainer (`Seq2SeqTrainer`, early stopping, checkpoints, resume), model registry, Colab/Kaggle notebook, smoke run on CPU | ✅ Code done; transformation run on real data; 🧪 real PEGASUS fine-tune pending (needs your Colab/Kaggle GPU run) |
| **P4 Evaluation & experiments** | ROUGE-1/2/L/Lsum + 95% CI, BLEU, latency p50/p95, length, compression ratio, repetition; baseline vs optimized; 12-config decoding sweep; auto-promotion | ✅ Code done and smoke-tested; 🧪 real numbers pending (sections 5–8 of the Colab notebook) |
| **P5 Serving** | PredictionPipeline, FastAPI (`/`, `/health`, `/config`, `/summarize`, `/summarize/batch`), frontend, API tests | ✅ Done; served with the tiny model locally (full model needs more RAM than this laptop has free) |
| **P6 Ship** | Dockerfile/compose, GitHub Actions (lint → test → build → ECR → EC2), AWS guide, full README, resume bullets, interview prep | ✅ Written and statically validated; 🧪 Docker/CI run on GitHub; 🔑 AWS needs your account |

Each phase ends with passing tests, and work stops there for your review.

---

## 5. Dataset & model selection

### Dataset: SAMSum
* About 16k messenger-style chat dialogues, each with a human-written summary. Measured in P2 (dataset commit `6b929ff`): 14,731 train / 818 validation / 819 test raw. After validation: **14,635** train (93 leaked rows, 2 junk summaries and 1 exact duplicate removed), with validation and test unchanged.
* **Why it fits.** Dialogue is harder to summarize than news. Information is spread across speakers and the text has slang and turn-taking. The summaries are abstractive: they are written in the third person ("Amanda will bring cookies"), so you can't produce them by copying sentences.
* **Source.** `knkarthick/samsum` (plain CSVs). The original `samsum` Hub entry depended on a loading script and an external archive, and current `datasets` no longer supports either.
* ⚠️ **License: CC BY-NC-ND 4.0 (non-commercial).** That is fine for a portfolio or for research. A commercial product would need different training data. The README will say so; showing you know the license is a plus in interviews.

### Model: PEGASUS
* **Why PEGASUS.** Its pre-training objective was built for summarization. Gap Sentence Generation masks whole *important* sentences and trains the model to regenerate them, which is close to the summarization task itself.
* **Checkpoint: `google/pegasus-xsum`.** It is already fine-tuned on XSum (one-sentence summaries), and SAMSum summaries are similarly short, so you start close to the target. One alternative is `google/pegasus-cnn_dailymail`, which writes longer, more extractive summaries. Another is `google/pegasus-large`, which is pre-trained only and so needs more fine-tuning. Changing the checkpoint is a one-line edit in `params.yaml`, so P4 can compare them.
* **Size and hardware.** About 570M parameters, roughly 2.3 GB of weights in fp32.
  * Fine-tuning needs a GPU. This laptop has no GPU and about 3 GB of free RAM, so training runs on **free Colab/Kaggle T4 GPUs**. Memory is kept down with Adafactor, gradient checkpointing, batch size 2 and 8 accumulation steps.
  * fp16 is avoided because it overflows to NaN on PEGASUS. bf16 is used where the GPU supports it.
* **Inference** works on CPU, but plan for at least 4 GB of RAM in the container. Real latency will be measured in P4, not guessed.
* **Smoke mode.** `SMOKE_TEST=1` swaps in `hf-internal-testing/tiny-random-PegasusForConditionalGeneration`, which is about 1 MB, plus 32 rows. The whole pipeline then runs on this laptop and in CI within minutes. Its outputs are random text, so smoke runs prove only that the code works, never quality.
* 💡 **Is PEGASUS outdated?** It dates from 2020. For a production product today you would also look at instruction-tuned models: FLAN-T5, BART-large, or a hosted LLM API. PEGASUS is still a strong, well-understood choice for showing that you can *fine-tune and serve your own* seq2seq model. ➕ P4 can add a FLAN-T5-base comparison through config alone.

---

## 6. What makes this stronger than a tutorial

1. **Config-driven and typed.** No hyperparameters in code. Bad config fails at start-up with a clear message, not two hours into training.
2. **Data validation gate with a report.** The pipeline refuses to train on broken data, and the report shows what was dropped and why.
3. **Same cleaning for training and inference.** One shared function prevents train/serve skew, a classic production bug.
4. **Smoke-test mode.** The full pipeline runs in CI on every push with a 1 MB model, so integration bugs are caught without a GPU.
5. **Model versioning.** Each run gets its own folder with params, data hash and git commit. A registry marks the production model, so you can roll back by editing one line.
6. **Generation experiments.** Baseline vs optimized decoding and a parameter sweep, reported in a table of *measured* numbers.
7. **Operational metrics.** Latency and compression ratio are reported alongside ROUGE. Real systems care about cost and speed, not only quality.
8. **Production API.** Typed request and response schemas, input limits, error-to-HTTP-status mapping, health checks that report model state, and structured JSON logs for CloudWatch.
9. **CI/CD to real infrastructure.** Every step needing secrets is gated on GitHub Secrets, and nothing is faked.
10. **Honest reporting.** Every number in the README comes from a file in `artifacts/reports/`.

---

## 7. Phase 3 notes

* **The PEGASUS vocabulary has no `<` character.** `<file_photo>` would tokenize as `<unk> file _ photo >`, and `<3` as `<unk> 3`. `TextCleaner` maps the ~40 tag spellings to `[photo] [gif] [video] [audio] [link] [file]`, and `<3` to `(heart)`. Newlines are dropped by the tokenizer, but each turn keeps its `Name:` prefix.
* **Transformers v5 API changes.** This code uses `eval_strategy`, `warmup_steps` (a float < 1 is a ratio; `warmup_ratio` was removed), `tokenizer(text_target=...)` and `Seq2SeqTrainer(processing_class=...)`.
* **Smoke isolation.** With `SMOKE_TEST=1`, processed data, models, checkpoints, reports and the registry go under `artifacts/smoke/`, so a smoke run can never overwrite real results or be promoted to production by accident.
* **Safety checks before GPU time is spent.** The trainer refuses processed data built by a different tokenizer or with different length settings. It also refuses `max_input_length` beyond the model's position embeddings, and `bf16` on GPUs without native support.
* **Reproducibility.** Each run records the git commit (with a `-dirty` flag), a data fingerprint, all hyperparameters, library versions and the GPU name, in `metadata.json` and in the registry.
* **Not yet verified:** a real `google/pegasus-xsum` fine-tune. This machine has no GPU and about 3 GB of free RAM, while the model needs about 2.3 GB for weights alone. The real hyperparameters *are* checked by a test that builds `Seq2SeqTrainingArguments` from them, but GPU memory fit and training time on a T4 are unknown until you run the Colab notebook.

## 8. Phase 4 notes

* **Why no real numbers yet.** Loading `google/pegasus-xsum` on this laptop was killed by the Linux OOM killer at
  3.7 GB of RAM (its legacy `pytorch_model.bin` is read fully into memory). The CPU (i5-1035G1) has no native bf16,
  so a half-precision workaround would be emulated and very slow. All real evaluation therefore runs in the Colab
  notebook: zero-shot baseline → fine-tune → evaluation → experiments → CPU latency.
* **Same rows, every time.** Evaluation uses a seeded sample of the test split. Its id-signature is stored with each
  result, and auto-promotion only compares models evaluated on an identical signature and profile. Runs with a custom
  `-n` are never written to the registry.
* **Confidence intervals.** With 100–200 test dialogues, ROUGE differences under about 1 point are often noise. Every
  result carries a 95% bootstrap CI; treat overlapping intervals as "no clear difference".
* **Latency is device-specific.** GPU latency in Colab says nothing about a CPU-only EC2 server, so the notebook also
  runs `--device cpu -n 20`. Results are saved per device (`evaluations/<model>__<profile>__<device>.json`).
* **Single generation path.** `SummaryGenerator` is shared with the Phase 5 API, so evaluated latency and quality are
  what the service delivers.

## 9. Phase 5 notes

* **One process, one generation at a time.** The pipeline serialises `generate()` with a lock. PyTorch already uses
  every CPU core for a single request, and parallel calls would multiply memory use. Throughput scales by running more
  containers behind a load balancer, not more threads.
* **Endpoints are sync `def`.** FastAPI runs them in a thread pool, so a slow generation never blocks `/health`.
* **Health semantics.** If the model fails to load, the server still starts and `/health` returns 503 with the error.
  The container is then marked unhealthy by Docker/ECS, instead of crash-looping without logs.
* **Errors.** Everything returns `{error, detail, request_id}`. The request id is also in the `X-Request-ID` header and
  in the logs, so one failed request can be traced.
* **Not yet:** authentication and rate limiting. A public demo should sit behind a reverse proxy with rate limits (see
  the Phase 6 deployment guide).

## 10. What remains for you (needs resources this machine doesn't have)

1. **Push to GitHub.** CI runs lint, the 167 tests, the smoke pipeline and the Docker build + container test.
2. **Colab/Kaggle GPU run** (`notebooks/train_on_colab.ipynb`): zero-shot baseline, fine-tune, evaluation,
   experiments, CPU latency. Commit the downloaded `artifacts/reports/`, then fill in the README results and the
   resume placeholders.
3. **Publish the model** on the Hugging Face Hub, then **deploy** with `docs/AWS_DEPLOYMENT.md` (optional but
   recommended), and add screenshots.
