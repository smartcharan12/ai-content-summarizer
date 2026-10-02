# Resume bullets

Every number below is either **measured in this repo** (with its source) or a **[PLACEHOLDER]** you must fill in from
your own runs. Never submit a placeholder, and never round a number up.

## Version A: with placeholders (use after the GPU run)

**AI-Powered Content Summarization | Python, NLP, Transformers, FastAPI, Docker, AWS**

- Fine-tuned Google PEGASUS (570M params) on 14.6k SAMSum chat dialogues with a config-driven 5-stage PyTorch/Hugging
  Face pipeline, improving ROUGE-L from **[BASELINE_ROUGE_L]** (zero-shot) to **[FINETUNED_ROUGE_L]** on a held-out
  test sample (95% CI reported); detected and removed 93 train/test-leaking dialogues during data validation.
- Ran **12** controlled decoding experiments (beam size, length penalty, n-gram blocking); the selected beam-search
  profile improved ROUGE-L by **[DELTA]** points over greedy decoding at **[LATENCY_P50]** s median CPU latency.
- Served the model through a FastAPI REST API and web UI (input validation, batch endpoint, health checks) in a
  non-root multi-stage Docker image, with a GitHub Actions pipeline (167 tests, lint, container test) that pushes to
  AWS ECR and deploys to EC2 via OIDC and SSM.

## Version B: no model metrics (valid today, before the GPU run)

**AI-Powered Content Summarization | Python, NLP, Transformers, FastAPI, Docker, AWS**

- Built a config-driven 5-stage NLP pipeline (ingestion, validation, tokenization, PEGASUS fine-tuning, evaluation)
  on 16.4k SAMSum dialogues; validation caught 93 train/test-leaking rows, and EDA-driven truncation settings covered
  99.3% of inputs within 512 tokens.
- Implemented a model registry with metric-gated promotion (ROUGE-1/2/L with bootstrap confidence intervals,
  latency, compression ratio) and a 12-configuration decoding-parameter experiment framework.
- Shipped a FastAPI summarization service and web UI in a multi-stage Docker image, backed by 167 automated tests
  and a GitHub Actions CI/CD pipeline that builds, container-tests and deploys to AWS (ECR + EC2) without stored
  credentials.

> ⚠️ Version B's last bullet says "deploys to AWS". Only keep that wording once you have actually completed the
> deployment in docs/AWS_DEPLOYMENT.md. Before that, write "**designed** a … pipeline for AWS (ECR + EC2)".

## Where each number comes from

| Number | Source | Status |
|---|---|---|
| 14.6k train dialogues (14,635) / 16.4k total (16,368 raw) | `artifacts/reports/validation_report.json`, `ingestion_stats.json` | ✅ measured |
| 93 leaked rows removed | `validation_report.json` → `cross_split_leakage` | ✅ measured |
| 99.3% of dialogues ≤ 512 tokens | `artifacts/reports/transformation_report.json` (train: 94 of 14,635 truncated) | ✅ measured |
| 167 tests | `pytest` output | ✅ measured (update if it changes) |
| 12 decoding configurations | `config/params.yaml` → `experiments.runs` | ✅ by design |
| `[BASELINE_ROUGE_L]` | `artifacts/reports/evaluations/google_pegasus-xsum__optimized__cuda.json` → `metrics.rougeL` | ⏳ Colab §5 |
| `[FINETUNED_ROUGE_L]` | `artifacts/reports/evaluations/<run_id>__optimized__cuda.json` → `metrics.rougeL` | ⏳ Colab §6 |
| `[DELTA]` | `generation_experiments.csv`: `optimized` minus `baseline (greedy)` ROUGE-L | ⏳ Colab §8 |
| `[LATENCY_P50]` | `evaluations/<run_id>__optimized__cpu__n20.json` → `latency.p50_seconds` | ⏳ Colab §8 (CPU) |

**Before claiming an improvement:** check that the two 95% confidence intervals don't heavily overlap. If they do,
say "comparable to" instead of "improved".
