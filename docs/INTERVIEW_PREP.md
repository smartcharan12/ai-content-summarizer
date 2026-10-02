# Interview Preparation: AI-Powered Content Summarization

Answers are written the way you would say them: short first, details if asked. Where an answer depends on your
measured results, it says **[from your reports]**. Fill those in from `artifacts/reports/`, never from memory.

---

## 1. NLP fundamentals

**Q: What is abstractive summarization?**
Generating a *new* text that conveys the main points, possibly with words and sentences that never appear in the
source. The model writes the summary token by token, conditioned on the input. In SAMSum, "Amanda: I'll bring you
tomorrow :-)" becomes "Amanda will bring Jerry cookies tomorrow": a change of person, a resolved reference, and
paraphrasing.

**Q: Extractive vs abstractive?**
Extractive selects and concatenates existing sentences or spans (TextRank, BERTSum). It is faithful and fluent per
sentence, but can't compress, rephrase or merge information. Abstractive generates new text, so it is more concise
and natural, but it can *hallucinate* (state things the source doesn't say). For chat dialogue, extractive fails: in
my EDA, 78% of reference-summary bigrams don't occur in the dialogue at all, so copying sentences has a low ceiling.

**Q: Why PEGASUS?**
Its pre-training objective was designed for summarization. **Gap Sentence Generation (GSG)** removes whole
sentences that look important (chosen by ROUGE overlap with the rest of the document) and trains the model to
regenerate them from the remaining text. That is very close to summarization itself, so it fine-tunes well even
with little data. I used `google/pegasus-xsum`, which was additionally fine-tuned for one-sentence "extreme"
summaries, close to SAMSum's ~20-word summaries. (The paper also tried masked-language modelling alongside GSG; the
authors found it didn't help at the large scale.)

**Q: How does the Transformer work?**
An **encoder** reads the whole input in parallel. Each layer has self-attention, so every token looks at every other
token, then a feed-forward network, with residual connections and layer norm around both. A **decoder** generates
the output one token at a time. Each decoder layer has *masked* self-attention over the tokens generated so far,
*cross-attention* over the encoder's outputs, and a feed-forward network. Positional information is added
(PEGASUS uses sinusoidal position embeddings), because attention itself has no notion of order. A final linear
layer plus softmax gives a probability distribution over the vocabulary for the next token.

**Q: What is attention?**
A learned, weighted lookup. Each position produces a **query**, and every position produces a **key** and a
**value**. Scores are `softmax(Q·Kᵀ / √d_k)`. Each output is the weighted sum of the values. Dividing by √d_k keeps
the dot products from growing with dimension and saturating the softmax. **Multi-head** attention runs several of
these in parallel on lower-dimensional projections, so different heads can capture different relations (who said
what, coreference, negation).

**Q: What is self-attention?**
Attention where queries, keys and values all come from the *same* sequence: each word builds its representation from
the other words of its own text. In the decoder it is *masked* (causal), so position *t* can't see positions after
*t*. Cross-attention is the other kind: decoder queries attend to encoder keys and values. That is how the summary
"looks at" the dialogue.

**Q: What is tokenization?**
Turning text into integer ids from a fixed vocabulary. PEGASUS uses a **SentencePiece** sub-word tokenizer
(~96k entries), so rare words, names and emoticons are split into pieces rather than mapped to "unknown". Dialogue
averaged about 1.4 tokens per word in my data. Tokenization also bit me: the vocabulary has **no `<` character**, so
SAMSum's `<file_photo>` tags became `<unk> file _ photo >`. I map them to `[photo]`-style tags in the shared cleaner.

**Q: What is fine-tuning?**
Continuing to train a pre-trained model on a smaller, task-specific dataset. All weights start from the pre-trained
values and are updated with a small learning rate. The model keeps its general language knowledge and adapts to the
new domain (chat language, third-person summaries, speaker names). It needs far less data and compute than training
from scratch.

---

## 2. Model & evaluation

**Q: Why SAMSum?**
About 16k messenger conversations with human-written abstractive summaries. Dialogue is harder than news: information
is spread across speakers, the language is informal (38% of dialogues have emoticons), and the summary must say
*who* did *what* (97% of references name a speaker). It is a standard benchmark, so results are comparable. Its
license is **CC BY-NC-ND 4.0** (non-commercial), which is fine for research or a portfolio, not for a commercial
product.

**Q: Why ROUGE?**
It is the standard automatic summarization metric: cheap, reproducible, and comparable with published work. It
measures n-gram overlap with a human reference. Its limits: it rewards word overlap, not correctness, so a fluent
summary with the wrong speaker can still score well, and a valid paraphrase can score badly. That is why I also save
per-row samples and read the worst ones, and why I'd add human or factuality evaluation for production.

**Q: ROUGE-1? ROUGE-2? ROUGE-L?**
* **ROUGE-1**: overlap of single words (content coverage).
* **ROUGE-2**: overlap of consecutive word pairs (phrasing). It is always much lower on SAMSum, because references
  are paraphrased.
* **ROUGE-L**: based on the **longest common subsequence**: words in the same order, gaps allowed. It rewards
  sentence structure without requiring contiguous matches.
* **ROUGE-Lsum**: LCS computed per sentence and combined; what most papers report.

Each has precision (share of generated n-grams found in the reference), recall (share of reference n-grams found)
and F1. I report F1 × 100 with stemming.

**Q: What is beam search?**
A decoding strategy that keeps the `num_beams` most probable partial sequences at every step, instead of only the
single best (greedy). It finds higher-probability summaries, avoiding greedy's early mistakes, at roughly
`num_beams`× the compute. My experiment table compares greedy and 2, 4 and 6 beams on the same test rows
**[from your reports]**.

**Q: What is length penalty?**
In Hugging Face beam search, a finished hypothesis is scored as `sum(log-probs) / length^length_penalty`. Log-probs
are negative and get more negative with every token, so dividing by a larger power of the length favours longer
outputs. `length_penalty > 1` therefore favours longer summaries and `< 1` shorter ones. It has no effect with
greedy decoding.

**Q: How do you prevent repetitive summaries?**
`no_repeat_ngram_size=3` forbids generating any 3-gram twice; `repetition_penalty` down-weights tokens already
produced; beam search with `early_stopping` ends cleanly. I measure the **repeated-trigram rate** of outputs, so the
experiment table shows the effect of n-gram blocking directly. Fine-tuning itself also reduces repetition.

**Q: What other generation parameters did you test?**
`min_length` (forces a minimum length, useful against overly short summaries), `max_length` (a hard cap, matched to
the 64-token training targets), and `early_stopping`. Each experiment changes **one** parameter relative to the
optimized profile, on the same seeded 100 test rows, so differences are attributable **[table from your reports]**.

---

## 3. ML engineering

**Q: Why separate ingestion / validation / transformation?**
Single responsibility and debuggability. Each stage has one job, its own config, typed inputs and outputs, files on
disk, and its own exception type. I can re-run only validation after changing a rule. A failure says exactly where
it happened. Each stage is unit-testable in isolation. Validation is a *gate*: it writes its report and stops the
pipeline if, for example, more than 5% of rows would be dropped, so a silently changed upstream format can't reach
training.

**Q: What did validation actually find?**
No nulls or duplicate ids, but three real issues. **Train/test leakage**: 42 test and 47 validation dialogues also
appeared in train; I removed 93 rows from *train*, otherwise ROUGE would be inflated. Two **junk references**
("test", "yyy"). And about 460 dialogues that look duplicated but have *different* summaries: multiple human
annotations, which I kept on purpose.

**Q: Why OOP?**
Each stage is a class that receives a config dataclass and exposes `run()`, which makes them interchangeable for the
pipeline runner and easy to test with injected fakes. My tests pass a fake data loader, a fake model generator and a
fake tokenizer. Shared behaviour (a single `TextCleaner` and `SummaryGenerator`) is reused by training, evaluation
and the API, which prevents train/serve skew.

**Q: How do you handle model versioning?**
Every training run gets a run id and its own folder with the weights, tokenizer and `metadata.json`: hyperparameters,
**git commit** (with a dirty flag), **data fingerprint**, library versions and GPU. A JSON **registry** records every
run with a stage: `candidate` → `production` → `archived`. Evaluation promotes a candidate only if its ROUGE-L beats
production **on the identical test rows** (it checks a signature of the row ids). Rollback means promoting the
previous run. Docker images are tagged with the git SHA, so code and model versions are traceable. At larger scale I
would use the MLflow Model Registry; the concepts are the same.

**Q: How would you reduce inference latency?**
In order of effort:
1. Decoding settings: fewer beams, a shorter `max_length` (my experiment table quantifies the quality cost).
2. Batching concurrent requests.
3. Dynamic int8 quantization or ONNX Runtime on CPU.
4. A distilled or smaller model.
5. bf16 on a GPU.
6. Caching repeated inputs.

Always re-measure ROUGE after each change. Speed gains that lose quality need a business decision.

**Q: How would you deploy at scale?**
Stateless containers behind a load balancer with autoscaling on CPU or request count (ECS or an EC2 Auto Scaling
group). Model weights come from S3 or the Hub into a cache. A queue (SQS) handles batch or long jobs. Monitoring
covers latency percentiles, error rate, input-length drift, the share of truncated inputs, and output length. A
canary or blue-green rollout ensures a new model receives a small share of traffic first.

**Q: Why was fp16 avoided?**
PEGASUS is known to overflow in fp16 (NaN loss); its activations exceed fp16's range. bf16 has fp32's exponent range,
so I use bf16 on GPUs that support it natively (Ampere and newer). On a T4 the trainer falls back to fp32 and uses
Adafactor plus gradient checkpointing to fit 16 GB.

---

## 4. FastAPI

**Q: Why FastAPI?**
Type hints drive Pydantic request validation and automatic OpenAPI/Swagger docs. It is fast (ASGI), has dependency
injection and lifespan hooks for loading the model once at start-up, and its TestClient makes API tests easy.

**Q: GET vs POST?**
GET reads a resource: no request body by convention, parameters in the URL, safe and idempotent, cacheable. POST
sends data in the body to be processed. `/summarize` is POST because the input can be 20,000 characters (too long and
too private for a URL) and it triggers computation. `/health` is GET.

**Q: What is Pydantic?**
A library that validates and converts data using Python type hints. `SummarizeRequest(text: str, min_length=1)`
means a missing field, a number instead of a string, or an empty string is rejected with HTTP 422 before my code
runs. The same models generate the OpenAPI schema.

**Q: How is input validated?**
Two layers. Pydantic handles *structure* (types, required fields, non-empty lists). The pipeline handles
*deployment-specific limits* from environment variables: empty after stripping, fewer than `MIN_INPUT_WORDS`, more
than `MAX_INPUT_CHARS`, or more than `MAX_BATCH_SIZE` texts. Batch errors name the offending index. Long-but-allowed
inputs are summarised from their first 512 tokens and flagged `truncated: true`.

**Q: Error handling?**
Custom exceptions map to status codes: `InputValidationError` → 422, `ModelNotReadyError` → 503,
`PredictionError` → 500, anything else → a generic 500. Every error has the same JSON shape with a `request_id` that
also appears in logs and the `X-Request-ID` header. Internal details are logged but never returned. If the model
fails to load, the server still starts and `/health` returns 503 with the reason, so the orchestrator marks it
unhealthy instead of the container crash-looping silently.

---

## 5. Docker

**Q: Why Docker?**
It packages the code, the exact Python and library versions, and OS libraries into one artifact that runs the same
on my laptop, in CI and on EC2. No more "works on my machine". For ML this matters even more: torch, CUDA and
tokenizer versions must match exactly.

**Q: Image vs container?**
An **image** is the immutable, layered template (the build output, stored in a registry such as ECR). A
**container** is a running instance of an image, with its own process, writable layer and network. One image → many
containers.

**Q: How did you keep the image small and safe?**
* **Multi-stage build:** compilers and pip caches stay in the builder stage.
* **CPU-only PyTorch:** about 200 MB instead of about 2 GB for the CUDA build.
* **Runtime dependencies only:** no pandas or training libraries.
* **Model not baked in:** it's downloaded to a cached volume. Otherwise every push and pull would carry 2.3 GB.
* **`.dockerignore`** keeps data, models and `.env` out of the build.
* **Non-root user (UID 1001)** and a health check.
* **One uvicorn worker per container,** because each worker would load its own model copy.

---

## 6. AWS

**Q: Why EC2?**
For a single CPU model service, EC2 + Docker is the cheapest always-on option and the easiest to reason about. The
model needs about 3 GB of RAM, which suits a t3.large. Fargate or App Runner would cost more for that memory, and
SageMaker is the most expensive option for a demo. I documented the trade-offs and the path to ECS or an Auto
Scaling group.

**Q: What is ECR?**
Elastic Container Registry: AWS's private Docker registry. CI pushes images there, and the EC2 instance pulls with
its IAM role, so no registry passwords are stored.

**Q: How does the image reach AWS?**
1. A push to `main` triggers GitHub Actions: lint → tests → build the image → run it and call the API.
2. Actions assumes an AWS role through **OIDC** (short-lived credentials, restricted to my repo's main branch) and
   pushes `ecr/ai-content-summarizer:<git-sha>`.
3. It runs `deploy.sh` on the instance via **SSM Run Command**, with no SSH key and port 22 closed.
4. The script pulls the image, restarts the containers with Docker Compose, and waits until `/health` returns 200.

**Q: How would you scale it?**
Vertically first: a compute-optimised instance, then quantization or ONNX. Horizontally: an AMI or launch template
with this Compose stack in an Auto Scaling group behind an ALB (health check on `/health`), scaling on CPU. Or ECS
with a capacity provider. The service is stateless, so it scales out cleanly.

---

## 7. Hard project-specific questions

**Q: Your ROUGE-L is [X]. Is that good?**
Compare it to two things. First, my own zero-shot baseline: the same model *before* fine-tuning, on the same rows
**[from your reports]**. Second, published SAMSum results; check the paper and model cards and cite exact numbers.
Mention evaluation differences: I score 200 seeded test rows (not all 819), use `rouge_score` with stemming, and
split sentences for Lsum with a simple rule. Small differences are within noise, which is why I report 95% bootstrap
confidence intervals.

**Q: How do you know the "optimized" decoding is actually better?**
Same model, same rows, one change at a time, with a bootstrap 95% CI on ROUGE-L. If the intervals overlap heavily, I
don't claim an improvement. I also check average length and repetition rate, because ROUGE often rises simply when
outputs get longer.

**Q: How did you make sure the test set wasn't leaking into training?**
Validation checks exact dialogue overlap across splits. 93 train rows matched validation or test dialogues and were
removed from train. Test stays untouched, so it remains comparable with other work. Near-duplicates (same chat,
slightly edited) would need fuzzy matching such as MinHash; that's a possible extension.

**Q: Why not just call an LLM API?**
For a business it's a real option and I'd benchmark it. A fine-tuned 570M model is cheaper per request at volume, has
predictable latency, can run on-premise (privacy for chat data), and is fully under my control (no API changes). An
LLM is likely better zero-shot on unusual inputs. The right answer is to compare quality, cost per 1k summaries,
latency and privacy on the same test set.

**Q: The model says Bob will bring cookies, but it was Amanda. How would you catch that?**
ROUGE won't reliably catch it. Options: NLI- or QA-based factual-consistency checks (for example, does the dialogue
entail each summary sentence), a speaker-attribution check (do named entities and their actions match the turns), and
human review of a sample. Reading the lowest-scoring evaluation rows is part of my evaluation workflow.

**Q: What happens with input that isn't dialogue (a news article)?**
That's domain shift: the model was fine-tuned on chats, so outputs may still be acceptable (PEGASUS-xsum saw news
during pre-training) but aren't evaluated. In production I'd track input characteristics (e.g. share of `Name:`
lines, input length) to detect drift, and evaluate on a sample of real traffic.

**Q: Why 512 input tokens?**
From the EDA: 99.3% of training dialogues fit within 512 PEGASUS tokens. Going to 1024 would quadruple attention
compute for 0.65% of rows. Longer inputs are truncated and flagged in the API response. For long documents I'd chunk
and summarise hierarchically.

**Q: What was the hardest problem?**
Pick a real one:
* **The tokenizer silently mangling attachment tags**, because `<` isn't in the vocabulary. I found it by inspecting
  tokens, fixed it in a shared cleaner, and wrote tests for 16 tag variants.
* **Transformers v5 removed APIs** that most tutorials use (`evaluation_strategy`, `warmup_ratio`,
  `as_target_tokenizer`). I checked the installed library's signatures and added a test that builds the real
  training arguments from config.
* **Memory:** loading the 2.3 GB model was killed by the Linux OOM killer on my 8 GB laptop. That's why training
  and evaluation run on a cloud GPU, and why the API caps the container at 4 GB and runs one worker.

**Q: Two users hit /summarize at the same time. What happens?**
FastAPI runs each request in a worker thread. The pipeline serialises `generate()` with a lock, because one
generation already uses all CPU cores and parallel runs would multiply memory. The second request waits. For more
throughput: more containers behind a load balancer, or dynamic batching.

**Q: How would you monitor this in production?**
* **Service:** latency p50/p95, request rate, 4xx/5xx rates (from the JSON logs in CloudWatch), memory.
* **Model:** input-length distribution, share of truncated inputs, output length, compression ratio, empty or
  repetitive outputs, and user feedback (thumbs up/down).
* **Alerts:** a failing `/health`, rising p95 latency or error rate.
* **Periodically:** re-evaluate on fresh labelled samples.

**Q: Is the API secure?**
Partially, and I'd say so. It has input size limits, rate limiting in nginx (10 requests/min/IP), a non-root
container, no SSH, least-privilege IAM, OIDC instead of stored keys, and no internal errors leaked. Missing:
authentication (API keys or OAuth) and HTTPS (Caddy or an ALB with ACM). Both are documented next steps.

**Q: What would you do differently or next?**
Evaluate on the full test set with human evaluation of faithfulness. Compare against FLAN-T5 and BART. Quantize for
CPU serving. Add authentication and HTTPS. Add drift monitoring. Use MLflow for experiment tracking once there are
many runs.
