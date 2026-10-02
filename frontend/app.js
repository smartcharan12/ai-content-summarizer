// Frontend for the AI Content Summarizer. Plain JavaScript, no build step.
// It talks to the same FastAPI server that serves this page (same origin).
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  input: $("input"), counter: $("counter"), summarize: $("summarize-btn"),
  clear: $("clear-btn"), example: $("example-btn"), copy: $("copy-btn"),
  summary: $("summary"), error: $("error"), truncated: $("truncated"),
  stats: $("stats"), status: $("status"), statusText: $("status-text"),
  modelInfo: $("model-info"), bar: $("bar"), barFill: $("bar-fill"),
  barCaption: $("bar-caption"), toast: $("toast"),
};

// Defaults until /config answers; the server is the source of truth.
const limits = { maxChars: 20000, minWords: 5 };
let modelReady = false;
let busy = false;

const EXAMPLE = [
  "Hannah: Hey, do you have Betty's number?",
  "Amanda: Lemme check",
  "Amanda: Sorry, can't find it.",
  "Amanda: Ask Larry",
  "Amanda: He called her last time we were at the park together",
  "Hannah: I don't know him well",
  "Amanda: Don't be shy, he's very nice",
  "Hannah: If you say so..",
  "Hannah: I'd rather you texted him",
  "Amanda: Just text him 🙂",
  "Hannah: Urgh.. Alright",
  "Hannah: Bye",
  "Amanda: Bye bye",
].join("\n");

const fmt = new Intl.NumberFormat();
const countWords = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0);

// ---------------------------------------------------------------- input state
function updateCounter() {
  const text = els.input.value;
  const chars = text.trim().length;
  const words = countWords(text);
  els.counter.textContent =
    `${fmt.format(chars)} / ${fmt.format(limits.maxChars)} characters · ${fmt.format(words)} words`;
  els.counter.classList.toggle("over", chars > limits.maxChars);
  els.summarize.disabled = busy || !modelReady || chars === 0 || chars > limits.maxChars;
}

function setBusy(on) {
  busy = on;
  els.summarize.classList.toggle("loading", on);
  els.summarize.querySelector(".btn-label").textContent = on ? "Summarizing…" : "Summarize";
  els.input.readOnly = on;
  updateCounter();
}

// ---------------------------------------------------------------- output state
function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function resetOutput() {
  els.error.hidden = true;
  els.truncated.hidden = true;
  els.stats.hidden = true;
  els.bar.hidden = true;
  els.barCaption.hidden = true;
  els.copy.disabled = true;
  els.summary.className = "summary empty";
  els.summary.textContent = "Your summary will appear here.";
}

function renderResult(r) {
  els.summary.className = "summary";
  els.summary.textContent = r.summary || "(The model returned an empty summary.)";
  $("orig-chars").textContent = fmt.format(r.original_length);
  $("orig-words").textContent = fmt.format(r.original_word_count);
  $("sum-chars").textContent = fmt.format(r.summary_length);
  $("sum-words").textContent = fmt.format(r.summary_word_count);
  $("ratio").textContent = r.compression_ratio.toFixed(1);
  $("time").textContent = r.processing_time_seconds.toFixed(2);
  els.stats.hidden = false;

  const share = Math.min(100, (100 * r.summary_length) / Math.max(r.original_length, 1));
  els.bar.hidden = false;
  els.barCaption.hidden = false;
  els.barFill.style.width = "0";
  requestAnimationFrame(() => { els.barFill.style.width = `${Math.max(share, 1)}%`; });
  els.barCaption.textContent = `Summary is ${share.toFixed(1)}% of the original length`;

  els.truncated.hidden = !r.truncated;
  els.copy.disabled = !r.summary;
}

// ---------------------------------------------------------------- API calls
async function summarize() {
  const text = els.input.value.trim();
  if (!text || busy) return;
  if (countWords(text) < limits.minWords) {
    showError(`Please enter at least ${limits.minWords} words.`);
    return;
  }
  resetOutput();
  els.summary.className = "summary loading";
  els.summary.textContent = "Reading the conversation…";
  setBusy(true);
  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail
        : Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ")
        : `Request failed (HTTP ${response.status})`;
      throw new Error(detail);
    }
    renderResult(data);
  } catch (err) {
    resetOutput();
    showError(err.message === "Failed to fetch" ? "Cannot reach the server." : err.message);
  } finally {
    setBusy(false);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    modelReady = response.ok && data.model_loaded;
    els.status.className = `status ${modelReady ? "ok" : "down"}`;
    els.statusText.textContent = modelReady
      ? `Model ready · ${data.device}` : "Model not loaded";
    els.modelInfo.textContent =
      `Model: ${data.model ?? "–"} · profile: ${data.generation_profile ?? "–"} · v${data.version}`;
    if (modelReady) await loadConfig();
  } catch {
    modelReady = false;
    els.status.className = "status down";
    els.statusText.textContent = "Server unreachable";
  }
  updateCounter();
  if (!modelReady) setTimeout(checkHealth, 5000); // keep polling while the model loads
}

async function loadConfig() {
  try {
    const response = await fetch("/config");
    if (!response.ok) return;
    const cfg = await response.json();
    limits.maxChars = cfg.max_input_chars ?? limits.maxChars;
    limits.minWords = cfg.min_input_words ?? limits.minWords;
  } catch { /* keep defaults */ }
}

// ---------------------------------------------------------------- helpers
let toastTimer;
function toast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 1800);
}

async function copySummary() {
  const text = els.summary.textContent;
  try {
    await navigator.clipboard.writeText(text);
    toast("Summary copied");
  } catch {
    // Clipboard API needs HTTPS or localhost; fall back to selecting the text.
    const range = document.createRange();
    range.selectNodeContents(els.summary);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    toast("Press Ctrl+C to copy");
  }
}

// ---------------------------------------------------------------- wiring
els.input.addEventListener("input", updateCounter);
els.input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") summarize();
});
els.summarize.addEventListener("click", summarize);
els.clear.addEventListener("click", () => {
  els.input.value = "";
  resetOutput();
  updateCounter();
  els.input.focus();
});
els.example.addEventListener("click", () => {
  els.input.value = EXAMPLE;
  updateCounter();
  els.input.focus();
});
els.copy.addEventListener("click", copySummary);

updateCounter();
checkHealth();
