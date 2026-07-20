const form = document.querySelector("#question-form");
const indexForm = document.querySelector("#index-form");
const input = document.querySelector("#question-input");
const indexKeyInput = document.querySelector("#index-key-input");
const askButton = document.querySelector("#ask-button");
const compareButton = document.querySelector("#compare-button");
const clearButton = document.querySelector("#clear-button");
const indexButton = document.querySelector("#index-button");
const healthStatus = document.querySelector("#health-status");
const result = document.querySelector("#result");
const resultState = document.querySelector("#result-state");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const comparison = document.querySelector("#comparison");
const comparisonState = document.querySelector("#comparison-state");
const comparisonGrid = document.querySelector("#comparison-grid");

function setBusy(isBusy, label = "Ask + Compare") {
  askButton.disabled = isBusy;
  compareButton.disabled = isBusy;
  indexButton.disabled = isBusy;
  indexKeyInput.disabled = isBusy;
  askButton.textContent = isBusy ? label : "Ask + Compare";
}

function showResult(state) {
  result.classList.remove("hidden");
  resultState.textContent = state;
}

function showComparison(state) {
  comparison.classList.remove("hidden");
  comparisonState.textContent = state;
}

function clearResult() {
  answer.textContent = "";
  sources.replaceChildren();
  comparisonGrid.replaceChildren();
  result.classList.add("hidden");
  comparison.classList.add("hidden");
}

function formatScore(value) {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return number >= 1 ? number.toFixed(3) : number.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function renderSources(items) {
  sources.replaceChildren();

  if (!items || items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "source-meta";
    empty.textContent = "No sources returned.";
    sources.append(empty);
    return;
  }

  const label = document.createElement("p");
  label.className = "label";
  label.textContent = "Sources";
  sources.append(label);

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "source-card";

    const title = document.createElement("p");
    title.className = "source-title";
    title.textContent = item.source;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    meta.textContent = `${item.heading || "Untitled"} · score ${formatScore(item.score)}`;

    const content = document.createElement("p");
    content.className = "source-content";
    content.textContent = item.content || "";

    card.append(title, meta, content);
    sources.append(card);
  }
}

function renderComparisonColumn(title, subtitle, items) {
  const column = document.createElement("article");
  column.className = "comparison-column";

  const heading = document.createElement("div");
  heading.className = "comparison-column-heading";

  const name = document.createElement("h3");
  name.textContent = title;

  const detail = document.createElement("p");
  detail.textContent = subtitle;

  heading.append(name, detail);
  column.append(heading);

  if (!items || items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "source-meta";
    empty.textContent = "No retrieval hits.";
    column.append(empty);
    return column;
  }

  items.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "comparison-card";

    const rank = document.createElement("p");
    rank.className = "comparison-rank";
    rank.textContent = `#${index + 1}`;

    const source = document.createElement("p");
    source.className = "source-title";
    source.textContent = item.source;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    const shown = item.score !== undefined ? `score ${formatScore(item.score)}` : null;
    const bm25 = item.bm25_score !== undefined ? `BM25 ${formatScore(item.bm25_score)}` : null;
    const vector = item.vector_score !== undefined ? `Vector ${formatScore(item.vector_score)}` : null;
    const rrf = item.rrf_score !== undefined ? `RRF ${formatScore(item.rrf_score)}` : null;
    const rerank = item.rerank_score !== undefined ? `Rerank ${formatScore(item.rerank_score)}` : null;
    const rerankRank = item.rerank_rank !== undefined && item.rerank_rank !== null ? `Rerank #${item.rerank_rank}` : null;
    meta.textContent = [item.heading || "Untitled", shown, bm25, vector, rrf, rerank, rerankRank].filter(Boolean).join(" · ");

    const content = document.createElement("p");
    content.className = "source-content";
    content.textContent = item.content || "";

    card.append(rank, source, meta, content);
    column.append(card);
  });

  return column;
}

function renderComparison(payload) {
  showComparison(payload.indexed ? "Compared" : "Not indexed");
  comparisonGrid.replaceChildren();

  if (!payload.indexed) {
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = payload.message || "The knowledge base has not been indexed yet.";
    comparisonGrid.append(message);
    return;
  }

  comparisonGrid.append(
    renderComparisonColumn(
      "Markdown KB",
      "SQLite FTS5/BM25 keyword ranking over Markdown heading sections.",
      payload.bm25,
    ),
    renderComparisonColumn(
      "Pure-vector baseline",
      "OpenAI embeddings + FAISS semantic ranking, with local vector fallback.",
      payload.vector,
    ),
    renderComparisonColumn(
      "Hybrid RRF",
      "Reciprocal Rank Fusion of Markdown KB and vector rankings.",
      payload.hybrid,
    ),
    renderComparisonColumn(
      "Reranked",
      "Final text-aware rerank used by /chat.",
      payload.reranked,
    ),
  );
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    healthStatus.textContent = payload.status === "ok" ? "Online" : JSON.stringify(payload);
    healthStatus.className = "status-text ok";
  } catch (error) {
    healthStatus.textContent = `Offline: ${error.message}`;
    healthStatus.className = "status-text error";
  }
}

async function buildIndex() {
  setBusy(true, "Indexing…");
  showResult("Indexing");
  answer.className = "answer";
  answer.textContent = "Building index from docs/*.md…";
  sources.replaceChildren();

  try {
    const response = await fetch("/index", {
      method: "POST",
      headers: { "X-Index-Key": indexKeyInput.value },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    const payload = await response.json();
    resultState.textContent = "Indexed";
    answer.textContent = `Indexed ${payload.files_indexed} files and ${payload.sections_indexed} sections. Changed ${payload.changed_files}, skipped ${payload.skipped_files}, deleted ${payload.deleted_files}.`;
  } catch (error) {
    resultState.textContent = "Error";
    answer.className = "answer error";
    answer.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function compareRetrieval(query, { manageBusy = true } = {}) {
  if (manageBusy) setBusy(true, "Comparing…");
  showComparison("Comparing");
  comparisonGrid.replaceChildren();

  try {
    const response = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 3 }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    const payload = await response.json();
    renderComparison(payload);
    return payload;
  } catch (error) {
    comparisonState.textContent = "Error";
    comparisonGrid.replaceChildren();
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = error.message;
    comparisonGrid.append(message);
    throw error;
  } finally {
    if (manageBusy) setBusy(false);
  }
}

async function askQuestion(query) {
  setBusy(true, "Asking…");
  showResult("Thinking");
  showComparison("Waiting");
  answer.className = "answer";
  answer.textContent = "";
  sources.replaceChildren();
  comparisonGrid.replaceChildren();

  try {
    const chatResponse = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!chatResponse.ok) throw new Error(`HTTP ${chatResponse.status}: ${await chatResponse.text()}`);
    const payload = await chatResponse.json();
    resultState.textContent = "Answered";
    answer.textContent = payload.answer;
    renderSources(payload.sources);
    try {
      await compareRetrieval(query, { manageBusy: false });
    } catch (_error) {
      // The comparison panel already renders its own error; keep the chat answer visible.
    }
  } catch (error) {
    resultState.textContent = "Error";
    answer.className = "answer error";
    answer.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) askQuestion(query);
});

clearButton.addEventListener("click", () => {
  input.value = "";
  input.focus();
  clearResult();
});

indexForm.addEventListener("submit", (event) => {
  event.preventDefault();
  buildIndex();
});

compareButton.addEventListener("click", () => {
  const query = input.value.trim();
  if (query) compareRetrieval(query);
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    input.focus();
  });
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

checkHealth();
