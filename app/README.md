# Markdown KB Application

`app/markdown_kb` is the repository's only supported FastAPI runtime. It combines lexical and dense retrieval while preserving a pure-vector inspection baseline through `POST /compare`.

## Install and run

From the repository root:

```bash
cd app/markdown_kb
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Replace this with the same secret used in the repository-root .env.
export INDEX_API_KEY='<same-random-index-admin-secret-as-.env>'
uvicorn app.main:app --reload --port 8026
```

The service reads repository `docs/*.md`. Generated state is split between repository-root `.kb/`, `app/markdown_kb/.kb/`, and repository-root `wiki/` as documented below.

## Configuration

Create a repository-root `.env` with placeholders, not real credentials:

```dotenv
OPENAI_API_KEY=<your-openai-api-key>
INDEX_API_KEY=<random-index-admin-secret>
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_API_KEY` enables OpenAI-compatible embeddings and grounded generation. Dense retrieval remains available without it through the deterministic local TF-IDF/cosine backend.

Missing or unavailable OpenAI configuration does not block startup. `GET /health` reports index availability; `POST /chat` and `POST /compare` expose request-level fallback metadata.

`INDEX_API_KEY` protects local writes:

| Operation | Server key unset/empty | Header missing/wrong | Matching `X-Index-Key` |
|---|---:|---:|---:|
| `POST /index` | `503` | `401` | allowed |
| `POST /chat` with `file_answer=true` | `503` | `401` | allowed |
| `POST /chat` with omitted/`file_answer=false` | public/read-only | public/read-only | public/read-only |

`file_answer=false` is the default. An answer card is written only when `file_answer=true`, authorization succeeds, and grounded output validation succeeds.

The browser UI has a password-style **Index API key** field for **Build / Rebuild index**. Enter the same `INDEX_API_KEY` exported in the server shell; the UI sends it as `X-Index-Key` only for `POST /index`. The value remains in current page memory only and is not written to local storage or logs.

## API

```text
GET  /
GET  /health
POST /index
POST /chat
POST /compare
```

Build/rebuild the index:

```bash
curl -X POST http://localhost:8026/index \
  -H "X-Index-Key: $INDEX_API_KEY"
```

Compare retrieval stages without answer generation:

```bash
curl -X POST http://localhost:8026/compare \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long do refunds take?","k":3}'
```

`POST /compare` returns `bm25`, `vector`, `hybrid`, and `reranked`. `vector` is the **pure-vector baseline** implemented inside this application; it does not represent another service. Internal dense child chunks are aggregated to unique full parent sections before these lists are serialized.

Ask without writing:

```bash
curl -X POST http://localhost:8026/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long do refunds take?","file_answer":false}'
```

Ask and file an authorized answer card:

```bash
curl -X POST http://localhost:8026/chat \
  -H 'Content-Type: application/json' \
  -H "X-Index-Key: $INDEX_API_KEY" \
  -d '{"query":"How long do refunds take?","file_answer":true}'
```

Chat and compare queries are trimmed, must be nonblank, and are limited to 2000 characters.

## Retrieval and fallback behavior

1. Markdown is parsed into stable heading-level parent sections.
2. Dense retrieval uses bounded overlapping children and retains each parent's best child score.
3. BM25 and dense parent ranks are fused with RRF.
4. A deterministic reranker orders the final candidate set.
5. An answerability gate blocks weak context before generation.

When OpenAI document embeddings fail, the app keeps the current BM25/local index and clears stale FAISS runtime state. When an OpenAI query embedding fails, that request uses local TF-IDF/cosine results; it does not fabricate or reuse stale dense scores. `/health` reports process health only; it does not claim remote embedding-provider availability.

The human prompt is strict JSON. Retrieved Markdown is **untrusted context**. A citation allowlist contains only complete selected parent source IDs. LLM errors, malformed/empty answers, or missing/partial/unknown citations return exactly `I cannot confirm from the knowledge base.` with `sources: []` and do not write an answer file.

## Persistence

Generated files are local and disposable; paths are repository-relative:

| Artifact | Purpose |
|---|---|
| `.kb/index.sqlite3`, `.kb/index.json` | Parent sections and incremental FTS5/BM25 state |
| `.kb/vector_index.json` | Deterministic local TF-IDF/cosine baseline |
| `app/markdown_kb/.kb/embedding_cache.sqlite3` | Namespaced incremental child embeddings |
| `.kb/faiss_index/index.faiss` | Raw FAISS `IndexFlatIP` bytes |
| `.kb/faiss_index/children.json` | Ordered safe child metadata |
| `.kb/faiss_index/manifest.json` | Strict version/config/corpus metadata and SHA-256 checksums |
| `wiki/index.md` | Optional filing output |

The FAISS format uses **no pickle**. Before native parsing, the loader rejects unsafe paths/layouts, symlinks, non-regular or oversized files, malformed JSON, and checksum/config/corpus incompatibility. After parsing, it validates flat inner-product type, dimension, and vector count.

Persistence writes a same-parent candidate generation, flushes it, validates it against an immutable corpus snapshot, and performs atomic generation activation where supported. Activation or provider failure preserves the previous valid generation and selects explicit local fallback instead of stale/fabricated vectors.

Old generated formats are not loaded or migrated. Delete obsolete `.kb/` artifacts if needed, then **rebuild** through authorized `POST /index`.

## Tests

```bash
cd app/markdown_kb
env -u OPENAI_API_KEY -u INDEX_API_KEY PYTHONPATH=. .venv/bin/python -m pytest -q
```

The suite uses deterministic fake embedding/chat providers and does not require network access.
