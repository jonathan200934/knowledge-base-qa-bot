# Knowledge Base QA — Hybrid Retrieval Demo

A small, inspectable RAG service for answering questions over Markdown documentation. `app/markdown_kb` is the repository's **only supported application/runtime**.

It exposes the same query through BM25, pure-vector retrieval, hybrid Reciprocal Rank Fusion (RRF), and deterministic reranking so ranking changes remain visible before answer generation.

## What it demonstrates

- Heading-aware Markdown indexing with stable `file.md#heading` parent citations
- Bounded overlapping child chunks for dense retrieval, aggregated back to one full parent section before fusion and serialization
- Incremental SQLite FTS5/BM25 indexing
- OpenAI-compatible document/query embeddings with a validated incremental SQLite cache and FAISS `IndexFlatIP`
- Deterministic local TF-IDF/cosine fallback when embedding configuration, document embedding, query embedding, or FAISS activation fails
- Side-by-side retrieval inspection through `POST /compare`
- Structured untrusted context, a retrieved-source citation allowlist, and fail-closed answer generation

## Retrieval pipeline

```text
docs/*.md
   → heading-level parent sections
   ├─→ SQLite FTS5 ─────────────────────────→ BM25 parents ─────┐
   └─→ bounded child chunks → embeddings     → dense children    │
                              ├─ FAISS        → best child/parent ├─→ RRF
                              └─ local TF-IDF → best child/parent │
                                                                └─→ reranker
                                                                    → answerability
                                                                    → grounded answer
```

Dense child matches are aggregated by stable parent source ID before RRF. API responses therefore expose readable parent sections, not internal child IDs.

`POST /compare` returns `bm25`, `vector`, `hybrid`, and `reranked`. The `vector` list is the retained **pure-vector baseline**; it is not a second application.

## Safety contract

### Protected writes

Set `INDEX_API_KEY` to enable operations that write local state. Supply it as `X-Index-Key`.

- `POST /index` is always protected.
- `POST /chat` defaults to `file_answer=false` and is public/read-only.
- `POST /chat` with request-body `file_answer=true` writes an answer card only after the same key check succeeds.
- Unset or empty server `INDEX_API_KEY` returns `503` for a protected write.
- Missing or wrong `X-Index-Key` returns `401`.
- A matching `X-Index-Key` allows the operation.

Keys are compared in constant time and are never included in responses or logs.

### Query and grounding rules

`POST /chat` and `POST /compare` trim input, reject blank queries, and accept at most **2000 characters**. Retrieved Markdown is serialized as strict JSON and treated as **untrusted context**, not instructions.

Generated answers must use complete citations from the selected parent-section citation allowlist. LLM errors, empty or malformed output, and absent, partial, or unknown citations fail closed to exactly:

```text
I cannot confirm from the knowledge base.
```

with `sources: []`; no answer file is written.

### Local artifacts and fallback

Generated local state uses these explicit repository-relative paths:

- `.kb/index.sqlite3` and `.kb/index.json`: canonical parent-section index
- `.kb/vector_index.json`: deterministic local TF-IDF/cosine pure-vector index
- `app/markdown_kb/.kb/embedding_cache.sqlite3`: namespaced incremental document-embedding cache
- `.kb/faiss_index/index.faiss`: raw FAISS bytes
- `.kb/faiss_index/children.json`: ordered, bounded child metadata
- `.kb/faiss_index/manifest.json`: strict version/config/corpus manifest with SHA-256 checksums
- `wiki/index.md`: optional filing output

FAISS persistence uses **no pickle**. Loading rejects symlinks, non-regular or oversized files, unsafe layouts, malformed JSON, checksum/config/corpus mismatches, and invalid native index type/metric/dimension/count before activation. A candidate generation is flushed, validated, and bound to an immutable corpus snapshot before atomic generation activation. Failed or unsupported activation leaves the prior generation intact and places the runtime in an explicit local-vector fallback state.

Legacy or incompatible FAISS/cache artifacts are never migrated by unsafe deserialization. Remove obsolete generated artifacts if necessary and **rebuild** with an authorized `POST /index`.

## Setup

```bash
cd app/markdown_kb
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repository root using placeholders, never committed credentials:

```dotenv
OPENAI_API_KEY=<your-openai-api-key>
INDEX_API_KEY=<random-index-admin-secret>

# Optional
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=https://api.openai.com/v1
```

Start the service:

```bash
# Replace the placeholder with the same secret used in the repository-root .env.
# export keeps it available to both Uvicorn and the curl commands below.
export INDEX_API_KEY='<same-random-index-admin-secret-as-.env>'
uvicorn app.main:app --reload --port 8026
```

Open <http://localhost:8026> or <http://localhost:8026/docs>.

In the browser UI, enter that same secret in the password-style **Index API key** field before selecting **Build / Rebuild index**. The page sends it only in the `X-Index-Key` header for that request. It remains in current page memory only; the UI does not put it in local storage or logs.

## API examples

```bash
curl http://localhost:8026/health

curl -X POST http://localhost:8026/index \
  -H "X-Index-Key: $INDEX_API_KEY"

curl -X POST http://localhost:8026/compare \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long do refunds take?","k":3}'

curl -X POST http://localhost:8026/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"How long do refunds take?","file_answer":false}'

curl -X POST http://localhost:8026/chat \
  -H 'Content-Type: application/json' \
  -H "X-Index-Key: $INDEX_API_KEY" \
  -d '{"query":"How long do refunds take?","file_answer":true}'
```

## Tests

Normal tests use deterministic fake providers and require no network or credentials:

```bash
cd app/markdown_kb
env -u OPENAI_API_KEY -u INDEX_API_KEY PYTHONPATH=. .venv/bin/python -m pytest -q
```
