from collections import Counter
from pathlib import Path

import pytest

from app import embedding_cache, filing, indexer, retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DOCS_DIR = PROJECT_ROOT / "docs"


@pytest.fixture(autouse=True)
def isolated_index_state(monkeypatch, tmp_path):
    # The application loads a repository .env at import time. Tests must not
    # inherit real credentials and accidentally select live OpenAI providers.
    for environment_name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    index_path = tmp_path / ".kb" / "index.json"
    index_db_path = tmp_path / ".kb" / "index.sqlite3"
    embedding_cache_path = tmp_path / ".kb" / embedding_cache.CACHE_FILENAME
    answers_dir = tmp_path / ".kb" / "answers"
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(indexer, "INDEX_PATH", index_path)
    monkeypatch.setattr(indexer, "INDEX_DB_PATH", index_db_path, raising=False)
    monkeypatch.setattr(embedding_cache, "CACHE_PATH", embedding_cache_path)
    monkeypatch.setattr(filing, "ANSWERS_DIR", answers_dir)
    monkeypatch.setattr(filing, "WIKI_DIR", wiki_dir)
    indexer.sections = []
    indexer.child_chunks = []
    indexer._child_chunking_policy = indexer.DEFAULT_CHUNKING_POLICY
    indexer.doc_freq = Counter()
    indexer.avg_doc_len = 0.0
    indexer.files_indexed = 0
    indexer.last_index_stats = {}
    indexer.section_vector_index = None
    indexer.faiss_section_index = None
    indexer.embedding_provider_override = None
    indexer._embedding_provider_cache = None
    indexer.last_faiss_status = {}
    indexer._faiss_fallback_generation = None
    retrieval._llm = None
    yield index_path
    indexer.sections = []
    indexer.child_chunks = []
    indexer._child_chunking_policy = indexer.DEFAULT_CHUNKING_POLICY
    indexer.doc_freq = Counter()
    indexer.avg_doc_len = 0.0
    indexer.files_indexed = 0
    indexer.last_index_stats = {}
    indexer.section_vector_index = None
    indexer.faiss_section_index = None
    indexer.embedding_provider_override = None
    indexer._embedding_provider_cache = None
    indexer.last_faiss_status = {}
    indexer._faiss_fallback_generation = None
    retrieval._llm = None


@pytest.fixture
def index_path(isolated_index_state):
    return isolated_index_state


@pytest.fixture
def embedding_cache_path(isolated_index_state):
    return embedding_cache.CACHE_PATH


@pytest.fixture
def sample_docs_dir():
    return SAMPLE_DOCS_DIR
