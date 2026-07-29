import copy
import inspect
import json
import math
import sqlite3
import struct
from dataclasses import replace
from typing import cast

import pytest

from app import embedding_cache, faiss_index, indexer
from app.chunking import ChildChunk, ChunkingPolicy, embedded_content_hash


def _namespace(**overrides):
    values = {
        "provider_kind": "openai-compatible",
        "embedding_model": "text-embedding-3-small",
        "base_url": "https://api.example.test/v1",
        "requested_dimension": 1_536,
        "resolved_dimension": 1_536,
        "vector_normalization": "l2",
        "metric": "inner-product",
        "distance_strategy": "cosine",
        "chunking_policy": ChunkingPolicy(
            chunk_size=1_000,
            chunk_overlap=150,
            separators=("\n\n", "\n", ". ", " ", ""),
            splitter_version="recursive-character-v1",
            heading_prefix_version="heading-path-v1",
        ),
        "heading_prefix_behavior": "heading-path-then-heading",
    }
    values.update(overrides)
    return embedding_cache.build_namespace(**values)


def _vector_row(namespace, chunk_id="doc.md#answer::chunk-0"):
    return (
        namespace.fingerprint,
        chunk_id,
        "doc.md#answer",
        "a" * 64,
        2,
        embedding_cache.VECTOR_ENCODING,
        struct.pack("<2f", 0.25, 0.75),
    )


def test_default_cache_is_isolated_sqlite_and_stores_only_sanitized_namespace(
    embedding_cache_path, tmp_path
):
    cache_path = embedding_cache_path
    secret_url = (
        "HTTPS://cache-user:super-secret@API.Example.Test:443/v1/"
        "?api_key=query-secret&user=private-user#ignored"
    )
    namespace = _namespace(base_url=secret_url)

    initialized_path = embedding_cache.initialize_cache(namespace=namespace)

    assert initialized_path == cache_path
    assert initialized_path.parent == tmp_path / ".kb"
    assert initialized_path.read_bytes().startswith(b"SQLite format 3\x00")
    assert not list(tmp_path.rglob("*.pkl"))
    assert not list(tmp_path.rglob("*.pickle"))

    with sqlite3.connect(cache_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
        stored_fingerprint, stored_json = conn.execute(
            "SELECT fingerprint, namespace_json FROM embedding_namespaces"
        ).fetchone()

    assert tables == {"embedding_namespaces", "embedding_vectors"}
    assert stored_fingerprint == namespace.fingerprint
    assert json.loads(stored_json) == namespace.to_dict()
    assert namespace.base_url == "https://api.example.test/v1"
    for sensitive_value in (
        "cache-user",
        "super-secret",
        "api_key",
        "query-secret",
        "private-user",
    ):
        assert sensitive_value not in stored_json
        assert sensitive_value.encode() not in cache_path.read_bytes()


def test_namespace_is_canonical_and_equivalent_base_urls_match():
    variants = (
        " HTTPS://API.Example.Test:443/v1/ ",
        "https://api.example.test/v1",
        "https://api.example.test:443//v1/./",
    )
    namespaces = [_namespace(base_url=value) for value in variants]

    assert {namespace.base_url for namespace in namespaces} == {
        "https://api.example.test/v1"
    }
    assert len({namespace.canonical_json for namespace in namespaces}) == 1
    assert len({namespace.fingerprint for namespace in namespaces}) == 1
    assert namespaces[0].canonical_json == json.dumps(
        namespaces[0].to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(namespaces[0].fingerprint) == 64
    assert set(namespaces[0].fingerprint) <= set("0123456789abcdef")


def test_every_vector_and_child_policy_field_changes_the_namespace():
    baseline = _namespace()
    policy = ChunkingPolicy(
        chunk_size=1_000,
        chunk_overlap=150,
        separators=("\n\n", "\n", ". ", " ", ""),
        splitter_version="recursive-character-v1",
        heading_prefix_version="heading-path-v1",
    )
    changed_policies = {
        "chunk_size": replace(policy, chunk_size=1_001),
        "chunk_overlap": replace(policy, chunk_overlap=149),
        "separators": replace(policy, separators=("\n", " ", "")),
        "splitter_version": replace(policy, splitter_version="recursive-character-v2"),
        "heading_prefix_version": replace(
            policy, heading_prefix_version="heading-path-v2"
        ),
    }
    changed_namespaces = {
        "provider_kind": _namespace(provider_kind="azure-openai-compatible"),
        "embedding_model": _namespace(embedding_model="text-embedding-3-large"),
        "base_url": _namespace(base_url="https://other.example.test/v1"),
        "requested_dimension": _namespace(requested_dimension=3_072),
        "resolved_dimension": _namespace(resolved_dimension=3_072),
        "vector_normalization": _namespace(vector_normalization="none"),
        "metric": _namespace(metric="l2"),
        "distance_strategy": _namespace(distance_strategy="euclidean"),
        "heading_prefix_behavior": _namespace(
            heading_prefix_behavior="heading-path-only"
        ),
        **{
            field: _namespace(chunking_policy=changed_policy)
            for field, changed_policy in changed_policies.items()
        },
    }

    assert set(changed_namespaces) == {
        "provider_kind",
        "embedding_model",
        "base_url",
        "requested_dimension",
        "resolved_dimension",
        "vector_normalization",
        "metric",
        "distance_strategy",
        "heading_prefix_behavior",
        "chunk_size",
        "chunk_overlap",
        "separators",
        "splitter_version",
        "heading_prefix_version",
    }
    for field, changed in changed_namespaces.items():
        assert changed.fingerprint != baseline.fingerprint, field
        assert changed.canonical_json != baseline.canonical_json, field


def test_url_credentials_query_and_fragment_are_not_fingerprint_inputs():
    clean = _namespace(base_url="https://api.example.test/v1")
    sensitive = _namespace(
        base_url=(
            "https://user:password@api.example.test:443/v1/"
            "?api-key=secret&tenant=private#token"
        )
    )

    assert sensitive == clean
    assert sensitive.fingerprint == clean.fingerprint
    serialized = sensitive.canonical_json
    assert "user" not in serialized
    assert "password" not in serialized
    assert "api-key" not in serialized
    assert "secret" not in serialized
    assert "private" not in serialized
    assert "token" not in serialized


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "api.example.test/v1",
        "ftp://api.example.test/v1",
        "https:///v1",
        "https://api.example.test:invalid/v1",
    ],
)
def test_base_url_rejects_malformed_or_incompatible_values(bad_url):
    with pytest.raises(embedding_cache.NamespaceValidationError):
        _namespace(base_url=bad_url)


def test_namespace_mapping_rejects_unknown_fields_types_and_unbounded_values():
    namespace = _namespace()
    valid = namespace.to_dict()
    assert embedding_cache.EmbeddingNamespace.from_mapping(valid) == namespace

    invalid_payloads = []
    unknown_top = copy.deepcopy(valid)
    unknown_top["api_key"] = "must-never-be-accepted"
    invalid_payloads.append(unknown_top)
    unknown_nested = copy.deepcopy(valid)
    cast(dict[str, object], unknown_nested["provider"])[
        "credential"
    ] = "must-never-be-accepted"
    invalid_payloads.append(unknown_nested)
    wrong_integer_type = copy.deepcopy(valid)
    cast(dict[str, object], wrong_integer_type["provider"])[
        "resolved_dimension"
    ] = True
    invalid_payloads.append(wrong_integer_type)
    wrong_string_type = copy.deepcopy(valid)
    cast(dict[str, object], wrong_string_type["vector"])["normalization"] = ["l2"]
    invalid_payloads.append(wrong_string_type)
    wrong_collection_type = copy.deepcopy(valid)
    cast(dict[str, object], wrong_collection_type["child_policy"])[
        "separators"
    ] = "not-a-list"
    invalid_payloads.append(wrong_collection_type)
    long_model = copy.deepcopy(valid)
    cast(dict[str, object], long_model["provider"])["embedding_model"] = (
        "m" * (embedding_cache.MAX_NAMESPACE_TEXT_CHARS + 1)
    )
    invalid_payloads.append(long_model)
    too_many_separators = copy.deepcopy(valid)
    cast(dict[str, object], too_many_separators["child_policy"])["separators"] = [
        "x"
    ] * (embedding_cache.MAX_SEPARATOR_COUNT + 1)
    invalid_payloads.append(too_many_separators)
    long_separator = copy.deepcopy(valid)
    cast(dict[str, object], long_separator["child_policy"])["separators"] = [
        "x" * (embedding_cache.MAX_SEPARATOR_CHARS + 1)
    ]
    invalid_payloads.append(long_separator)

    for payload in invalid_payloads:
        with pytest.raises(embedding_cache.NamespaceValidationError):
            embedding_cache.EmbeddingNamespace.from_mapping(payload)
        with pytest.raises(embedding_cache.NamespaceValidationError):
            embedding_cache.namespace_fingerprint(payload)

    with pytest.raises(embedding_cache.NamespaceValidationError):
        embedding_cache.EmbeddingNamespace.from_mapping([])
    with pytest.raises(embedding_cache.NamespaceValidationError):
        _namespace(base_url="https://example.test/" + "x" * embedding_cache.MAX_BASE_URL_CHARS)


def test_sqlite_schema_enforces_bounded_safe_records(tmp_path):
    cache_path = tmp_path / ".kb" / "embedding_cache.sqlite3"
    namespace = _namespace()
    embedding_cache.initialize_cache(cache_path, namespace)
    insert_sql = """
        INSERT INTO embedding_vectors(
            namespace_fingerprint, chunk_id, source_id, content_hash,
            dimension, vector_encoding, vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    with sqlite3.connect(cache_path) as conn:
        # SQLite foreign-key enforcement is connection-local. Production cache
        # transactions enable it, and this direct schema probe does the same.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(insert_sql, _vector_row(namespace))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                insert_sql,
                _vector_row(
                    namespace,
                    chunk_id="x" * (embedding_cache.MAX_ID_CHARS + 1),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                insert_sql,
                (
                    namespace.fingerprint,
                    "doc.md#bad-dimension::chunk-0",
                    "doc.md#bad-dimension",
                    "b" * 64,
                    embedding_cache.MAX_EMBEDDING_DIMENSION + 1,
                    embedding_cache.VECTOR_ENCODING,
                    b"",
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                insert_sql,
                (
                    namespace.fingerprint,
                    "doc.md#bad-bytes::chunk-0",
                    "doc.md#bad-bytes",
                    "c" * 64,
                    2,
                    embedding_cache.VECTOR_ENCODING,
                    b"wrong-size",
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO embedding_namespaces(
                    fingerprint, namespace_version, namespace_json
                ) VALUES (?, ?, ?)
                """,
                (
                    "d" * 64,
                    embedding_cache.NAMESPACE_VERSION,
                    "x" * (embedding_cache.MAX_NAMESPACE_JSON_BYTES + 1),
                ),
            )

        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0]

    assert foreign_keys == 1
    assert schema_version == embedding_cache.CACHE_SCHEMA_VERSION
    assert count == 1
    source = inspect.getsource(embedding_cache)
    assert "import pickle" not in source
    assert "pickle.loads" not in source
    assert "eval(" not in source


def test_cache_transaction_commits_and_rolls_back_as_one_bounded_unit(tmp_path):
    cache_path = tmp_path / ".kb" / "embedding_cache.sqlite3"
    namespace = _namespace()
    embedding_cache.initialize_cache(cache_path, namespace)
    insert_sql = """
        INSERT INTO embedding_vectors(
            namespace_fingerprint, chunk_id, source_id, content_hash,
            dimension, vector_encoding, vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    with embedding_cache.cache_transaction(cache_path) as conn:
        assert conn.in_transaction
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] <= (
            embedding_cache.SQLITE_TIMEOUT_SECONDS * 1_000
        )
        conn.execute(insert_sql, _vector_row(namespace))

    with sqlite3.connect(cache_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 1

    rolled_back_connection: sqlite3.Connection | None = None
    with pytest.raises(RuntimeError, match="abort candidate"):
        with embedding_cache.cache_transaction(cache_path) as conn:
            rolled_back_connection = conn
            conn.execute(
                insert_sql,
                _vector_row(namespace, "doc.md#answer::chunk-1"),
            )
            raise RuntimeError("abort candidate")

    with sqlite3.connect(cache_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 1
    assert rolled_back_connection is not None
    with pytest.raises(sqlite3.ProgrammingError):
        rolled_back_connection.execute("SELECT 1")


class CountingEmbeddingProvider:
    provider_kind = "openai-compatible"
    vector_normalization = "l2"
    metric = "inner-product"
    distance_strategy = "cosine"

    def __init__(
        self,
        *,
        model="counting-embedding-v1",
        base_url="https://api.example.test/v1",
        dimension=3,
        requested_dimension=3,
        invalid_response=None,
    ):
        self.model = model
        self.base_url = base_url
        self.dimension = dimension
        self.requested_dimension = requested_dimension
        self.invalid_response = invalid_response
        self.document_batches = []
        self.query_calls = []

    def _vector(self, text):
        value = float(sum(text.encode("utf-8")) % 97 + 1)
        return [value, 1.0, 0.5][: self.dimension]

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        vectors = [self._vector(text) for text in texts]
        if self.invalid_response == "count":
            return vectors[:-1]
        if self.invalid_response == "dimension" and vectors:
            vectors[0] = vectors[0][:-1]
        if self.invalid_response == "non-finite" and vectors:
            vectors[0][0] = math.nan
        if self.invalid_response == "non-numeric" and vectors:
            vectors[0] = json.loads('["secret-provider-payload", 1.0, 0.5]')[
                : self.dimension
            ]
        return vectors

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


def _child(number, text=None):
    source_id = f"doc-{number}.md#section"
    embedding_text = text or f"Heading {number}\n\ncontent {number}"
    return ChildChunk(
        chunk_id=f"{source_id}::chunk-0",
        source_id=source_id,
        file=f"doc-{number}.md",
        heading=f"Heading {number}",
        heading_path=[f"Heading {number}"],
        chunk_index=0,
        chunk_count=1,
        content=embedding_text.split("\n\n", 1)[-1],
        embedding_text=embedding_text,
        content_hash=embedded_content_hash(embedding_text),
        splitter_version="recursive-character-v1",
        heading_prefix_version="heading-path-v1",
    )


def _build(children, provider, cache_path, *, batch_size=2):
    return faiss_index.FaissSectionIndex.build(
        children,
        provider,
        cache_path=cache_path,
        batch_size=batch_size,
    )


def test_incremental_build_batches_only_misses_and_replaces_deleted_rows(
    embedding_cache_path,
):
    children = [_child(number) for number in range(5)]
    first_provider = CountingEmbeddingProvider()
    first = _build(children, first_provider, embedding_cache_path)
    assert [len(batch) for batch in first_provider.document_batches] == [2, 2, 1]
    assert first.source_ids == [child.chunk_id for child in children]

    identical_provider = CountingEmbeddingProvider()
    identical = _build(children, identical_provider, embedding_cache_path)
    assert identical_provider.document_batches == []
    assert identical.source_ids == first.source_ids

    changed = list(children)
    changed[2] = _child(2, "Heading 2\n\nchanged content")
    changed_provider = CountingEmbeddingProvider()
    _build(changed, changed_provider, embedding_cache_path)
    assert changed_provider.document_batches == [[changed[2].embedding_text]]

    retained = changed[:-1]
    deleted_provider = CountingEmbeddingProvider()
    current = _build(retained, deleted_provider, embedding_cache_path)
    assert deleted_provider.document_batches == []
    assert current.source_ids == [child.chunk_id for child in retained]

    with sqlite3.connect(embedding_cache_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_id, content_hash FROM embedding_vectors"
        ).fetchall()
    assert rows == [
        (child.chunk_id, child.source_id, child.content_hash) for child in retained
    ]


@pytest.mark.parametrize(
    "provider_overrides",
    [
        {"model": "counting-embedding-v2"},
        {"base_url": "https://other.example.test/v1"},
        {"requested_dimension": 2},
        {"dimension": 2, "requested_dimension": 2},
    ],
)
def test_provider_namespace_change_forces_document_cache_misses(
    embedding_cache_path, provider_overrides
):
    children = [_child(1), _child(2)]
    _build(children, CountingEmbeddingProvider(), embedding_cache_path)
    changed_provider = CountingEmbeddingProvider(**provider_overrides)
    _build(children, changed_provider, embedding_cache_path)
    assert changed_provider.document_batches == [
        [child.embedding_text for child in children]
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        ("chunk_id", "wrong::chunk-0"),
        ("source_id", "wrong-source"),
        ("content_hash", "f" * 64),
        ("vector_encoding", "wrong-encoding"),
        ("dimension", 2),
        ("vector", struct.pack("<3f", math.nan, 1.0, 0.5)),
        ("vector", struct.pack("<3f", math.inf, 1.0, 0.5)),
        ("vector", struct.pack("<3f", -math.inf, 1.0, 0.5)),
        ("vector", b"wrong-byte-count"),
    ],
)
def test_malformed_or_mismatched_cache_rows_are_misses_and_are_repaired(
    embedding_cache_path, mutation
):
    child = _child(1)
    _build([child], CountingEmbeddingProvider(), embedding_cache_path)
    column, value = mutation
    with sqlite3.connect(embedding_cache_path) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(f"UPDATE embedding_vectors SET {column} = ?", (value,))

    provider = CountingEmbeddingProvider()
    _build([child], provider, embedding_cache_path)
    assert provider.document_batches == [[child.embedding_text]]
    with sqlite3.connect(embedding_cache_path) as conn:
        row = conn.execute(
            """
            SELECT chunk_id, source_id, content_hash, dimension,
                   vector_encoding, length(vector)
            FROM embedding_vectors
            """
        ).fetchone()
    assert row == (
        child.chunk_id,
        child.source_id,
        child.content_hash,
        3,
        embedding_cache.VECTOR_ENCODING,
        12,
    )


@pytest.mark.parametrize("invalid_response", ["count", "dimension", "non-finite"])
def test_invalid_provider_document_responses_never_enter_cache(
    embedding_cache_path, invalid_response
):
    provider = CountingEmbeddingProvider(invalid_response=invalid_response)
    with pytest.raises(faiss_index.FaissIndexError):
        _build([_child(1), _child(2)], provider, embedding_cache_path)
    if embedding_cache_path.exists():
        with sqlite3.connect(embedding_cache_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 0


def test_query_embeddings_are_never_read_from_or_written_to_document_cache(
    embedding_cache_path,
):
    provider = CountingEmbeddingProvider()
    index = _build([_child(1)], provider, embedding_cache_path)
    provider.query_calls.clear()
    provider.document_batches.clear()

    index.search("same question", provider, k=1)
    index.search("same question", provider, k=1)

    assert provider.query_calls == ["same question", "same question"]
    assert provider.document_batches == []


def test_non_numeric_query_embedding_is_sanitized_provider_failure(
    embedding_cache_path,
):
    provider = CountingEmbeddingProvider()
    index = _build([_child(1)], provider, embedding_cache_path)

    class NonNumericQueryProvider(CountingEmbeddingProvider):
        def embed_query(self, text):
            self.query_calls.append(text)
            return json.loads('["secret-query-payload", 1.0, 0.5]')

    invalid_provider = NonNumericQueryProvider()
    with pytest.raises(faiss_index.FaissIndexError) as exc_info:
        index.search("same question", invalid_provider, k=1)

    assert "secret-query-payload" not in str(exc_info.value)


def _active_vector_rows(cache_path):
    with sqlite3.connect(cache_path) as conn:
        return conn.execute(
            """
            SELECT namespace_fingerprint, chunk_id, source_id, content_hash,
                   dimension, vector_encoding, vector
            FROM embedding_vectors
            ORDER BY namespace_fingerprint, chunk_id
            """
        ).fetchall()


def test_interrupted_candidate_activation_rolls_back_to_prior_mapping(
    embedding_cache_path, monkeypatch
):
    original_children = [_child(1), _child(2)]
    _build(original_children, CountingEmbeddingProvider(), embedding_cache_path)
    prior_rows = _active_vector_rows(embedding_cache_path)

    original_activate = embedding_cache._activate_candidate_rows

    def interrupt_after_candidate_write(conn, namespace_fingerprint, rows):
        original_activate(conn, namespace_fingerprint, rows)
        raise sqlite3.OperationalError("simulated activation interruption")

    monkeypatch.setattr(
        embedding_cache, "_activate_candidate_rows", interrupt_after_candidate_write
    )
    changed_children = [_child(1, "Heading 1\n\nchanged content")]
    with pytest.raises(faiss_index.FaissIndexError):
        _build(changed_children, CountingEmbeddingProvider(), embedding_cache_path)

    assert _active_vector_rows(embedding_cache_path) == prior_rows


@pytest.mark.parametrize("invalid_response", ["non-finite", "non-numeric"])
def test_invalid_changed_candidate_keeps_prior_active_mapping(
    embedding_cache_path, invalid_response
):
    original_children = [_child(1), _child(2)]
    _build(original_children, CountingEmbeddingProvider(), embedding_cache_path)
    prior_rows = _active_vector_rows(embedding_cache_path)

    changed_children = [_child(1, "Heading 1\n\nchanged content")]
    with pytest.raises(faiss_index.FaissIndexError) as exc_info:
        _build(
            changed_children,
            CountingEmbeddingProvider(invalid_response=invalid_response),
            embedding_cache_path,
        )

    assert "secret-provider-payload" not in str(exc_info.value)
    assert _active_vector_rows(embedding_cache_path) == prior_rows


def test_corpus_change_during_embedding_aborts_cache_activation(
    embedding_cache_path,
):
    original_children = [_child(1), _child(2)]
    indexer.child_chunks = original_children
    indexer.refresh_runtime_state()
    indexer.embedding_provider_override = CountingEmbeddingProvider()
    assert indexer.build_faiss_index(persist=False) is not None
    prior_rows = _active_vector_rows(embedding_cache_path)

    changed_children = [_child(1, "Heading 1\n\nchanged content")]
    indexer.child_chunks = changed_children
    indexer.refresh_runtime_state()

    class CorpusMutatingProvider(CountingEmbeddingProvider):
        def embed_documents(self, texts):
            vectors = super().embed_documents(texts)
            indexer.child_chunks = [_child(99)]
            indexer.refresh_runtime_state()
            return vectors

    provider = CorpusMutatingProvider()
    indexer.embedding_provider_override = provider

    assert indexer.build_faiss_index(persist=False) is None
    assert provider.document_batches == [[changed_children[0].embedding_text]]
    assert _active_vector_rows(embedding_cache_path) == prior_rows
