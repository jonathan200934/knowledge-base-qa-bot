import json
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import faiss_index, hybrid, indexer, reranker
from app.chunking import ChunkingPolicy
from app.faiss_index import (
    FAISS_INDEX_VERSION,
    FaissIndexError,
    FaissSectionIndex,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from app.main import load_persisted_index
from app.routes import router


def make_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _openai_provider_with_failing_sdk(secret: str = "secret-sdk-payload"):
    class FailingEmbeddings:
        def create(self, **_request):
            raise RuntimeError(secret)

    class FailingClient:
        embeddings = FailingEmbeddings()

    provider = object.__new__(OpenAIEmbeddingProvider)
    provider.model = "sdk-failure-provider"
    provider.api_key = "[REDACTED]"
    provider.base_url = "https://provider.invalid/v1"
    provider.provider_kind = "openai-compatible"
    provider.requested_dimension = 2
    provider.dimension = 2
    setattr(provider, "_client", FailingClient())
    return provider


def test_faiss_index_builds_metadata_and_searches_with_fake_embeddings(monkeypatch, sample_docs_dir):
    monkeypatch.setattr(indexer, "embedding_provider_override", HashEmbeddingProvider("fake-openai-test"))

    files_count, sections_count = indexer.build_index(sample_docs_dir)

    assert files_count >= 10
    assert sections_count >= 30
    faiss_dir = indexer.faiss_index_dir()
    assert (faiss_dir / "index.faiss").exists()
    manifest = json.loads((faiss_dir / "manifest.json").read_text(encoding="utf-8"))
    children = json.loads((faiss_dir / "children.json").read_text(encoding="utf-8"))
    assert manifest["index_version"] == FAISS_INDEX_VERSION
    assert manifest["provider"]["model"] == "fake-openai-test"
    assert manifest["index"]["dimension"] > 0
    assert manifest["counts"]["chunks"] == len(children)
    assert manifest["counts"]["sections"] == sections_count
    assert "refund_policy.md#refund-timeline" in {
        child["source_id"] for child in children
    }

    ranked = indexer.vector_search("How long do refunds take?", k=3)

    assert ranked
    assert ranked[0][0].id == "refund_policy.md#refund-timeline"
    assert ranked[0][1] > 0


def test_startup_loads_compatible_faiss_index_after_restart(monkeypatch, sample_docs_dir):
    monkeypatch.setattr(indexer, "embedding_provider_override", HashEmbeddingProvider("fake-openai-test"))
    indexer.build_index(sample_docs_dir)
    expected = [(section.id, round(score, 6)) for section, score in indexer.vector_search("Can I change my email address?", k=3)]
    assert expected[0][0] == "account_help.md#change-email-address"

    indexer.sections = []
    indexer.section_vector_index = None
    indexer.faiss_section_index = None
    load_persisted_index()
    loaded = [(section.id, round(score, 6)) for section, score in indexer.vector_search("Can I change my email address?", k=3)]

    assert loaded == expected


def test_faiss_compatibility_hash_covers_exact_embedding_payload_hash_and_versions():
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Exact embedded answer payload.",
        tokens=["exact", "embedded", "answer", "payload"],
    )
    child = indexer.create_child_chunks([parent])[0]
    provider = HashEmbeddingProvider("compatibility-test")
    built = FaissSectionIndex.build([child], provider)

    changed_children = [
        replace(child, embedding_text=child.embedding_text + " changed"),
        replace(child, content_hash="0" * 64),
        replace(child, splitter_version=child.splitter_version + "-changed"),
        replace(
            child,
            heading_prefix_version=child.heading_prefix_version + "-changed",
        ),
    ]

    assert built.is_compatible([child], provider.model)
    assert all(
        not built.is_compatible([changed], provider.model)
        for changed in changed_children
    )


def test_repeated_faiss_search_reuses_runtime_maps_and_compatibility_generation(
    monkeypatch,
):
    parents = [
        indexer.Section(
            id=f"{name}.md#answer",
            file=f"{name}.md",
            heading="Answer",
            heading_path=[name.title(), "Answer"],
            content=f"Shared searchable answer for {name}.",
            tokens=["shared", "searchable", "answer", name],
        )
        for name in ("alpha", "beta")
    ]
    indexer.sections = parents
    indexer.child_chunks = indexer.create_child_chunks(parents)
    indexer.refresh_runtime_state()
    monkeypatch.setattr(
        indexer,
        "embedding_provider_override",
        HashEmbeddingProvider("runtime-generation-test"),
    )
    built = indexer.build_faiss_index(persist=False)
    assert built is not None
    section_map = indexer._section_by_id
    child_map = indexer._child_by_id

    refresh_calls = 0
    original_refresh = indexer.refresh_runtime_state

    def recording_refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        return original_refresh()

    def unexpected_linear_work(*_args, **_kwargs):
        raise AssertionError("repeated search performed corpus-linear compatibility work")

    monkeypatch.setattr(indexer, "refresh_runtime_state", recording_refresh)
    monkeypatch.setattr(faiss_index, "section_hash", unexpected_linear_work)
    monkeypatch.setattr(FaissSectionIndex, "is_compatible", unexpected_linear_work)

    first = indexer.vector_search("shared searchable", k=1)
    second = indexer.vector_search("shared searchable", k=1)

    assert first == second
    assert first
    assert refresh_calls == 0
    assert indexer._section_by_id is section_map
    assert indexer._child_by_id is child_map


def test_runtime_maps_detect_replaced_global_corpora_without_stale_parent_lookup(
    monkeypatch,
):
    old_parent = indexer.Section(
        id="old.md#answer",
        file="old.md",
        heading="Answer",
        heading_path=["Old", "Answer"],
        content="Old answer.",
        tokens=["old", "answer"],
    )
    indexer.sections = [old_parent]
    indexer.child_chunks = indexer.create_child_chunks([old_parent])
    indexer.refresh_runtime_state()

    new_parent = replace(
        old_parent,
        id="new.md#answer",
        file="new.md",
        heading_path=["New", "Answer"],
        content="New answer.",
        tokens=["new", "answer"],
    )
    new_child = indexer.create_child_chunks([new_parent])[0]
    indexer.sections = [new_parent]
    indexer.child_chunks = [new_child]

    class NewCorpusIndex:
        def search(self, _query, k=3):
            return [(new_child.chunk_id, 0.9)][:k]

    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: None)
    monkeypatch.setattr(indexer, "ensure_vector_index_loaded", lambda: NewCorpusIndex())

    results = indexer.vector_search("new", k=1)

    assert results == [(new_parent, 0.9)]
    assert indexer._section_by_id == {new_parent.id: new_parent}
    assert indexer._child_by_id == {new_child.chunk_id: new_child}


def test_unexpected_runtime_errors_propagate_from_faiss_build_load_and_search(
    monkeypatch,
):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Answer content.",
        tokens=["answer", "content"],
    )
    indexer.sections = [parent]
    indexer.child_chunks = indexer.create_child_chunks([parent])
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: object())

    def unexpected(*_args, **_kwargs):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(FaissSectionIndex, "build", classmethod(unexpected))
    with pytest.raises(RuntimeError, match="programming bug"):
        indexer.build_faiss_index(persist=False)

    monkeypatch.setattr(FaissSectionIndex, "load", classmethod(unexpected))
    with pytest.raises(RuntimeError, match="programming bug"):
        indexer.load_faiss_index()

    class BrokenSearchIndex:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("programming bug")

    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: BrokenSearchIndex())
    with pytest.raises(RuntimeError, match="programming bug"):
        indexer.vector_search("answer", k=1)


def test_expected_faiss_failure_falls_back_locally_with_sanitized_status(monkeypatch):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Local fallback answer.",
        tokens=["local", "fallback", "answer"],
    )
    child = indexer.create_child_chunks([parent])[0]
    indexer.sections = [parent]
    indexer.child_chunks = [child]

    class UnavailableFaissIndex:
        def search(self, *_args, **_kwargs):
            raise FaissIndexError("secret-provider-payload")

    class LocalFallbackIndex:
        def search(self, _tokens, k=3):
            return [(child.chunk_id, 0.75)][:k]

    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: UnavailableFaissIndex())
    monkeypatch.setattr(indexer, "ensure_vector_index_loaded", lambda: LocalFallbackIndex())
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: object())

    results = indexer.vector_search("local fallback", k=1)

    assert results == [(parent, 0.75)]
    assert indexer.last_faiss_status["status"] == "query_failed"
    assert "secret-provider-payload" not in json.dumps(indexer.last_faiss_status)


def test_query_embedding_failure_never_searches_native_faiss_and_uses_local_vector(monkeypatch):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Deterministic local fallback answer.",
        tokens=["deterministic", "local", "fallback", "answer"],
    )
    child = indexer.create_child_chunks([parent])[0]
    indexer.sections = [parent]
    indexer.child_chunks = [child]
    indexer.refresh_runtime_state()
    local_index = indexer.rebuild_vector_index(persist=False)

    built = FaissSectionIndex.build([child], HashEmbeddingProvider("fake-query-provider"))

    class NativeSearchSpy:
        def __init__(self):
            self.calls = 0

        def search(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("native FAISS search must not run without a valid query vector")

    native_spy = NativeSearchSpy()
    built.index = native_spy

    class FailingQueryProvider(HashEmbeddingProvider):
        def embed_query(self, text):
            del text
            raise FaissIndexError("provider query failure with secret details")

    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: built)
    monkeypatch.setattr(indexer, "ensure_vector_index_loaded", lambda: local_index)
    monkeypatch.setattr(
        indexer,
        "get_embedding_provider",
        lambda: FailingQueryProvider("fake-query-provider"),
    )

    results = indexer.vector_search("local fallback", k=1)

    assert results and results[0][0] == parent
    assert native_spy.calls == 0
    assert indexer.faiss_section_index is None
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "query_failed",
        "reason": indexer.FAISS_QUERY_FAILED_REASON,
    }


def test_openai_sdk_exception_is_normalized_without_provider_payload():
    provider = _openai_provider_with_failing_sdk()

    with pytest.raises(FaissIndexError) as raised:
        provider.embed_documents(["safe text"])

    assert str(raised.value) == "OpenAI embedding request failed"
    assert "secret-sdk-payload" not in str(raised.value)


def test_real_openai_query_sdk_failure_uses_deterministic_local_fallback(monkeypatch):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Deterministic local fallback answer.",
        tokens=["deterministic", "local", "fallback", "answer"],
    )
    child = indexer.create_child_chunks([parent])[0]
    indexer.sections = [parent]
    indexer.child_chunks = [child]
    indexer.refresh_runtime_state()
    local_index = indexer.rebuild_vector_index(persist=False)
    built = FaissSectionIndex.build(
        [child], HashEmbeddingProvider("sdk-failure-provider", dimension=2)
    )
    provider = _openai_provider_with_failing_sdk()

    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: built)
    monkeypatch.setattr(indexer, "ensure_vector_index_loaded", lambda: local_index)
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: provider)

    results = indexer.vector_search("local fallback", k=1)

    assert results and results[0][0] == parent
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "query_failed",
        "reason": indexer.FAISS_QUERY_FAILED_REASON,
    }


def test_real_openai_document_sdk_failure_keeps_local_retrieval(monkeypatch):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Local retrieval remains available.",
        tokens=["local", "retrieval", "remains", "available"],
    )
    indexer.sections = [parent]
    indexer.child_chunks = indexer.create_child_chunks([parent])
    indexer.refresh_runtime_state()
    indexer.rebuild_vector_index(persist=False)
    monkeypatch.setattr(
        indexer, "embedding_provider_override", _openai_provider_with_failing_sdk()
    )

    assert indexer.build_faiss_index(persist=False) is None
    assert indexer.vector_search("local retrieval", k=1)
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "unavailable",
        "reason": indexer.FAISS_BUILD_UNAVAILABLE_REASON,
    }


def test_document_embedding_failure_keeps_new_local_corpus_and_clears_stale_faiss(
    monkeypatch, sample_docs_dir
):
    class FailingDocumentProvider(HashEmbeddingProvider):
        def embed_documents(self, texts):
            del texts
            raise FaissIndexError("provider document failure with secret details")

    indexer.faiss_section_index = object()
    indexer._faiss_runtime_generation = 123
    indexer._faiss_runtime_model = "stale-provider"
    monkeypatch.setattr(
        indexer,
        "embedding_provider_override",
        FailingDocumentProvider("failing-document-provider"),
    )

    files_count, sections_count = indexer.build_index(sample_docs_dir)

    assert files_count >= 10
    assert sections_count >= 30
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "unavailable",
        "reason": indexer.FAISS_BUILD_UNAVAILABLE_REASON,
    }
    assert indexer.search("refund timeline", k=1)
    assert indexer.ensure_vector_index_loaded() is not None
    assert indexer.vector_search("refund timeline", k=1)
    assert indexer.faiss_section_index is None
    assert indexer._faiss_runtime_generation is None
    assert indexer._faiss_runtime_model is None


def test_missing_openai_configuration_remains_a_graceful_sanitized_fallback():
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Local fallback answer.",
        tokens=["local", "fallback", "answer"],
    )
    indexer.sections = [parent]
    indexer.child_chunks = indexer.create_child_chunks([parent])
    indexer.faiss_section_index = object()
    indexer._faiss_runtime_generation = 123
    indexer._faiss_runtime_model = "stale-provider"

    assert indexer.ensure_faiss_index_loaded() is None
    assert indexer.faiss_section_index is None
    assert indexer._faiss_runtime_generation is None
    assert indexer._faiss_runtime_model is None
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "unavailable",
        "reason": "FAISS embedding provider is unavailable",
    }


def test_loaded_runtime_rechecks_complete_provider_identity(monkeypatch):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Provider identity answer.",
        tokens=["provider", "identity", "answer"],
    )
    child = indexer.create_child_chunks([parent])[0]
    original_provider = HashEmbeddingProvider("same-model")
    changed_provider = HashEmbeddingProvider("same-model")
    changed_provider.base_url = "https://changed-provider.invalid/v1"
    built = FaissSectionIndex.build([child], original_provider)
    indexer.sections = [parent]
    indexer.child_chunks = [child]
    indexer.faiss_section_index = built
    indexer._runtime_generation = 7
    indexer._faiss_runtime_generation = 7
    indexer._faiss_runtime_model = "same-model"
    load_calls = []

    monkeypatch.setattr(indexer, "_ensure_runtime_state_current", lambda: None)
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: changed_provider)
    monkeypatch.setattr(indexer, "load_faiss_index", lambda: load_calls.append(True) or None)

    assert indexer.ensure_faiss_index_loaded() is None
    assert load_calls == [True]
    assert indexer.faiss_section_index is None


def test_persisted_faiss_load_uses_active_child_chunking_policy(monkeypatch):
    policy = ChunkingPolicy(
        chunk_size=512,
        chunk_overlap=64,
        splitter_version="test-splitter-v2",
        heading_prefix_version="test-heading-v2",
    )
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content="Custom policy answer.",
        tokens=["custom", "policy", "answer"],
    )
    child = indexer.create_child_chunks([parent], policy)[0]
    provider = HashEmbeddingProvider("custom-policy-provider")
    built = FaissSectionIndex.build([child], provider, chunking_policy=policy)
    indexer.sections = [parent]
    indexer.child_chunks = [child]
    indexer.refresh_runtime_state()
    calls = []

    def record_load(directory, children, selected_provider, **kwargs):
        calls.append((directory, children, selected_provider, kwargs))
        return built

    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: provider)
    monkeypatch.setattr(indexer.FaissSectionIndex, "load", record_load)

    assert indexer.load_faiss_index() is built
    assert calls[0][3] == {"chunking_policy": policy}


def test_reranker_reorders_fused_candidates_and_keeps_rrf_debug_fields():
    semantic_only = indexer.Section(
        id="generic.md#refunds",
        file="generic.md",
        heading="Refunds",
        heading_path=["Refunds"],
        content="Refund requests can be submitted online.",
        tokens=["refund", "requests", "submitted", "online"],
    )
    exact_answer = indexer.Section(
        id="refund_policy.md#refund-timeline",
        file="refund_policy.md",
        heading="Refund Timeline",
        heading_path=["Refund Policy", "Refund Timeline"],
        content="Approved refunds are processed within 5-7 business days.",
        tokens=["refund", "timeline", "approved", "refunds", "processed", "5", "7", "business", "days"],
    )
    fused = [
        hybrid.HybridResult(section=semantic_only, score=0.04, vector_rank=1, vector_score=0.91),
        hybrid.HybridResult(section=exact_answer, score=0.02, bm25_rank=1, bm25_score=2.5),
    ]

    reranked = reranker.rerank_hybrid_results("How long do refunds take, 5-7 business days?", fused, k=2)

    assert [result.section.id for result in reranked] == [
        "refund_policy.md#refund-timeline",
        "generic.md#refunds",
    ]
    assert reranked[0].rrf_score == 0.02
    assert reranked[0].rerank_rank == 1
    assert reranked[0].rerank_score is not None
    assert reranked[0].score == reranked[0].rerank_score
    assert reranked[0].bm25_rank == 1
    assert reranked[1].vector_rank == 1


def test_reranker_drops_semantic_only_candidate_without_lexical_overlap():
    refund_timeline = indexer.Section(
        id="refund_policy.md#refund-timeline",
        file="refund_policy.md",
        heading="Refund Timeline",
        heading_path=["Refund Policy", "Refund Timeline"],
        content=(
            "Approved refunds are processed within 5-7 business days. "
            "The exact arrival time depends on the customer's bank or card provider."
        ),
        tokens=["approved", "refunds", "processed", "5", "7", "business", "days"],
    )
    shipping_timeline = indexer.Section(
        id="shipping_faq.md#standard-shipping",
        file="shipping_faq.md",
        heading="Standard Shipping",
        heading_path=["Shipping FAQ", "Standard Shipping"],
        content="Standard shipping usually takes 3-5 business days.",
        tokens=["standard", "shipping", "usually", "takes", "3", "5", "business", "days"],
    )
    fused = [
        hybrid.HybridResult(
            section=refund_timeline,
            score=1 / 61,
            vector_rank=1,
            vector_score=0.498908,
            rrf_score=1 / 61,
        ),
        hybrid.HybridResult(
            section=shipping_timeline,
            score=1 / 70,
            vector_rank=10,
            vector_score=0.268841,
            rrf_score=1 / 70,
        ),
    ]

    reranked = reranker.rerank_hybrid_results(
        "How long does it take to get my money back?",
        fused,
        k=2,
    )

    assert [getattr(result.section, "id") for result in reranked] == [
        "shipping_faq.md#standard-shipping"
    ]


def test_compare_response_includes_reranked_stage(monkeypatch, sample_docs_dir):
    monkeypatch.setattr(indexer, "embedding_provider_override", HashEmbeddingProvider("fake-openai-test"))
    monkeypatch.setenv("INDEX_API_KEY", "test-index-key")
    client = make_test_client()
    assert (
        client.post("/index", headers={"X-Index-Key": "test-index-key"}).status_code
        == 200
    )

    response = client.post("/compare", json={"query": "How long do refunds take?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed"] is True
    assert payload["bm25"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["vector"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["hybrid"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["reranked"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["reranked"][0]["rrf_score"] > 0
    assert payload["reranked"][0]["rerank_rank"] == 1
    assert payload["reranked"][0]["rerank_score"] > 0
