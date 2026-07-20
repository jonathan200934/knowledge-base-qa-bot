import hashlib
import unicodedata
from dataclasses import replace

import pytest

from app import hybrid, indexer, reranker, retrieval
from app.chunking import (
    DEFAULT_CHUNKING_POLICY,
    HEADING_PREFIX_VERSION,
    SPLITTER_VERSION,
    ChunkingPolicy,
    chunk_section,
)
from app.faiss_index import FaissSectionIndex
from app.vector_index import LocalVectorIndex


def make_section(*, source_id="guide.md#details", content: str, heading_path=None):
    path = heading_path or ["Guide", "Details"]
    return indexer.Section(
        id=source_id,
        file=source_id.partition("#")[0],
        heading=path[-1],
        heading_path=list(path),
        content=content,
        tokens=indexer.tokenize("\n".join([*path, content])),
    )


def test_long_section_has_deterministic_bounded_overlapping_children():
    policy = ChunkingPolicy(chunk_size=80, chunk_overlap=20, separators=("",))
    section = make_section(content="".join(f"{number:04d}" for number in range(80)))

    first = chunk_section(section, policy)
    second = chunk_section(section, policy)

    assert first == second
    assert len(first) > 1
    assert all(0 < len(child.content) <= policy.chunk_size for child in first)
    assert [child.chunk_index for child in first] == list(range(len(first)))
    assert all(child.chunk_count == len(first) for child in first)
    assert [child.chunk_id for child in first] == [
        f"guide.md#details::chunk-{ordinal}" for ordinal in range(len(first))
    ]
    for previous, current in zip(first, first[1:]):
        assert previous.content[-policy.chunk_overlap :] == current.content[: policy.chunk_overlap]


def test_short_section_creates_one_child_without_mutating_parent_content():
    original = "A short answer remains the full canonical parent section."
    section = make_section(content=original)

    children = chunk_section(section)

    assert section.content == original
    assert len(children) == 1
    child = children[0]
    assert child.source_id == section.id
    assert child.chunk_id == f"{section.id}::chunk-0"
    assert child.chunk_index == 0
    assert child.chunk_count == 1
    assert child.content == original


def test_every_child_repeats_versioned_heading_path_context_for_embedding():
    policy = ChunkingPolicy(chunk_size=50, chunk_overlap=10, separators=("",))
    section = make_section(
        source_id="api.md#authentication-v2",
        heading_path=["API Guide", "Authentication", "Version 2"],
        content="Version two authentication details. " * 8,
    )

    children = chunk_section(section, policy)

    assert len(children) > 1
    assert policy.splitter_version == SPLITTER_VERSION
    assert policy.heading_prefix_version == HEADING_PREFIX_VERSION
    assert DEFAULT_CHUNKING_POLICY.splitter_version == SPLITTER_VERSION
    assert DEFAULT_CHUNKING_POLICY.heading_prefix_version == HEADING_PREFIX_VERSION
    for child in children:
        assert child.heading_path == section.heading_path
        assert child.heading == section.heading
        assert child.embedding_text == (
            "API Guide > Authentication > Version 2\n\n" + child.content
        )
        assert child.splitter_version == SPLITTER_VERSION
        assert child.heading_prefix_version == HEADING_PREFIX_VERSION


def test_content_hash_is_deterministic_sha256_of_normalized_embedded_text():
    decomposed = "Cafe\u0301 policy\r\n\r\nDetails"
    composed = unicodedata.normalize("NFC", decomposed).replace("\r\n", "\n")
    first = chunk_section(make_section(content=decomposed))[0]
    second = chunk_section(make_section(content=composed))[0]

    assert first.embedding_text == second.embedding_text
    assert first.content_hash == second.content_hash
    assert first.content_hash == hashlib.sha256(first.embedding_text.encode("utf-8")).hexdigest()
    assert len(first.content_hash) == 64


def test_duplicate_headings_get_distinct_parent_and_child_ids(tmp_path):
    markdown = tmp_path / "faq.md"
    markdown.write_text(
        "# FAQ\n\n"
        "## Same Heading\n\nFirst answer.\n\n"
        "## Same Heading\n\nSecond answer.\n",
        encoding="utf-8",
    )

    parents = indexer.parse_markdown(markdown)
    children = indexer.create_child_chunks(parents)

    assert [parent.id for parent in parents] == [
        "faq.md#same-heading",
        "faq.md#same-heading-2",
    ]
    assert [child.source_id for child in children] == [
        "faq.md#same-heading",
        "faq.md#same-heading-2",
    ]
    assert len({child.chunk_id for child in children}) == len(children)


def test_index_build_materializes_children_from_canonical_sections_without_parent_changes(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    markdown = docs_dir / "versions.md"
    markdown.write_text(
        "# Product Guide\n\n"
        "## Setup v1\n\n" + ("Legacy setup details. " * 80) + "\n\n"
        "## Setup v2\n\nCurrent setup details.\n",
        encoding="utf-8",
    )

    indexer.build_index(docs_dir)

    parent_snapshot = [section.to_dict() for section in indexer.sections]
    expected = indexer.create_child_chunks(indexer.sections)
    assert indexer.child_chunks == expected
    assert [section.to_dict() for section in indexer.sections] == parent_snapshot
    assert {child.source_id for child in indexer.child_chunks} == {
        "versions.md#setup-v1",
        "versions.md#setup-v2",
    }
    assert all(child.source_id in {section.id for section in indexer.sections} for child in indexer.child_chunks)


class RecordingEmbeddingProvider:
    model = "recording-child-embeddings"
    dimension = 2
    requested_dimension = 2

    def __init__(self):
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [[1.0, float(position + 1)] for position, _text in enumerate(texts)]

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


def test_dense_backends_build_from_child_embedding_text_and_identity(monkeypatch):
    parent = make_section(
        source_id="guide.md#long-answer",
        heading_path=["Guide", "Long Answer"],
        content=("focused child passage " * 80),
    )
    policy = ChunkingPolicy(chunk_size=120, chunk_overlap=20, separators=("",))
    indexer.sections = [parent]
    indexer.child_chunks = indexer.create_child_chunks(indexer.sections, policy)
    assert len(indexer.child_chunks) > 1

    local_index = indexer.rebuild_vector_index(persist=False)
    child_ids = [child.chunk_id for child in indexer.child_chunks]
    assert [record.id for record in local_index.records] == sorted(child_ids)
    assert parent.id not in local_index.section_ids
    assert local_index.search(["focused"], k=len(child_ids))[0][0].startswith(
        f"{parent.id}::chunk-"
    )

    provider = RecordingEmbeddingProvider()
    monkeypatch.setattr(indexer, "embedding_provider_override", provider)
    faiss_index = indexer.build_faiss_index(persist=False)

    assert faiss_index is not None
    assert provider.document_batches == [
        [child.embedding_text for child in indexer.child_chunks]
    ]
    assert faiss_index.metadata.source_ids == child_ids
    assert faiss_index.metadata.section_count == len(indexer.child_chunks)
    assert parent.id not in faiss_index.metadata.source_ids


def test_dense_backend_ties_are_stable_by_internal_child_identity():
    parents = [
        make_section(source_id="z.md#answer", content="shared answer"),
        make_section(source_id="a.md#answer", content="shared answer"),
    ]
    children = indexer.create_child_chunks(parents)

    local_index = LocalVectorIndex.build(children)
    expected_child_ids = sorted(child.chunk_id for child in children)
    assert [source_id for source_id, _score in local_index.search(["shared"], k=2)] == expected_child_ids
    assert local_index.search(["shared"], k=2) == local_index.search(["shared"], k=2)

    class TiedProvider:
        model = "tied-child-embeddings"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _text in texts]

        def embed_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    provider = TiedProvider()
    faiss_index = FaissSectionIndex.build(children, provider)
    assert [source_id for source_id, _score in faiss_index.search("shared", provider, k=2)] == expected_child_ids
    assert faiss_index.search("shared", provider, k=2) == faiss_index.search(
        "shared", provider, k=2
    )


def _install_child_hit_fixture(monkeypatch):
    parents = [
        make_section(
            source_id="z.md#best",
            heading_path=["Zed", "Best"],
            content="FULL ZED PARENT CONTENT " + ("repeated passage " * 20),
        ),
        make_section(
            source_id="a.md#tied",
            heading_path=["Alpha", "Tied"],
            content="FULL ALPHA PARENT CONTENT",
        ),
        make_section(
            source_id="b.md#tied",
            heading_path=["Beta", "Tied"],
            content="FULL BETA PARENT CONTENT",
        ),
    ]
    policy = ChunkingPolicy(chunk_size=80, chunk_overlap=10, separators=("",))
    indexer.sections = parents
    indexer.child_chunks = indexer.create_child_chunks(parents, policy)
    children_by_parent = {
        parent.id: [
            child for child in indexer.child_chunks if child.source_id == parent.id
        ]
        for parent in parents
    }
    assert len(children_by_parent["z.md#best"]) > 1

    child_hits = [
        (children_by_parent["z.md#best"][0].chunk_id, 0.40),
        (children_by_parent["z.md#best"][1].chunk_id, 0.99),
        (children_by_parent["b.md#tied"][0].chunk_id, 0.80),
        (children_by_parent["a.md#tied"][0].chunk_id, 0.80),
    ]

    class FakeDenseIndex:
        def __init__(self):
            self.requested_ks: list[int] = []

        def search(self, _query, _provider, k=3):
            self.requested_ks.append(k)
            return child_hits[:k]

    dense_index = FakeDenseIndex()
    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: dense_index)
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: object())
    return parents, dense_index


def test_vector_search_overfetches_children_and_emits_best_full_parent_once(monkeypatch):
    parents, dense_index = _install_child_hit_fixture(monkeypatch)

    results = indexer.vector_search("anything", k=2)

    assert dense_index.requested_ks[0] > 2
    assert [(section.id, score) for section, score in results] == [
        ("z.md#best", pytest.approx(0.99)),
        ("a.md#tied", pytest.approx(0.80)),
    ]
    assert results[0][0] is parents[0]
    assert results[0][0].content == "FULL ZED PARENT CONTENT " + (
        "repeated passage " * 20
    )
    assert all("::chunk-" not in section.id for section, _score in results)
    assert all(section.content.startswith("FULL ") for section, _score in results)


def test_vector_search_adaptively_grows_past_dominant_parent_and_score_cutoff_ties(
    monkeypatch,
):
    parents = [
        make_section(source_id=f"{name}.md#answer", content=f"FULL {name} ANSWER")
        for name in ("dominant", "a", "b", "z", "low")
    ]
    parent_by_id = {parent.id: parent for parent in parents}
    children = []

    def add_children(source_id: str, count: int):
        base = indexer.create_child_chunks([parent_by_id[source_id]])[0]
        made = [
            replace(
                base,
                chunk_id=f"{source_id}::chunk-{ordinal}",
                chunk_index=ordinal,
                chunk_count=count,
            )
            for ordinal in range(count)
        ]
        children.extend(made)
        return made

    dominant = add_children("dominant.md#answer", 32)
    tied_z = add_children("z.md#answer", 8)
    tied_b = add_children("b.md#answer", 8)
    tied_a = add_children("a.md#answer", 2)
    low = add_children("low.md#answer", 10)
    indexer.sections = parents
    indexer.child_chunks = children

    # Model a backend whose arbitrary equal-score subset changes as k grows.
    # The lexicographically smallest tied parent does not appear until after the
    # first score-cutoff boundary, so stopping at equality would be incorrect.
    child_hits = [
        (child.chunk_id, 0.99 - ordinal * 0.005)
        for ordinal, child in enumerate(dominant[:12])
    ]
    child_hits.extend((child.chunk_id, 0.80) for child in tied_z)
    child_hits.extend((child.chunk_id, 0.80) for child in tied_b[:4])
    child_hits.extend((child.chunk_id, 0.80) for child in tied_a)
    child_hits.extend((child.chunk_id, 0.80) for child in tied_b[4:])
    child_hits.extend((child.chunk_id, 0.79 - ordinal * 0.001) for ordinal, child in enumerate(low))
    child_hits.extend(
        (child.chunk_id, 0.70 - ordinal * 0.001)
        for ordinal, child in enumerate(dominant[12:])
    )
    assert len(child_hits) == len(children)

    class AdaptiveDenseIndex:
        def __init__(self):
            self.requested_ks: list[int] = []

        def search(self, _query, _provider, k=3):
            self.requested_ks.append(k)
            selected = child_hits[:k]
            return sorted(selected, key=lambda item: (-item[1], item[0]))

    dense_index = AdaptiveDenseIndex()
    monkeypatch.setattr(indexer, "ensure_faiss_index_loaded", lambda: dense_index)
    monkeypatch.setattr(indexer, "get_embedding_provider", lambda: object())

    results = indexer.vector_search("anything", k=2)

    assert dense_index.requested_ks == [10, 20, 40]
    assert dense_index.requested_ks[-1] < len(children)
    assert [(section.id, score) for section, score in results] == [
        ("dominant.md#answer", pytest.approx(0.99)),
        ("a.md#answer", pytest.approx(0.80)),
    ]


def test_parent_aggregation_happens_before_rrf_and_before_reranking(monkeypatch):
    parents, _dense_index = _install_child_hit_fixture(monkeypatch)
    monkeypatch.setattr(indexer, "search", lambda _query, k=3: [(parents[2], 500_000.0)])

    original_merge_rrf = hybrid.merge_rrf
    vector_inputs: list[list[tuple[object, float]]] = []

    def inspecting_merge_rrf(bm25_results, vector_results, k=3, rrf_k=hybrid.RRF_K):
        materialized_vector = list(vector_results)
        vector_inputs.append(materialized_vector)
        assert [section.id for section, _score in materialized_vector] == [
            "z.md#best",
            "a.md#tied",
            "b.md#tied",
        ]
        assert all("::chunk-" not in section.id for section, _score in materialized_vector)
        return original_merge_rrf(
            bm25_results,
            materialized_vector,
            k=k,
            rrf_k=rrf_k,
        )

    monkeypatch.setattr(hybrid, "merge_rrf", inspecting_merge_rrf)
    fused = indexer.hybrid_rrf_search("anything", k=3)

    assert vector_inputs
    by_id = {result.section.id: result for result in fused}
    assert by_id["z.md#best"].vector_score == pytest.approx(0.99)
    assert by_id["b.md#tied"].bm25_score == 500_000.0
    assert by_id["b.md#tied"].score == pytest.approx(
        1 / (hybrid.RRF_K + 1) + 1 / (hybrid.RRF_K + 3)
    )

    reranker_inputs: list[list[hybrid.HybridResult]] = []

    def inspecting_reranker(_query, fused_results, k=3):
        materialized = list(fused_results)
        reranker_inputs.append(materialized)
        assert len({result.section.id for result in materialized}) == len(materialized)
        assert all("::chunk-" not in result.section.id for result in materialized)
        return materialized[:k]

    monkeypatch.setattr(reranker, "rerank_hybrid_results", inspecting_reranker)
    indexer.hybrid_search("anything", k=3)
    assert reranker_inputs


def test_rrf_fuses_backend_ranks_without_adding_incomparable_raw_scores():
    tiny_vector = make_section(source_id="a.md#tiny-vector", content="answer")
    huge_bm25 = make_section(source_id="z.md#huge-bm25", content="answer")

    fused = hybrid.merge_rrf(
        bm25_results=[(huge_bm25, 1_000_000_000.0)],
        vector_results=[(tiny_vector, 0.000_001)],
        k=2,
    )

    assert [result.section.id for result in fused] == [
        "a.md#tiny-vector",
        "z.md#huge-bm25",
    ]
    assert [result.score for result in fused] == pytest.approx(
        [1 / (hybrid.RRF_K + 1), 1 / (hybrid.RRF_K + 1)]
    )
    assert fused[0].vector_score == 0.000_001
    assert fused[1].bm25_score == 1_000_000_000.0


def test_compare_serializes_aggregated_child_hits_as_full_unique_parents(monkeypatch):
    parents, _dense_index = _install_child_hit_fixture(monkeypatch)
    monkeypatch.setattr(
        indexer,
        "search",
        lambda _query, k=3: [(parents[2], 500_000.0)][:k],
    )

    payload = retrieval.compare("full", k=3)

    assert set(payload) == {
        "query",
        "indexed",
        "message",
        "bm25",
        "vector",
        "hybrid",
        "reranked",
    }
    canonical_by_id = {parent.id: parent for parent in parents}
    for stage in ("bm25", "vector", "hybrid", "reranked"):
        sources = payload[stage]
        source_ids = [source["source"] for source in sources]
        assert len(source_ids) == len(set(source_ids))
        assert all("::chunk-" not in source_id for source_id in source_ids)
        assert all(source_id in canonical_by_id for source_id in source_ids)
        assert all(
            source["content"] == canonical_by_id[source["source"]].content
            for source in sources
        )

    assert [source["source"] for source in payload["vector"]] == [
        "z.md#best",
        "a.md#tied",
        "b.md#tied",
    ]
    assert [source["score"] for source in payload["vector"]] == pytest.approx(
        [0.99, 0.80, 0.80]
    )


def test_chat_serializes_cited_child_hit_as_one_full_canonical_parent(monkeypatch):
    parents, _dense_index = _install_child_hit_fixture(monkeypatch)
    monkeypatch.setattr(indexer, "search", lambda _query, k=3: [])

    class CitingParentLLM:
        def invoke(self, _messages):
            return type(
                "Message",
                (),
                {"content": "Grounded answer. [Source: z.md#best]"},
            )()

    monkeypatch.setattr(retrieval, "get_llm", lambda: CitingParentLLM())

    payload = retrieval.query("full", file_answer=False)

    assert payload["answer"] == "Grounded answer. [Source: z.md#best]"
    assert [source["source"] for source in payload["sources"]] == ["z.md#best"]
    assert len({source["source"] for source in payload["sources"]}) == len(
        payload["sources"]
    )
    assert "::chunk-" not in payload["sources"][0]["source"]
    assert payload["sources"][0]["content"] == parents[0].content
