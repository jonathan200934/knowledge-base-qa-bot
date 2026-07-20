import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import indexer, retrieval
from app.routes import router


SOURCE_FIELDS = {
    "source",
    "heading",
    "score",
    "content",
    "bm25_rank",
    "vector_rank",
    "bm25_score",
    "vector_score",
    "rrf_score",
    "rerank_rank",
    "rerank_score",
}


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, answer: str = "Grounded answer from fake LLM."):
        self.answer = answer
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeMessage(self.answer)


def make_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_sample_questions_rank_expected_top_sources(sample_docs_dir):
    files_count, sections_count = indexer.build_index(sample_docs_dir)
    assert files_count >= 10
    assert sections_count >= 30

    cases = {
        "How long do refunds take?": "refund_policy.md#refund-timeline",
        "Can I change my email address?": "account_help.md#change-email-address",
        "How long does standard shipping take?": "shipping_faq.md#standard-shipping",
    }

    for question, expected_source in cases.items():
        ranked = indexer.search(question, k=3)
        assert ranked, f"Expected at least one result for {question!r}"
        assert ranked[0][0].id == expected_source

        hybrid_ranked = indexer.hybrid_search(question, k=3)
        assert hybrid_ranked, f"Expected at least one hybrid result for {question!r}"
        assert hybrid_ranked[0].section.id == expected_source
        assert hybrid_ranked[0].score > 0
        assert hybrid_ranked[0].bm25_rank == 1
        assert hybrid_ranked[0].vector_rank is not None


def test_out_of_scope_query_has_no_usable_retrieval_result(sample_docs_dir):
    files_count, sections_count = indexer.build_index(sample_docs_dir)
    assert files_count >= 10
    assert sections_count >= 30

    assert indexer.search("Which restaurants are nearby?", k=3) == []
    assert indexer.hybrid_search("Which restaurants are nearby?", k=3) == []


def test_build_prompt_is_one_structured_json_object():
    section = indexer.Section(
        id="refund_policy.md#refund-timeline",
        file="refund_policy.md",
        heading="Refund Timeline",
        heading_path=["Refund Policy", "Refund Timeline"],
        content="Approved refunds are processed within 5-7 business days.",
        tokens=["refund", "timeline", "approved", "refunds"],
    )

    prompt = retrieval.build_prompt("How long do refunds take?", [(section, 2.5)])

    payload = json.loads(prompt)
    assert payload["question"] == "How long do refunds take?"
    assert payload["selected_sections"] == [
        {
            "source": "refund_policy.md#refund-timeline",
            "heading": "Refund Policy > Refund Timeline",
            "score": 2.5,
            "text": "Approved refunds are processed within 5-7 business days.",
        }
    ]


def test_chat_before_index_returns_unindexed_message_without_sources():
    client = make_test_client()

    response = client.post(
        "/chat", json={"query": "How long do refunds take?", "file_answer": False}
    )

    assert response.status_code == 200
    payload = response.json()
    assert "knowledge base has not been indexed" in payload["answer"].lower()
    assert payload["sources"] == []


def test_chat_after_index_uses_fake_llm_and_returns_source_metadata(monkeypatch, sample_docs_dir):
    fake_llm = FakeLLM(
        "Refunds take 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setenv("INDEX_API_KEY", "test-index-key")
    client = make_test_client()

    index_response = client.post("/index", headers={"X-Index-Key": "test-index-key"})
    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload["files_indexed"] >= 10
    assert index_payload["sections_indexed"] >= 30

    response = client.post(
        "/chat",
        json={"query": "How long do refunds take?", "file_answer": True},
        headers={"X-Index-Key": "test-index-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == (
        "Refunds take 5-7 business days. "
        "[Source: refund_policy.md#refund-timeline]"
    )
    assert payload["sources"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["sources"][0]["heading"] == "Refund Policy > Refund Timeline"
    assert payload["sources"][0]["score"] > 0
    assert payload["sources"][0]["bm25_rank"] == 1
    assert payload["sources"][0]["vector_rank"] is not None
    assert payload["sources"][0]["bm25_score"] > 0
    assert payload["sources"][0]["vector_score"] > 0
    assert "5-7 business days" in payload["sources"][0]["content"]
    assert fake_llm.messages is not None
    human_message = fake_llm.messages[1]
    selected_sections = json.loads(human_message.content)["selected_sections"]
    assert selected_sections[0]["source"] == "refund_policy.md#refund-timeline"
    answer_file = Path(payload["answer_file"])
    assert answer_file.exists()
    answer_card = json.loads(answer_file.read_text(encoding="utf-8"))
    assert answer_card["question"] == "How long do refunds take?"
    assert answer_card["sources"][0]["source"] == "refund_policy.md#refund-timeline"


def test_chat_fails_closed_if_llm_fails(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(retrieval, "get_llm", lambda: FailingLLM())
    monkeypatch.setenv("INDEX_API_KEY", "test-index-key")
    client = make_test_client()

    index_response = client.post("/index", headers={"X-Index-Key": "test-index-key"})
    assert index_response.status_code == 200

    response = client.post(
        "/chat", json={"query": "How long do refunds take?", "file_answer": False}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I cannot confirm from the knowledge base.",
        "sources": [],
    }


def test_compare_before_index_returns_unindexed_empty_results():
    client = make_test_client()

    response = client.post("/compare", json={"query": "How long do refunds take?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "How long do refunds take?"
    assert payload["indexed"] is False
    assert "not been indexed" in payload["message"].lower()
    assert payload["bm25"] == []
    assert payload["vector"] == []
    assert payload["hybrid"] == []
    assert payload["reranked"] == []


def test_compare_rejects_invalid_k_values():
    client = make_test_client()

    for invalid_k in (0, -1, 21):
        response = client.post("/compare", json={"query": "How long do refunds take?", "k": invalid_k})
        assert response.status_code == 422


def test_compare_after_index_returns_bm25_vector_and_hybrid_debug_results(monkeypatch):
    monkeypatch.setenv("INDEX_API_KEY", "test-index-key")
    client = make_test_client()
    index_response = client.post("/index", headers={"X-Index-Key": "test-index-key"})
    assert index_response.status_code == 200

    response = client.post("/compare", json={"query": "How long do refunds take?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "How long do refunds take?"
    assert payload["indexed"] is True
    assert payload["message"] is None
    assert payload["bm25"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["vector"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["hybrid"][0]["source"] == "refund_policy.md#refund-timeline"
    assert payload["hybrid"][0]["score"] > 0
    assert payload["hybrid"][0]["bm25_rank"] == 1
    assert payload["hybrid"][0]["vector_rank"] == 1
    assert payload["hybrid"][0]["bm25_score"] > 0
    assert payload["hybrid"][0]["vector_score"] > 0


def test_out_of_scope_chat_after_index_returns_exact_fallback_without_calling_llm(monkeypatch):
    class ExplodingLLM:
        def invoke(self, messages):  # pragma: no cover - should never be called
            raise AssertionError("LLM should not be called for unsupported questions")

    monkeypatch.setattr(retrieval, "get_llm", lambda: ExplodingLLM())
    monkeypatch.setenv("INDEX_API_KEY", "test-index-key")
    client = make_test_client()
    index_response = client.post("/index", headers={"X-Index-Key": "test-index-key"})
    assert index_response.json()["files_indexed"] >= 10

    response = client.post(
        "/chat", json={"query": "Which restaurants are nearby?", "file_answer": False}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I cannot confirm from the knowledge base.",
        "sources": [],
    }


def test_chat_uses_lower_ranked_answerable_hybrid_candidate(monkeypatch):
    weak = indexer.Section(
        id="a.md#weak-vector-match",
        file="a.md",
        heading="Weak Vector Match",
        heading_path=["Weak Vector Match"],
        content="This section mentions refunds but not the timeline.",
        tokens=["refunds"],
    )
    answerable = indexer.Section(
        id="z.md#refund-timeline",
        file="z.md",
        heading="Refund Timeline",
        heading_path=["Refund Timeline"],
        content="Approved refunds are processed within 5-7 business days.",
        tokens=["approved", "refunds", "processed", "5", "7", "business", "days"],
    )
    fake_llm = FakeLLM(
        "Refunds take 5-7 business days. [Source: z.md#refund-timeline]"
    )
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(indexer, "sections", [weak, answerable])
    monkeypatch.setattr(
        indexer,
        "hybrid_search",
        lambda question, k=3: [
            retrieval.HybridResult(
                section=weak,
                score=0.02,
                vector_rank=1,
                vector_score=0.01,
            ),
            retrieval.HybridResult(
                section=answerable,
                score=0.016,
                bm25_rank=1,
                bm25_score=retrieval.MIN_RETRIEVAL_SCORE,
            ),
        ],
    )

    payload = retrieval.query("How long do refunds take?", file_answer=False)

    assert payload["answer"] == (
        "Refunds take 5-7 business days. [Source: z.md#refund-timeline]"
    )
    assert [source["source"] for source in payload["sources"]] == ["z.md#refund-timeline"]
    assert fake_llm.messages is not None
    human_message = fake_llm.messages[1]
    selected_sources = [
        section["source"]
        for section in json.loads(human_message.content)["selected_sections"]
    ]
    assert selected_sources == ["z.md#refund-timeline"]


def test_chat_returns_cannot_confirm_when_hybrid_hits_are_below_answer_threshold(monkeypatch):
    class ExplodingLLM:
        def invoke(self, messages):  # pragma: no cover - should never be called
            raise AssertionError("LLM should not be called for weak hybrid hits")

    weak = indexer.Section(
        id="a.md#weak-match",
        file="a.md",
        heading="Weak Match",
        heading_path=["Weak Match"],
        content="This section only has a weak lexical overlap.",
        tokens=["weak", "lexical", "overlap"],
    )
    monkeypatch.setattr(retrieval, "get_llm", lambda: ExplodingLLM())
    monkeypatch.setattr(indexer, "sections", [weak])
    monkeypatch.setattr(
        indexer,
        "hybrid_search",
        lambda question, k=3: [
            retrieval.HybridResult(
                section=weak,
                score=0.032,
                bm25_rank=1,
                vector_rank=1,
                bm25_score=retrieval.MIN_RETRIEVAL_SCORE - 0.001,
                vector_score=retrieval.MIN_VECTOR_SCORE - 0.001,
            )
        ],
    )

    payload = retrieval.query("Weakly related question", file_answer=False)

    assert payload == {
        "answer": "I cannot confirm from the knowledge base.",
        "sources": [],
    }


def _canonical_api_section(source_id: str, label: str) -> indexer.Section:
    heading = f"{label} Parent"
    content = (
        f"FULL CANONICAL {label.upper()} PARENT START. "
        + (f"{label} parent detail. " * 30)
        + f"FULL CANONICAL {label.upper()} PARENT END."
    )
    return indexer.Section(
        id=source_id,
        file=source_id.partition("#")[0],
        heading=heading,
        heading_path=["API Contract", heading],
        content=content,
        tokens=indexer.tokenize(content),
    )


def test_chat_api_exposes_full_unique_canonical_parent_sources_and_schema(monkeypatch):
    first = _canonical_api_section("a.md#first-parent", "first")
    second = _canonical_api_section("b.md#second-parent", "second")
    indexer.sections = [first, second]
    ranked = [
        retrieval.HybridResult(
            section=first,
            score=1.75,
            bm25_rank=1,
            vector_rank=2,
            bm25_score=4.5,
            vector_score=0.82,
            rrf_score=0.032,
            rerank_rank=1,
            rerank_score=1.75,
        ),
        retrieval.HybridResult(
            section=second,
            score=1.25,
            bm25_rank=2,
            vector_rank=1,
            bm25_score=3.5,
            vector_score=0.91,
            rrf_score=0.031,
            rerank_rank=2,
            rerank_score=1.25,
        ),
    ]
    fake_llm = FakeLLM(
        "Canonical answer. [Source: a.md#first-parent] [Source: b.md#second-parent]"
    )
    monkeypatch.setattr(indexer, "hybrid_search", lambda _query, k=3: ranked[:k])
    monkeypatch.setattr(retrieval, "get_llm", lambda: fake_llm)

    response = make_test_client().post(
        "/chat",
        json={"query": "parent detail", "file_answer": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"answer", "sources"}
    assert payload["answer"] == fake_llm.answer
    assert [source["source"] for source in payload["sources"]] == [
        first.id,
        second.id,
    ]
    assert len({source["source"] for source in payload["sources"]}) == len(
        payload["sources"]
    )
    assert all("::chunk-" not in source["source"] for source in payload["sources"])
    assert all(set(source) == SOURCE_FIELDS for source in payload["sources"])
    assert [source["content"] for source in payload["sources"]] == [
        first.content,
        second.content,
    ]
    assert fake_llm.messages is not None
    selected_sections = json.loads(fake_llm.messages[1].content)["selected_sections"]
    assert [section["source"] for section in selected_sections] == [first.id, second.id]
    assert [section["text"] for section in selected_sections] == [
        first.content,
        second.content,
    ]


def test_compare_api_preserves_parent_only_stage_order_schema_and_pure_vector(monkeypatch):
    first = _canonical_api_section("a.md#first-parent", "first")
    second = _canonical_api_section("b.md#second-parent", "second")
    third = _canonical_api_section("c.md#third-parent", "third")
    indexer.sections = [first, second, third]

    calls: list[tuple[str, str, int]] = []

    def bm25_search(query, k=3):
        calls.append(("bm25", query, k))
        return [(first, 9.0), (second, 8.0)][:k]

    def pure_vector_search(query, k=3):
        calls.append(("vector", query, k))
        return [(third, 0.91), (first, 0.82)][:k]

    def hybrid_search(query, k=3):
        calls.append(("hybrid", query, k))
        return [
            retrieval.HybridResult(
                section=second,
                score=0.032,
                bm25_rank=2,
                vector_rank=1,
                bm25_score=8.0,
                vector_score=0.91,
                rrf_score=0.032,
            ),
            retrieval.HybridResult(
                section=third,
                score=0.031,
                bm25_rank=None,
                vector_rank=2,
                bm25_score=None,
                vector_score=0.82,
                rrf_score=0.031,
            ),
        ][:k]

    def reranked_search(query, k=3):
        calls.append(("reranked", query, k))
        return [
            retrieval.HybridResult(
                section=third,
                score=2.5,
                bm25_rank=None,
                vector_rank=2,
                bm25_score=None,
                vector_score=0.82,
                rrf_score=0.031,
                rerank_rank=1,
                rerank_score=2.5,
            ),
            retrieval.HybridResult(
                section=second,
                score=2.0,
                bm25_rank=2,
                vector_rank=1,
                bm25_score=8.0,
                vector_score=0.91,
                rrf_score=0.032,
                rerank_rank=2,
                rerank_score=2.0,
            ),
        ][:k]

    monkeypatch.setattr(indexer, "search", bm25_search)
    monkeypatch.setattr(indexer, "vector_search", pure_vector_search)
    monkeypatch.setattr(indexer, "hybrid_rrf_search", hybrid_search)
    monkeypatch.setattr(indexer, "hybrid_search", reranked_search)
    client = make_test_client()

    first_response = client.post("/compare", json={"query": "parent detail", "k": 2})
    second_response = client.post("/compare", json={"query": "parent detail", "k": 2})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    payload = first_response.json()
    assert payload == second_response.json()
    assert set(payload) == {
        "query",
        "indexed",
        "message",
        "bm25",
        "vector",
        "hybrid",
        "reranked",
    }
    assert payload["query"] == "parent detail"
    assert payload["indexed"] is True
    assert payload["message"] is None
    expected_orders = {
        "bm25": [first.id, second.id],
        "vector": [third.id, first.id],
        "hybrid": [second.id, third.id],
        "reranked": [third.id, second.id],
    }
    canonical_by_id = {section.id: section for section in indexer.sections}
    for stage, expected_order in expected_orders.items():
        results = payload[stage]
        source_ids = [result["source"] for result in results]
        assert source_ids == expected_order
        assert len(source_ids) == len(set(source_ids))
        assert all("::chunk-" not in source_id for source_id in source_ids)
        assert all(set(result) == SOURCE_FIELDS for result in results)
        assert all(
            result["content"] == canonical_by_id[result["source"]].content
            for result in results
        )

    # The vector list is wired directly to the pure vector backend, not to
    # BM25, RRF fusion, or reranking.
    assert [result["score"] for result in payload["vector"]] == [0.91, 0.82]
    optional_fields = SOURCE_FIELDS - {"source", "heading", "score", "content"}
    assert all(payload["vector"][0][field] is None for field in optional_fields)
    assert payload["hybrid"][0]["rrf_score"] == 0.032
    assert payload["reranked"][0]["rerank_rank"] == 1
    assert payload["reranked"][0]["rerank_score"] == 2.5
    assert calls == [
        (stage, "parent detail", 2)
        for _request in range(2)
        for stage in ("bm25", "vector", "hybrid", "reranked")
    ]
