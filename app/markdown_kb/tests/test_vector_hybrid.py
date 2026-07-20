import json
import os

import pytest

from app import hybrid, indexer
from app.main import load_persisted_index
from app.vector_index import ALGORITHM, LocalVectorIndex, MAX_VECTOR_INDEX_BYTES


def test_vector_index_builds_persists_and_searches_sample_sections(sample_docs_dir):
    files_count, sections_count = indexer.build_index(sample_docs_dir)

    assert files_count >= 10
    assert sections_count >= 30
    vector_path = indexer.vector_index_path()
    assert vector_path.exists()

    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "local-tfidf-cosine"
    assert payload["section_count"] == sections_count
    assert "refund_policy.md#refund-timeline" in {record["id"] for record in payload["records"]}
    assert payload["vocabulary"] == sorted(payload["vocabulary"])

    ranked = indexer.vector_search("How long do refunds take?", k=3)

    assert ranked
    assert ranked[0][0].id == "refund_policy.md#refund-timeline"
    assert ranked[0][1] > 0


def test_startup_loads_or_rebuilds_vector_index_from_persisted_sections(sample_docs_dir):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    expected = [
        (section.id, round(score, 6))
        for section, score in indexer.vector_search("Can I change my email address?", k=3)
    ]
    assert expected[0][0] == "account_help.md#change-email-address"

    indexer.sections = []
    indexer.section_vector_index = None
    load_persisted_index()
    loaded = [
        (section.id, round(score, 6))
        for section, score in indexer.vector_search("Can I change my email address?", k=3)
    ]
    assert loaded == expected

    vector_path.unlink()
    indexer.sections = []
    indexer.section_vector_index = None
    load_persisted_index()

    assert vector_path.exists()
    rebuilt = indexer.vector_search("Can I change my email address?", k=1)
    assert rebuilt[0][0].id == "account_help.md#change-email-address"


def test_stale_same_id_vector_index_rebuilds_from_current_sections(sample_docs_dir):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    stale_payload = {
        "algorithm": ALGORITHM,
        "version": 1,
        "section_count": len(indexer.sections),
        "vocabulary": [],
        "idf": {},
        "records": [
            {
                "id": section.id,
                "file": section.file,
                "heading": section.heading,
                "heading_path": section.heading_path,
                "token_count": 0,
                "norm": 0.0,
                "vector": {},
            }
            for section in indexer.sections
        ],
    }
    vector_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    indexer.section_vector_index = None

    indexer.load_or_rebuild_vector_index()
    rebuilt = indexer.vector_search("How long do refunds take?", k=1)

    assert rebuilt[0][0].id == "refund_policy.md#refund-timeline"
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    assert payload["vocabulary"]


def test_malformed_vector_index_rebuilds_instead_of_blocking_startup(sample_docs_dir):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    vector_path.write_text(
        json.dumps({"algorithm": ALGORITHM, "records": [{"missing": "id"}]}),
        encoding="utf-8",
    )
    indexer.section_vector_index = None

    indexer.load_or_rebuild_vector_index()
    rebuilt = indexer.vector_search("How long do refunds take?", k=1)

    assert rebuilt[0][0].id == "refund_policy.md#refund-timeline"


def test_bounded_huge_number_vector_corruption_rebuilds_instead_of_aborting(
    sample_docs_dir,
):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    payload["idf"][next(iter(payload["idf"]))] = int("9" * 400)
    vector_path.write_text(json.dumps(payload), encoding="utf-8")
    indexer.section_vector_index = None

    rebuilt = indexer.load_or_rebuild_vector_index()

    assert rebuilt is not None
    assert rebuilt.is_compatible(indexer.child_chunks)
    assert LocalVectorIndex.load(vector_path).is_compatible(indexer.child_chunks)


def test_deeply_nested_json_vector_corruption_rebuilds_instead_of_aborting(
    sample_docs_dir,
):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    nesting_depth = 2_000
    vector_path.write_text(
        "[" * nesting_depth + "0" + "]" * nesting_depth,
        encoding="utf-8",
    )
    indexer.section_vector_index = None

    rebuilt = indexer.load_or_rebuild_vector_index()

    assert rebuilt is not None
    assert rebuilt.is_compatible(indexer.child_chunks)
    assert LocalVectorIndex.load(vector_path).is_compatible(indexer.child_chunks)


@pytest.mark.parametrize("unsafe_case", ["symlink", "fifo", "oversized"])
def test_local_vector_loader_rejects_unsafe_or_oversized_artifacts(tmp_path, unsafe_case):
    artifact = tmp_path / "vector_index.json"
    if unsafe_case == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        artifact.symlink_to(outside)
    elif unsafe_case == "fifo":
        os.mkfifo(artifact)
    else:
        artifact.write_bytes(b" " * (MAX_VECTOR_INDEX_BYTES + 1))

    with pytest.raises((OSError, ValueError)):
        LocalVectorIndex.load(artifact)


def test_corrupt_same_signature_vector_index_rebuilds_from_sections(sample_docs_dir):
    indexer.build_index(sample_docs_dir)
    vector_path = indexer.vector_index_path()
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    expected_top = indexer.vector_search("How long do refunds take?", k=1)[0][0].id
    assert expected_top == "refund_policy.md#refund-timeline"

    for record in payload["records"]:
        if record["id"] == "account_help.md#change-email-address":
            record["vector"] = {"refunds": 999999.0}
            record["norm"] = 1.0
        else:
            record["vector"] = {}
            record["norm"] = 0.0
    vector_path.write_text(json.dumps(payload), encoding="utf-8")
    indexer.section_vector_index = None

    indexer.load_or_rebuild_vector_index()
    rebuilt = indexer.vector_search("How long do refunds take?", k=1)

    assert rebuilt[0][0].id == expected_top
    rebuilt_payload = json.loads(vector_path.read_text(encoding="utf-8"))
    account_record = next(
        record
        for record in rebuilt_payload["records"]
        if record["id"] == "account_help.md#change-email-address"
    )
    assert account_record["vector"] != {"refunds": 999999.0}


def test_rrf_hybrid_merges_ranks_and_keeps_debug_scores():
    refund = indexer.Section(
        id="refund_policy.md#refund-timeline",
        file="refund_policy.md",
        heading="Refund Timeline",
        heading_path=["Refund Policy", "Refund Timeline"],
        content="Approved refunds are processed within 5-7 business days.",
        tokens=["refund", "timeline", "approved", "refunds"],
    )
    shipping = indexer.Section(
        id="shipping_faq.md#standard-shipping",
        file="shipping_faq.md",
        heading="Standard Shipping",
        heading_path=["Shipping FAQ", "Standard Shipping"],
        content="Standard shipping usually takes 3-5 business days.",
        tokens=["shipping", "standard", "business", "days"],
    )
    account = indexer.Section(
        id="account_help.md#change-email-address",
        file="account_help.md",
        heading="Change Email Address",
        heading_path=["Account Help", "Change Email Address"],
        content="Customers can change their email address from Account Settings.",
        tokens=["account", "change", "email", "address"],
    )

    merged = hybrid.merge_rrf(
        bm25_results=[(refund, 5.0), (shipping, 3.0)],
        vector_results=[(shipping, 0.9), (account, 0.7)],
        k=3,
    )

    assert [result.section.id for result in merged] == [
        "shipping_faq.md#standard-shipping",
        "refund_policy.md#refund-timeline",
        "account_help.md#change-email-address",
    ]
    assert merged[0].bm25_rank == 2
    assert merged[0].vector_rank == 1
    assert merged[0].bm25_score == 3.0
    assert merged[0].vector_score == 0.9
    assert merged[0].score > merged[1].score
    assert merged[1].vector_rank is None
    assert merged[2].bm25_rank is None


def test_hybrid_search_returns_empty_for_out_of_scope_query(sample_docs_dir):
    indexer.build_index(sample_docs_dir)

    assert hybrid.merge_rrf([], [], k=3) == []
    assert indexer.hybrid_search("Which restaurants are nearby?", k=3) == []
