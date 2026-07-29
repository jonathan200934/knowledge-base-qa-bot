import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from app import indexer
from app.main import load_persisted_index


def test_parse_markdown_creates_heading_sections_with_deterministic_unique_source_ids(tmp_path):
    markdown = tmp_path / "faq.md"
    markdown.write_text(
        "# FAQ\n\n"
        "Intro text belongs to the document title, not a retrievable answer.\n\n"
        "## Duplicate\n\n"
        "First answer.\n\n"
        "## Duplicate\n\n"
        "Second answer.\n\n"
        "### Child Topic\n\n"
        "Nested answer.\n",
        encoding="utf-8",
    )

    sections = indexer.parse_markdown(markdown)

    assert [section.id for section in sections] == [
        "faq.md#duplicate",
        "faq.md#duplicate-2",
        "faq.md#child-topic",
    ]
    assert sections[0].heading_path == ["FAQ", "Duplicate"]
    assert sections[2].heading_path == ["FAQ", "Duplicate", "Child Topic"]
    assert "First answer." in sections[0].content
    assert {"faq", "duplicate", "first", "answer"}.issubset(sections[0].tokens)


def test_write_and_load_index_json_persists_sections_and_rebuilds_stats(index_path):
    indexer.sections = [
        indexer.Section(
            id="refund_policy.md#refund-timeline",
            file="refund_policy.md",
            heading="Refund Timeline",
            heading_path=["Refund Policy", "Refund Timeline"],
            content="Approved refunds are processed within 5-7 business days.",
            tokens=["refund", "policy", "refund", "timeline", "approved", "refunds", "processed"],
        ),
        indexer.Section(
            id="refund_policy.md#non-refundable-items",
            file="refund_policy.md",
            heading="Non-Refundable Items",
            heading_path=["Refund Policy", "Non-Refundable Items"],
            content="Digital gift cards are not refundable.",
            tokens=["refund", "policy", "non", "refundable", "items", "digital", "gift", "cards"],
        ),
    ]
    indexer.rebuild_stats()

    indexer.write_index_json()
    indexer.sections = []
    indexer.doc_freq.clear()
    indexer.avg_doc_len = 0.0
    indexer.files_indexed = 0

    counts = indexer.load_index_json()

    assert counts == (1, 2)
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["sections"][0]["id"] == "refund_policy.md#refund-timeline"
    assert payload["stats"]["files_indexed"] == 1
    assert payload["stats"]["sections_indexed"] == 2
    assert payload["stats"]["doc_freq"]["refund"] == 2
    assert indexer.doc_freq["refund"] == 2
    assert indexer.avg_doc_len > 0


def test_build_index_over_sample_docs_writes_expected_counts_and_sources(sample_docs_dir, index_path):
    files_count, sections_count = indexer.build_index(sample_docs_dir)

    assert files_count >= 10
    assert sections_count >= 30
    assert index_path.exists()
    source_ids = {section.id for section in indexer.sections}
    assert {
        "refund_policy.md#refund-timeline",
        "account_help.md#change-email-address",
        "shipping_faq.md#standard-shipping",
    }.issubset(source_ids)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["stats"]["files_indexed"] == files_count
    assert payload["stats"]["sections_indexed"] == sections_count


def test_startup_loads_persisted_index(index_path):
    payload = {
        "sections": [
            {
                "id": "account_help.md#change-email-address",
                "file": "account_help.md",
                "heading": "Change Email Address",
                "heading_path": ["Account Help", "Change Email Address"],
                "content": "Customers can change their email address from Account Settings.",
                "tokens": ["account", "help", "change", "email", "address", "customers"],
            }
        ],
        "stats": {},
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    load_persisted_index()

    assert [section.id for section in indexer.sections] == ["account_help.md#change-email-address"]
    assert indexer.files_indexed == 1


def test_load_index_json_falls_back_to_sqlite_when_json_is_corrupt(sample_docs_dir, index_path):
    files_count, sections_count = indexer.build_index(sample_docs_dir)
    expected_source_ids = [section.id for section in indexer.sections]
    index_path.write_text("{not valid json", encoding="utf-8")
    indexer.sections = []
    indexer.section_vector_index = None

    counts = indexer.load_index_json()

    assert counts == (files_count, sections_count)
    assert [section.id for section in indexer.sections] == expected_source_ids
    assert indexer.vector_search("How long do refunds take?", k=1)[0][0].id == "refund_policy.md#refund-timeline"


def test_load_index_json_prefers_sqlite_when_json_drifted_from_db(sample_docs_dir, index_path):
    files_count, sections_count = indexer.build_index(sample_docs_dir)
    drifted_payload = {
        "sections": [
            {
                "id": "drifted.md#stale",
                "file": "drifted.md",
                "heading": "Stale",
                "heading_path": ["Stale"],
                "content": "This section should not survive SQLite reconciliation.",
                "tokens": ["stale"],
            }
        ],
        "stats": {},
    }
    index_path.write_text(json.dumps(drifted_payload), encoding="utf-8")
    indexer.sections = []
    indexer.section_vector_index = None

    counts = indexer.load_index_json()

    assert counts == (files_count, sections_count)
    assert "drifted.md#stale" not in {section.id for section in indexer.sections}
    assert len(indexer.sections) == sections_count


@pytest.mark.parametrize("unsafe_case", ["symlink", "fifo", "oversized"])
def test_load_index_json_rejects_unsafe_or_oversized_artifacts(
    sample_docs_dir, index_path, unsafe_case
):
    expected = indexer.build_index(sample_docs_dir)
    index_path.unlink()
    if unsafe_case == "symlink":
        outside = index_path.parent / "outside.json"
        outside.write_text('{"sections": []}', encoding="utf-8")
        index_path.symlink_to(outside)
    elif unsafe_case == "fifo":
        os.mkfifo(index_path)
    else:
        index_path.write_bytes(b" " * (indexer.MAX_INDEX_JSON_BYTES + 1))
    indexer.sections = []
    indexer.section_vector_index = None

    assert indexer.load_index_json() == expected
    assert len(indexer.sections) == expected[1]


def test_build_index_ignores_symlinked_and_non_regular_markdown(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "safe.md").write_text(
        "# Safe\n\n## Answer\n\nTrusted content.\n", encoding="utf-8"
    )

    outside = tmp_path / "outside.md"
    outside.write_text(
        "# Secret\n\n## Outside\n\nMust not be indexed.\n", encoding="utf-8"
    )
    (docs_dir / "linked.md").symlink_to(outside)
    fifo = docs_dir / "pipe.md"
    os.mkfifo(fifo)

    original_read_bytes = Path.read_bytes

    def reject_unsafe_read(path):
        if path in {docs_dir / "linked.md", fifo}:
            raise AssertionError("unsafe Markdown entries must not be opened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unsafe_read)

    assert indexer.build_index(docs_dir) == (1, 1)
    assert [section.id for section in indexer.sections] == ["safe.md#answer"]


def test_build_index_hashes_and_parses_the_same_opened_snapshot(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    markdown = docs_dir / "race.md"
    original_text = "# FAQ\n\n## Original\n\nOriginal answer.\n"
    changed_text = "# FAQ\n\n## Changed\n\nChanged answer.\n"
    markdown.write_text(original_text, encoding="utf-8")

    original_fingerprint = indexer.file_fingerprint

    def swap_after_fingerprint(path):
        fingerprint = original_fingerprint(path)
        path.write_text(changed_text, encoding="utf-8")
        return fingerprint

    monkeypatch.setattr(indexer, "file_fingerprint", swap_after_fingerprint)

    indexer.build_index(docs_dir)

    section = indexer.sections[0]
    indexed_text = original_text if section.heading == "Original" else changed_text
    expected_hash = hashlib.sha256(indexed_text.encode("utf-8")).hexdigest()
    with sqlite3.connect(indexer.INDEX_DB_PATH) as conn:
        stored_hash = conn.execute(
            "SELECT content_hash FROM files WHERE path = ?", ("race.md",)
        ).fetchone()[0]
    assert stored_hash == expected_hash
