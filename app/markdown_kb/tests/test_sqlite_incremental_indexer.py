import sqlite3

from app import indexer


def write_doc(path, title, sections):
    body = [f"# {title}\n"]
    for heading, content in sections:
        body.append(f"## {heading}\n\n{content}\n")
    path.write_text("\n".join(body), encoding="utf-8")


def section_count(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]


def indexed_at_by_file(db_path):
    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute("SELECT path, indexed_at FROM files"))


def test_build_index_creates_sqlite_fts_tables_and_searches_with_bm25(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    write_doc(
        docs_dir / "shipping_faq.md",
        "Shipping FAQ",
        [("Standard Shipping", "Standard shipping usually takes 3-5 business days.")],
    )
    write_doc(
        docs_dir / "refund_policy.md",
        "Refund Policy",
        [("Refund Timeline", "Approved refunds are processed within 5-7 business days.")],
    )

    files_count, sections_count = indexer.build_index(docs_dir)

    assert (files_count, sections_count) == (2, 2)
    assert indexer.INDEX_DB_PATH.exists()
    with sqlite3.connect(indexer.INDEX_DB_PATH) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            )
        }
        assert {"files", "sections", "sections_fts"}.issubset(tables)
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM sections_fts").fetchone()[0] == 2

    ranked = indexer.search("How long does standard shipping take?", k=1)

    assert ranked[0][0].id == "shipping_faq.md#standard-shipping"
    assert ranked[0][1] > 0


def test_build_index_is_incremental_for_unchanged_changed_and_deleted_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shipping = docs_dir / "shipping_faq.md"
    refund = docs_dir / "refund_policy.md"
    write_doc(shipping, "Shipping FAQ", [("Standard Shipping", "Standard shipping takes 3-5 business days.")])
    write_doc(refund, "Refund Policy", [("Refund Timeline", "Refunds take 5-7 business days.")])

    indexer.build_index(docs_dir)
    first_indexed_at = indexed_at_by_file(indexer.INDEX_DB_PATH)

    indexer.build_index(docs_dir)

    assert indexer.last_index_stats["changed_files"] == 0
    assert indexer.last_index_stats["skipped_files"] == 2
    assert indexed_at_by_file(indexer.INDEX_DB_PATH) == first_indexed_at

    write_doc(shipping, "Shipping FAQ", [("Standard Shipping", "Standard shipping now takes 4-6 business days.")])

    indexer.build_index(docs_dir)

    assert indexer.last_index_stats["changed_files"] == 1
    assert indexer.last_index_stats["skipped_files"] == 1
    assert section_count(indexer.INDEX_DB_PATH) == 2
    assert indexer.search("4-6 business days shipping", k=1)[0][0].id == "shipping_faq.md#standard-shipping"

    refund.unlink()

    files_count, sections_count = indexer.build_index(docs_dir)

    assert (files_count, sections_count) == (1, 1)
    assert indexer.last_index_stats["deleted_files"] == 1
    assert indexer.search("refund timeline", k=3) == []
