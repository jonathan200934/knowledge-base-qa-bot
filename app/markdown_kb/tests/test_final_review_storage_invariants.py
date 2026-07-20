import hashlib
import math
import sqlite3
import struct
from types import SimpleNamespace

import pytest

from app import embedding_cache
from app.safe_io import UnsafeFileError, read_regular_file


def _namespace():
    return embedding_cache.build_namespace(
        provider_kind="openai-compatible",
        embedding_model="test-embedding-model",
        base_url="https://api.example.test/v1",
        requested_dimension=3,
        resolved_dimension=3,
        vector_normalization="l2",
        metric="inner-product",
        distance_strategy="cosine",
    )


def _chunk():
    return SimpleNamespace(
        chunk_id="doc.md#section::chunk-0",
        source_id="doc.md#section",
        content_hash=hashlib.sha256(b"embedded content").hexdigest(),
    )


def test_read_regular_file_rejects_symlinked_ancestor_of_absolute_path(tmp_path):
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = real_directory / "document.md"
    target.write_bytes(b"trusted contents")

    # A normal absolute path must still be accepted; only actual symlink path
    # components are unsafe.
    snapshot = read_regular_file(target, max_bytes=1_024)
    assert snapshot.data == b"trusted contents"

    symlinked_directory = tmp_path / "linked"
    symlinked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(UnsafeFileError, match="symlinked path ancestors"):
        read_regular_file(symlinked_directory / target.name, max_bytes=1_024)


def test_finite_bit_flip_in_cached_vector_is_a_cache_miss(tmp_path):
    cache_path = tmp_path / "embedding_cache.sqlite3"
    namespace = _namespace()
    chunk = _chunk()
    expected = (0.25, 0.5, 0.75)

    embedding_cache.replace_cached_vectors(
        [chunk],
        [expected],
        namespace,
        cache_path,
    )
    assert embedding_cache.load_cached_vectors([chunk], namespace, cache_path) == {
        chunk.chunk_id: expected
    }

    with sqlite3.connect(cache_path) as connection:
        vector_blob, stored_sha256 = connection.execute(
            "SELECT vector, vector_sha256 FROM embedding_vectors"
        ).fetchone()
        assert hashlib.sha256(vector_blob).hexdigest() == stored_sha256

        corrupted_blob = bytearray(vector_blob)
        corrupted_blob[0] ^= 0x01
        corrupted_blob = bytes(corrupted_blob)
        assert all(math.isfinite(value) for value in struct.unpack("<3f", corrupted_blob))
        assert hashlib.sha256(corrupted_blob).hexdigest() != stored_sha256

        connection.execute(
            "UPDATE embedding_vectors SET vector = ?",
            (corrupted_blob,),
        )

    assert embedding_cache.load_cached_vectors([chunk], namespace, cache_path) == {}


def test_v1_cache_migration_preserves_but_does_not_trust_legacy_vectors(tmp_path):
    cache_path = tmp_path / "embedding_cache.sqlite3"
    namespace = _namespace()
    chunk = _chunk()
    legacy_blob = struct.pack("<3f", 0.25, 0.5, 0.75)

    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            """
            CREATE TABLE embedding_vectors (
                namespace_fingerprint TEXT,
                chunk_id TEXT,
                source_id TEXT,
                content_hash TEXT,
                dimension INTEGER,
                vector_encoding TEXT,
                vector BLOB,
                PRIMARY KEY (namespace_fingerprint, chunk_id, content_hash)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO embedding_vectors(
                namespace_fingerprint, chunk_id, source_id, content_hash,
                dimension, vector_encoding, vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace.fingerprint,
                chunk.chunk_id,
                chunk.source_id,
                chunk.content_hash,
                3,
                embedding_cache.VECTOR_ENCODING,
                legacy_blob,
            ),
        )
        connection.execute("PRAGMA user_version = 1")

    embedding_cache.initialize_cache(cache_path, namespace)

    with sqlite3.connect(cache_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(embedding_vectors)")
        }
        row_count, checksum = connection.execute(
            "SELECT COUNT(*), vector_sha256 FROM embedding_vectors"
        ).fetchone()
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert "vector_sha256" in columns
    assert schema_version == embedding_cache.CACHE_SCHEMA_VERSION
    assert row_count == 1
    assert checksum is None
    assert embedding_cache.load_cached_vectors([chunk], namespace, cache_path) == {}
