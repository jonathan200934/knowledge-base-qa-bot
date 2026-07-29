"""Safe deterministic primitives for the document-embedding cache.

This module owns the cache format, compatibility namespace, validated vector
reads, and the atomic active-generation switch used by index activation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import sqlite3
import struct
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .chunking import ChunkingPolicy, DEFAULT_CHUNKING_POLICY


CACHE_FILENAME = "embedding_cache.sqlite3"
CACHE_PATH = Path(__file__).resolve().parents[1] / ".kb" / CACHE_FILENAME
CACHE_SCHEMA_VERSION = 2
NAMESPACE_VERSION = 1
VECTOR_ENCODING = "float32-le-v1"
SQLITE_TIMEOUT_SECONDS = 5

MAX_NAMESPACE_JSON_BYTES = 8_192
MAX_NAMESPACE_TEXT_CHARS = 256
MAX_BASE_URL_CHARS = 2_048
MAX_SEPARATOR_COUNT = 16
MAX_SEPARATOR_CHARS = 128
MAX_CHUNK_SIZE = 1_000_000
MAX_ID_CHARS = 2_048
MAX_EMBEDDING_DIMENSION = 65_536

_DEFAULT_PORTS = {"http": 80, "https": 443}
_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*$")


class EmbeddingCacheError(ValueError):
    """Base error for invalid cache configuration or storage."""


class NamespaceValidationError(EmbeddingCacheError):
    """Raised when vector-affecting namespace metadata is invalid."""


class CacheSchemaError(EmbeddingCacheError):
    """Raised when the on-disk cache schema is incompatible."""


class CacheActivationError(EmbeddingCacheError):
    """Raised when a candidate no longer matches the active corpus."""


def _fail(message: str) -> NamespaceValidationError:
    return NamespaceValidationError(message)


def _bounded_text(
    value: object,
    field: str,
    *,
    maximum: int = MAX_NAMESPACE_TEXT_CHARS,
    allow_empty: bool = False,
    allow_line_breaks: bool = False,
) -> str:
    if not isinstance(value, str):
        raise _fail(f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not allow_empty and not normalized:
        raise _fail(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise _fail(f"{field} is too long")
    if "\x00" in normalized:
        raise _fail(f"{field} contains invalid characters")
    if not allow_line_breaks and any(character in "\r\n" for character in normalized):
        raise _fail(f"{field} contains invalid characters")
    return normalized


def _token(value: object, field: str) -> str:
    normalized = _bounded_text(value, field).lower()
    if not _TOKEN_PATTERN.fullmatch(normalized):
        raise _fail(f"{field} must be a bounded identifier")
    return normalized


def _positive_dimension(value: object, field: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(f"{field} must be an integer")
    if not 1 <= value <= MAX_EMBEDDING_DIMENSION:
        raise _fail(f"{field} is outside the supported range")
    return value


def _normalize_url_path(path: str) -> str:
    """Normalize duplicate separators and dot segments without decoding data."""

    normalized = posixpath.normpath("/" + path.lstrip("/"))
    if normalized in ("/", "/."):
        return ""
    # posixpath preserves exactly two leading slashes; a compatible HTTP base
    # URL always has one authority/path separator.
    return "/" + normalized.lstrip("/").rstrip("/")


def normalize_base_url(value: object) -> str:
    """Return a credential-free canonical HTTP(S) API base URL.

    URL user information, query parameters, and fragments are intentionally
    discarded before namespace serialization and hashing. They can contain API
    credentials or user-specific routing data and are not safe compatibility
    metadata.
    """

    raw = _bounded_text(value, "base_url", maximum=MAX_BASE_URL_CHARS)
    if any(ord(character) < 32 or character.isspace() for character in raw):
        raise _fail("base_url contains invalid characters")

    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise _fail("base_url is invalid") from None

    if scheme not in _DEFAULT_PORTS or not hostname:
        raise _fail("base_url must be an absolute HTTP(S) URL")

    try:
        canonical_host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise _fail("base_url host is invalid") from None
    if not canonical_host or len(canonical_host) > 253:
        raise _fail("base_url host is invalid")

    # urlsplit removes IPv6 brackets in hostname; restore them for urlunsplit.
    rendered_host = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        rendered_host = f"{rendered_host}:{port}"

    sanitized = SplitResult(
        scheme=scheme,
        netloc=rendered_host,
        path=_normalize_url_path(parsed.path),
        query="",
        fragment="",
    )
    canonical = urlunsplit(sanitized)
    if len(canonical) > MAX_BASE_URL_CHARS:
        raise _fail("base_url is too long")
    return canonical


def _require_plain_mapping(
    value: object, field: str, expected_fields: frozenset[str]
) -> dict[str, object]:
    if type(value) is not dict:
        raise _fail(f"{field} must be a JSON object")
    fields = frozenset(value)
    if fields != expected_fields or any(not isinstance(key, str) for key in value):
        raise _fail(f"{field} contains unknown or missing fields")
    return value


@dataclass(frozen=True)
class ChildPolicyNamespace:
    """Complete bounded child-text policy participating in compatibility."""

    chunk_size: int
    chunk_overlap: int
    separators: tuple[str, ...]
    splitter_version: str
    heading_prefix_behavior: str
    heading_prefix_version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or not 1 <= self.chunk_size <= MAX_CHUNK_SIZE
        ):
            raise _fail("chunk_size is outside the supported range")
        if (
            not isinstance(self.chunk_overlap, int)
            or isinstance(self.chunk_overlap, bool)
            or not 0 <= self.chunk_overlap < self.chunk_size
        ):
            raise _fail("chunk_overlap must be smaller than chunk_size")
        if not isinstance(self.separators, tuple) or not (
            1 <= len(self.separators) <= MAX_SEPARATOR_COUNT
        ):
            raise _fail("separators must be a bounded tuple")

        separators: list[str] = []
        for separator in self.separators:
            separators.append(
                _bounded_text(
                    separator,
                    "separator",
                    maximum=MAX_SEPARATOR_CHARS,
                    allow_empty=True,
                    allow_line_breaks=True,
                )
            )
        object.__setattr__(self, "separators", tuple(separators))
        object.__setattr__(
            self,
            "splitter_version",
            _token(self.splitter_version, "splitter_version"),
        )
        object.__setattr__(
            self,
            "heading_prefix_behavior",
            _token(self.heading_prefix_behavior, "heading_prefix_behavior"),
        )
        object.__setattr__(
            self,
            "heading_prefix_version",
            _token(self.heading_prefix_version, "heading_prefix_version"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
            "splitter_version": self.splitter_version,
            "heading_prefix_behavior": self.heading_prefix_behavior,
            "heading_prefix_version": self.heading_prefix_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ChildPolicyNamespace:
        payload = _require_plain_mapping(
            value,
            "child_policy",
            frozenset(
                {
                    "chunk_size",
                    "chunk_overlap",
                    "separators",
                    "splitter_version",
                    "heading_prefix_behavior",
                    "heading_prefix_version",
                }
            ),
        )
        raw_separators = payload["separators"]
        if type(raw_separators) is not list:
            raise _fail("child_policy.separators must be a JSON array")
        if len(raw_separators) > MAX_SEPARATOR_COUNT:
            raise _fail("child_policy.separators is too long")
        return cls(
            chunk_size=cast(int, payload["chunk_size"]),
            chunk_overlap=cast(int, payload["chunk_overlap"]),
            separators=tuple(raw_separators),
            splitter_version=cast(str, payload["splitter_version"]),
            heading_prefix_behavior=cast(str, payload["heading_prefix_behavior"]),
            heading_prefix_version=cast(str, payload["heading_prefix_version"]),
        )


@dataclass(frozen=True)
class EmbeddingNamespace:
    """Canonical vector-compatibility namespace with no credential fields."""

    provider_kind: str
    embedding_model: str
    base_url: str
    requested_dimension: int | None
    resolved_dimension: int
    vector_normalization: str
    metric: str
    distance_strategy: str
    child_policy: ChildPolicyNamespace
    namespace_version: int = NAMESPACE_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace_version, int)
            or isinstance(self.namespace_version, bool)
            or self.namespace_version != NAMESPACE_VERSION
        ):
            raise _fail("namespace_version is unsupported")
        object.__setattr__(self, "provider_kind", _token(self.provider_kind, "provider_kind"))
        object.__setattr__(
            self,
            "embedding_model",
            _bounded_text(self.embedding_model, "embedding_model"),
        )
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(
            self,
            "requested_dimension",
            _positive_dimension(
                self.requested_dimension,
                "requested_dimension",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "resolved_dimension",
            _positive_dimension(self.resolved_dimension, "resolved_dimension"),
        )
        object.__setattr__(
            self,
            "vector_normalization",
            _token(self.vector_normalization, "vector_normalization"),
        )
        object.__setattr__(self, "metric", _token(self.metric, "metric"))
        object.__setattr__(
            self,
            "distance_strategy",
            _token(self.distance_strategy, "distance_strategy"),
        )
        if not isinstance(self.child_policy, ChildPolicyNamespace):
            raise _fail("child_policy must be a ChildPolicyNamespace")
        # Enforce the final serialized-size bound for direct construction too.
        if len(self.canonical_json.encode("utf-8")) > MAX_NAMESPACE_JSON_BYTES:
            raise _fail("namespace JSON is too large")

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace_version": self.namespace_version,
            "provider": {
                "kind": self.provider_kind,
                "embedding_model": self.embedding_model,
                "base_url": self.base_url,
                "requested_dimension": self.requested_dimension,
                "resolved_dimension": self.resolved_dimension,
            },
            "vector": {
                "normalization": self.vector_normalization,
                "metric": self.metric,
                "distance_strategy": self.distance_strategy,
            },
            "child_policy": self.child_policy.to_dict(),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> EmbeddingNamespace:
        payload = _require_plain_mapping(
            value,
            "namespace",
            frozenset(
                {
                    "namespace_version",
                    "provider",
                    "vector",
                    "child_policy",
                }
            ),
        )
        provider = _require_plain_mapping(
            payload["provider"],
            "provider",
            frozenset(
                {
                    "kind",
                    "embedding_model",
                    "base_url",
                    "requested_dimension",
                    "resolved_dimension",
                }
            ),
        )
        vector = _require_plain_mapping(
            payload["vector"],
            "vector",
            frozenset({"normalization", "metric", "distance_strategy"}),
        )
        return cls(
            namespace_version=cast(int, payload["namespace_version"]),
            provider_kind=cast(str, provider["kind"]),
            embedding_model=cast(str, provider["embedding_model"]),
            base_url=cast(str, provider["base_url"]),
            requested_dimension=cast(int | None, provider["requested_dimension"]),
            resolved_dimension=cast(int, provider["resolved_dimension"]),
            vector_normalization=cast(str, vector["normalization"]),
            metric=cast(str, vector["metric"]),
            distance_strategy=cast(str, vector["distance_strategy"]),
            child_policy=ChildPolicyNamespace.from_mapping(payload["child_policy"]),
        )


def build_namespace(
    *,
    provider_kind: str,
    embedding_model: str,
    base_url: str,
    requested_dimension: int | None,
    resolved_dimension: int,
    vector_normalization: str,
    metric: str,
    distance_strategy: str,
    chunking_policy: ChunkingPolicy | ChildPolicyNamespace = DEFAULT_CHUNKING_POLICY,
    heading_prefix_behavior: str = "heading-path-then-heading",
) -> EmbeddingNamespace:
    """Build a canonical namespace from the app's versioned chunk policy."""

    if isinstance(chunking_policy, ChildPolicyNamespace):
        child_policy = chunking_policy
        if child_policy.heading_prefix_behavior != _token(
            heading_prefix_behavior, "heading_prefix_behavior"
        ):
            raise _fail("heading_prefix_behavior conflicts with child_policy")
    elif isinstance(chunking_policy, ChunkingPolicy):
        child_policy = ChildPolicyNamespace(
            chunk_size=chunking_policy.chunk_size,
            chunk_overlap=chunking_policy.chunk_overlap,
            separators=chunking_policy.separators,
            splitter_version=chunking_policy.splitter_version,
            heading_prefix_behavior=heading_prefix_behavior,
            heading_prefix_version=chunking_policy.heading_prefix_version,
        )
    else:
        raise _fail("chunking_policy must be a bounded versioned policy")

    return EmbeddingNamespace(
        provider_kind=provider_kind,
        embedding_model=embedding_model,
        base_url=base_url,
        requested_dimension=requested_dimension,
        resolved_dimension=resolved_dimension,
        vector_normalization=vector_normalization,
        metric=metric,
        distance_strategy=distance_strategy,
        child_policy=child_policy,
    )


def _coerce_namespace(value: object) -> EmbeddingNamespace:
    if isinstance(value, EmbeddingNamespace):
        return value
    return EmbeddingNamespace.from_mapping(value)


def canonical_namespace_json(value: object) -> str:
    """Serialize only a validated namespace using deterministic JSON."""

    return _coerce_namespace(value).canonical_json


def namespace_fingerprint(value: object) -> str:
    """Return SHA-256 over canonical, credential-free namespace JSON."""

    return _coerce_namespace(value).fingerprint


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS embedding_namespaces (
        fingerprint TEXT PRIMARY KEY
            CHECK(
                typeof(fingerprint) = 'text'
                AND length(fingerprint) = 64
                AND fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
        namespace_version INTEGER NOT NULL
            CHECK(namespace_version = 1),
        namespace_json TEXT NOT NULL
            CHECK(
                typeof(namespace_json) = 'text'
                AND length(CAST(namespace_json AS BLOB)) BETWEEN 2 AND 8192
            )
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_vectors (
        namespace_fingerprint TEXT NOT NULL,
        chunk_id TEXT NOT NULL
            CHECK(
                typeof(chunk_id) = 'text'
                AND length(CAST(chunk_id AS BLOB)) BETWEEN 1 AND 2048
            ),
        source_id TEXT NOT NULL
            CHECK(
                typeof(source_id) = 'text'
                AND length(CAST(source_id AS BLOB)) BETWEEN 1 AND 2048
            ),
        content_hash TEXT NOT NULL
            CHECK(
                typeof(content_hash) = 'text'
                AND length(content_hash) = 64
                AND content_hash NOT GLOB '*[^0-9a-f]*'
            ),
        dimension INTEGER NOT NULL
            CHECK(dimension BETWEEN 1 AND 65536),
        vector_encoding TEXT NOT NULL
            CHECK(vector_encoding = 'float32-le-v1'),
        vector BLOB NOT NULL
            CHECK(
                typeof(vector) = 'blob'
                AND length(vector) = dimension * 4
            ),
        vector_sha256 TEXT
            CHECK(
                vector_sha256 IS NULL
                OR (
                    typeof(vector_sha256) = 'text'
                    AND length(vector_sha256) = 64
                    AND vector_sha256 NOT GLOB '*[^0-9a-f]*'
                )
            ),
        PRIMARY KEY (namespace_fingerprint, chunk_id, content_hash),
        FOREIGN KEY (namespace_fingerprint)
            REFERENCES embedding_namespaces(fingerprint)
            ON DELETE CASCADE
    ) STRICT, WITHOUT ROWID
    """,
)


_VECTOR_COLUMNS_V1 = (
    "namespace_fingerprint",
    "chunk_id",
    "source_id",
    "content_hash",
    "dimension",
    "vector_encoding",
    "vector",
)
_VECTOR_COLUMNS_V2 = (*_VECTOR_COLUMNS_V1, "vector_sha256")


def _ensure_vector_checksum_schema(
    connection: sqlite3.Connection,
    *,
    user_version: int,
) -> None:
    """Safely migrate known pre-checksum rows without trusting their bytes."""

    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(embedding_vectors)")
    )
    if columns == _VECTOR_COLUMNS_V2:
        return
    if columns != _VECTOR_COLUMNS_V1 or user_version not in (0, 1):
        raise CacheSchemaError("embedding cache schema is incompatible")

    # Existing vectors cannot be assigned a trustworthy digest during
    # migration: doing so would bless corruption that predated checksums.  A
    # NULL digest preserves the cache rows non-destructively while making each
    # one a strict cache miss.  All production inserts below store a digest.
    connection.execute(
        """
        ALTER TABLE embedding_vectors ADD COLUMN vector_sha256 TEXT
            CHECK(
                vector_sha256 IS NULL
                OR (
                    typeof(vector_sha256) = 'text'
                    AND length(vector_sha256) = 64
                    AND vector_sha256 NOT GLOB '*[^0-9a-f]*'
                )
            )
        """
    )


def _cache_path(path: str | os.PathLike[str] | None) -> Path:
    if path is None:
        return CACHE_PATH
    if isinstance(path, (str, os.PathLike)):
        return Path(path)
    raise EmbeddingCacheError("cache path must be path-like")


def _prepare_cache_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise EmbeddingCacheError("cache path must not be a symbolic link")
    if path.exists() and not path.is_file():
        raise EmbeddingCacheError("cache path must be a regular file")


@contextmanager
def cache_transaction(
    path: str | os.PathLike[str] | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open one bounded, immediate SQLite transaction and always close it."""

    resolved_path = _cache_path(path)
    _prepare_cache_path(resolved_path)
    connection = sqlite3.connect(
        resolved_path,
        timeout=SQLITE_TIMEOUT_SECONDS,
        isolation_level=None,
    )
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT_SECONDS * 1_000}")
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
    finally:
        connection.close()


def initialize_cache(
    path: str | os.PathLike[str] | None = None,
    namespace: EmbeddingNamespace | Mapping[str, object] | None = None,
) -> Path:
    """Create the bounded schema and optionally register one namespace."""

    resolved_path = _cache_path(path)
    validated_namespace = None if namespace is None else _coerce_namespace(namespace)
    with cache_transaction(resolved_path) as connection:
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if user_version not in (0, 1, CACHE_SCHEMA_VERSION):
            raise CacheSchemaError("embedding cache schema version is incompatible")
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_vector_checksum_schema(connection, user_version=user_version)
        if user_version != CACHE_SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")

        if validated_namespace is not None:
            connection.execute(
                """
                INSERT OR IGNORE INTO embedding_namespaces(
                    fingerprint, namespace_version, namespace_json
                ) VALUES (?, ?, ?)
                """,
                (
                    validated_namespace.fingerprint,
                    validated_namespace.namespace_version,
                    validated_namespace.canonical_json,
                ),
            )
            stored = connection.execute(
                """
                SELECT namespace_version, namespace_json
                FROM embedding_namespaces
                WHERE fingerprint = ?
                """,
                (validated_namespace.fingerprint,),
            ).fetchone()
            if stored != (
                validated_namespace.namespace_version,
                validated_namespace.canonical_json,
            ):
                raise CacheSchemaError("namespace fingerprint collision")
    return resolved_path


def _bounded_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EmbeddingCacheError(f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_ID_CHARS:
        raise EmbeddingCacheError(f"{field} is too long")
    return value


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _content_hash(value: object) -> str:
    if not _is_lowercase_sha256(value):
        raise EmbeddingCacheError("content_hash must be a lowercase SHA-256 digest")
    return cast(str, value)


def _chunk_identity(chunk: object) -> tuple[str, str, str]:
    return (
        _bounded_id(getattr(chunk, "chunk_id", None), "chunk_id"),
        _bounded_id(getattr(chunk, "source_id", None), "source_id"),
        _content_hash(getattr(chunk, "content_hash", None)),
    )


def _validated_vector(
    value: object,
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingCacheError("embedding vector must be a numeric sequence")
    if len(value) != expected_dimension:
        raise EmbeddingCacheError("embedding vector dimension is incompatible")
    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise EmbeddingCacheError("embedding vector contains a non-numeric value")
        converted = float(component)
        if not math.isfinite(converted):
            raise EmbeddingCacheError("embedding vector contains a non-finite value")
        vector.append(converted)
    return tuple(vector)


def load_cached_vectors(
    chunks: Sequence[object],
    namespace: EmbeddingNamespace | Mapping[str, object],
    path: str | os.PathLike[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Load exact, structurally valid vectors for the requested children.

    Invalid rows are cache misses. Callers atomically replace the namespace
    only after all missing vectors have been obtained and validated.
    """

    validated_namespace = _coerce_namespace(namespace)
    resolved_path = initialize_cache(path, validated_namespace)
    result: dict[str, tuple[float, ...]] = {}
    with sqlite3.connect(resolved_path, timeout=SQLITE_TIMEOUT_SECONDS) as connection:
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_TIMEOUT_SECONDS * 1_000}")
        connection.execute("PRAGMA query_only = ON")
        for chunk in chunks:
            chunk_id, source_id, content_hash = _chunk_identity(chunk)
            row = connection.execute(
                """
                SELECT source_id, content_hash, dimension, vector_encoding,
                       vector, vector_sha256
                FROM embedding_vectors
                WHERE namespace_fingerprint = ?
                  AND chunk_id = ?
                  AND content_hash = ?
                """,
                (validated_namespace.fingerprint, chunk_id, content_hash),
            ).fetchone()
            if row is None:
                continue
            (
                stored_source,
                stored_hash,
                dimension,
                encoding,
                vector_blob,
                vector_sha256,
            ) = row
            if (
                stored_source != source_id
                or stored_hash != content_hash
                or dimension != validated_namespace.resolved_dimension
                or encoding != VECTOR_ENCODING
                or not isinstance(vector_blob, bytes)
                or len(vector_blob) != dimension * 4
                or not _is_lowercase_sha256(vector_sha256)
                or hashlib.sha256(vector_blob).hexdigest() != vector_sha256
            ):
                continue
            try:
                vector = struct.unpack(f"<{dimension}f", vector_blob)
            except (struct.error, TypeError):
                continue
            if not all(math.isfinite(component) for component in vector):
                continue
            result[chunk_id] = vector
    return result


def _activate_candidate_rows(
    connection: sqlite3.Connection,
    namespace_fingerprint: str,
    rows: Sequence[tuple[object, ...]],
) -> None:
    """Switch one namespace to validated candidate rows inside a caller transaction."""

    connection.execute(
        "DELETE FROM embedding_vectors WHERE namespace_fingerprint = ?",
        (namespace_fingerprint,),
    )
    connection.executemany(
        """
        INSERT INTO embedding_vectors(
            namespace_fingerprint, chunk_id, source_id, content_hash,
            dimension, vector_encoding, vector, vector_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def replace_cached_vectors(
    chunks: Sequence[object],
    vectors: Sequence[Sequence[float]],
    namespace: EmbeddingNamespace | Mapping[str, object],
    path: str | os.PathLike[str] | None = None,
    *,
    activation_check: Callable[[], bool] | None = None,
) -> None:
    """Atomically replace one namespace's active child vectors.

    ``activation_check`` runs inside the write transaction immediately before
    the active-row switch. A stale candidate therefore leaves the prior active
    mapping untouched.
    """

    validated_namespace = _coerce_namespace(namespace)
    if len(chunks) != len(vectors):
        raise EmbeddingCacheError("chunk/vector count mismatch")

    rows: list[tuple[object, ...]] = []
    seen_chunk_ids: set[str] = set()
    for chunk, raw_vector in zip(chunks, vectors, strict=True):
        chunk_id, source_id, content_hash = _chunk_identity(chunk)
        if chunk_id in seen_chunk_ids:
            raise EmbeddingCacheError("duplicate chunk_id in active generation")
        seen_chunk_ids.add(chunk_id)
        vector = _validated_vector(
            raw_vector,
            expected_dimension=validated_namespace.resolved_dimension,
        )
        vector_blob = struct.pack(f"<{len(vector)}f", *vector)
        rows.append(
            (
                validated_namespace.fingerprint,
                chunk_id,
                source_id,
                content_hash,
                validated_namespace.resolved_dimension,
                VECTOR_ENCODING,
                vector_blob,
                hashlib.sha256(vector_blob).hexdigest(),
            )
        )

    resolved_path = initialize_cache(path, validated_namespace)
    with cache_transaction(resolved_path) as connection:
        if activation_check is not None:
            try:
                candidate_is_current = activation_check()
            except Exception as exc:
                raise CacheActivationError(
                    "candidate corpus validation failed"
                ) from exc
            if candidate_is_current is not True:
                raise CacheActivationError(
                    "candidate corpus changed before cache activation"
                )
        _activate_candidate_rows(
            connection,
            validated_namespace.fingerprint,
            rows,
        )
