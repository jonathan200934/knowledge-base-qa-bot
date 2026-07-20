from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, cast

from dotenv import load_dotenv

from . import embedding_cache
from .chunking import DEFAULT_CHUNKING_POLICY, ChunkingPolicy


DEFAULT_EMBEDDING_BATCH_SIZE = 64


REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX_VERSION = 2
FAISS_INDEX_FILENAME = "index.faiss"
FAISS_CHILDREN_FILENAME = "children.json"
FAISS_MANIFEST_FILENAME = "manifest.json"
# Source-level compatibility only. The v2 artifact never writes metadata.json.
FAISS_METADATA_FILENAME = FAISS_MANIFEST_FILENAME
FAISS_MANIFEST_SCHEMA = "markdown-kb-faiss-manifest"
FAISS_MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CHILDREN_BYTES = 64 * 1024 * 1024
MAX_CHILD_RECORD_BYTES = 16 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024 * 1024
_EXPECTED_ARTIFACTS = frozenset(
    {FAISS_INDEX_FILENAME, FAISS_CHILDREN_FILENAME, FAISS_MANIFEST_FILENAME}
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


class EmbeddingProvider(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class FaissIndexError(RuntimeError):
    """Raised when FAISS artifacts cannot be built or loaded safely."""


@dataclass(frozen=True)
class FaissMetadata:
    index_version: int
    embedding_model: str
    embedding_dimension: int
    section_count: int
    source_ids: list[str]
    section_hashes: dict[str, str]
    child_records: list[dict[str, str]] = field(default_factory=list)
    namespace: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "index_version": self.index_version,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "section_count": self.section_count,
            # Preserve the historical canonical source list for diagnostics,
            # while persisting the identities that actually address vectors.
            "source_ids": [_parent_source_id(source_id) for source_id in self.source_ids],
            "child_ids": self.source_ids,
            "section_hashes": {source_id: self.section_hashes[source_id] for source_id in sorted(self.section_hashes)},
            "child_records": self.child_records,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FaissMetadata":
        if not isinstance(payload, dict):
            raise FaissIndexError("FAISS metadata must be an object")
        try:
            index_version = int(payload["index_version"])
            embedding_model = str(payload["embedding_model"])
            embedding_dimension = int(payload["embedding_dimension"])
            section_count = int(payload["section_count"])
            source_ids_payload = payload.get("child_ids", payload["source_ids"])
            section_hashes_payload = payload.get("section_hashes", {})
            child_records_payload = payload.get("child_records", [])
            namespace_payload = payload.get("namespace", {})
        except (KeyError, TypeError, ValueError) as exc:
            raise FaissIndexError("Malformed FAISS metadata") from exc
        if index_version != FAISS_INDEX_VERSION:
            raise FaissIndexError("Unsupported FAISS index version")
        if not embedding_model:
            raise FaissIndexError("FAISS metadata embedding model is empty")
        if embedding_dimension <= 0:
            raise FaissIndexError("FAISS embedding dimension must be positive")
        if section_count < 0:
            raise FaissIndexError("FAISS section count must be non-negative")
        if not isinstance(source_ids_payload, list) or not all(isinstance(item, str) for item in source_ids_payload):
            raise FaissIndexError("FAISS source_ids must be a string list")
        if len(source_ids_payload) != section_count:
            raise FaissIndexError("FAISS section_count does not match source_ids")
        if len(set(source_ids_payload)) != len(source_ids_payload):
            raise FaissIndexError("FAISS source_ids contain duplicates")
        if not isinstance(section_hashes_payload, dict):
            raise FaissIndexError("FAISS section_hashes must be an object")
        section_hashes = {str(key): str(value) for key, value in section_hashes_payload.items()}
        return cls(
            index_version=index_version,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            section_count=section_count,
            source_ids=list(source_ids_payload),
            section_hashes=section_hashes,
            child_records=list(child_records_payload),
            namespace=dict(namespace_payload),
        )


class OpenAIEmbeddingProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL")
        self.provider_kind = "openai-compatible"
        requested_dimension = os.getenv("OPENAI_EMBEDDING_DIMENSION")
        try:
            self.requested_dimension = int(requested_dimension) if requested_dimension else None
        except ValueError as exc:
            raise FaissIndexError("OPENAI_EMBEDDING_DIMENSION must be an integer") from exc
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = self.requested_dimension or known_dimensions.get(self.model)
        if not self.api_key:
            raise FaissIndexError("OPENAI_API_KEY is not set; OpenAI embeddings are unavailable")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency check
            raise FaissIndexError("openai package is required for OpenAI embeddings") from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request: dict[str, Any] = {"model": self.model, "input": texts}
        if self.requested_dimension is not None:
            request["dimensions"] = self.requested_dimension
        try:
            response = self._client.embeddings.create(**request)
            return [
                list(item.embedding)
                for item in sorted(response.data, key=lambda item: item.index)
            ]
        except Exception as exc:
            # Normalize SDK, transport, API, and malformed-response failures at
            # the provider boundary. Callers can then enter the deterministic
            # local fallback without exposing provider payloads or credentials.
            raise FaissIndexError("OpenAI embedding request failed") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbeddingProvider:
    """Deterministic embedding provider for tests and offline smoke checks.

    It creates normalized hashed lexical vectors, so tests can exercise the same
    FAISS persistence/search path without any live OpenAI request or credential.
    """

    def __init__(self, model: str = "fake-openai-hash-embedding", dimension: int = 256):
        self.model = model
        self.dimension = dimension
        self.requested_dimension = dimension
        self.provider_kind = "local-hash"
        self.base_url = "https://local.invalid/v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        tokens = _tokens(text)
        features: list[str] = []
        for token in tokens:
            features.append(token)
            if token.endswith("s") and len(token) > 3:
                features.append(token[:-1])
        features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            values[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            return values
        return [value / norm for value in values]


def _provider_dimension(provider: EmbeddingProvider) -> int:
    dimension = getattr(provider, "dimension", None)
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise FaissIndexError(
            "Embedding provider must declare a positive resolved dimension when cache is enabled"
        )
    return dimension


def _provider_namespace(
    provider: EmbeddingProvider,
    resolved_dimension: int,
    chunking_policy: ChunkingPolicy,
) -> embedding_cache.EmbeddingNamespace:
    requested_dimension = getattr(provider, "requested_dimension", None)
    if requested_dimension is not None and (
        not isinstance(requested_dimension, int)
        or isinstance(requested_dimension, bool)
        or requested_dimension <= 0
    ):
        raise FaissIndexError("Embedding provider requested dimension is invalid")
    try:
        return embedding_cache.build_namespace(
            provider_kind=str(getattr(provider, "provider_kind", "openai-compatible")),
            base_url=str(
                getattr(provider, "base_url", None) or "https://api.openai.com/v1"
            ),
            embedding_model=provider.model,
            requested_dimension=requested_dimension,
            resolved_dimension=resolved_dimension,
            vector_normalization="l2",
            metric="inner-product",
            distance_strategy="cosine-similarity",
            chunking_policy=chunking_policy,
        )
    except embedding_cache.EmbeddingCacheError as exc:
        raise FaissIndexError("Embedding provider cache namespace is invalid") from exc


def provider_config_fingerprint(
    provider: EmbeddingProvider,
    resolved_dimension: int,
    chunking_policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> str:
    """Return the complete vector-affecting provider/config identity."""

    return _provider_namespace(provider, resolved_dimension, chunking_policy).fingerprint


class FaissSectionIndex:
    def __init__(self, index, metadata: FaissMetadata):
        self.index = index
        self.metadata = metadata
        self.source_ids = list(metadata.source_ids)

    @classmethod
    def build(
        cls,
        sections: Iterable[object],
        provider: EmbeddingProvider,
        *,
        cache_path: str | os.PathLike[str] | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        chunking_policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
        activation_check: Callable[[], bool] | None = None,
    ) -> "FaissSectionIndex":
        materialized = list(sections)
        pending_cache_activation: tuple[
            embedding_cache.EmbeddingNamespace, list[list[float]]
        ] | None = None
        source_ids = [
            str(getattr(section, "chunk_id", getattr(section, "id")))
            for section in materialized
        ]
        if len(set(source_ids)) != len(source_ids):
            raise FaissIndexError("Cannot build FAISS index with duplicate section IDs")
        texts = [section_text(section) for section in materialized]

        if cache_path is None:
            embeddings = provider.embed_documents(texts)
            if len(embeddings) != len(materialized):
                raise FaissIndexError("Embedding provider returned the wrong number of vectors")
            matrix = _normalized_matrix(embeddings)
            if materialized:
                dimension = int(matrix.shape[1])
                # Preserve compatibility with simple in-memory providers that
                # predate the resolved-dimension contract, but never ignore a
                # dimension that a provider does declare.
                if getattr(provider, "dimension", None) is not None:
                    resolved_dimension = _provider_dimension(provider)
                    if dimension != resolved_dimension:
                        raise FaissIndexError(
                            "Embedding vector dimension does not match provider dimension"
                        )
            else:
                # There is no vector from which to infer the native index
                # dimension. Use the provider contract so an empty generation
                # has the same dimension expected by load().
                dimension = _provider_dimension(provider)
        else:
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
                raise FaissIndexError("Embedding batch size must be a positive integer")
            resolved_dimension = _provider_dimension(provider)
            namespace = _provider_namespace(provider, resolved_dimension, chunking_policy)
            try:
                cached = embedding_cache.load_cached_vectors(
                    materialized,
                    namespace,
                    cache_path,
                )
            except (embedding_cache.EmbeddingCacheError, sqlite3.Error, OSError) as exc:
                raise FaissIndexError("Embedding cache could not be read safely") from exc

            vectors_by_id: dict[str, list[float]] = {
                source_id: list(vector) for source_id, vector in cached.items()
            }
            missing = [
                (section, source_id, text)
                for section, source_id, text in zip(
                    materialized, source_ids, texts, strict=True
                )
                if source_id not in vectors_by_id
            ]
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                response = provider.embed_documents([text for _section, _source_id, text in batch])
                if not isinstance(response, Sequence) or len(response) != len(batch):
                    raise FaissIndexError(
                        "Embedding provider returned the wrong number of vectors"
                    )
                for (_section, source_id, _text), vector in zip(
                    batch, response, strict=True
                ):
                    vectors_by_id[source_id] = vector

            try:
                ordered_embeddings = [vectors_by_id[source_id] for source_id in source_ids]
            except KeyError as exc:  # defensive invariant
                raise FaissIndexError("Embedding cache merge lost an active child") from exc
            matrix = _normalized_matrix(ordered_embeddings)
            if materialized and matrix.shape[1] != resolved_dimension:
                raise FaissIndexError(
                    "Embedding vector dimension does not match provider namespace"
                )
            dimension = int(matrix.shape[1]) if materialized else resolved_dimension
            pending_cache_activation = (
                namespace,
                matrix.tolist() if materialized else [],
            )

        if dimension <= 0:
            raise FaissIndexError("Embedding dimension must be positive")
        faiss = _require_faiss()
        index = faiss.IndexFlatIP(dimension)
        if len(materialized):
            cast(Any, index).add(matrix)
        artifact_namespace = _provider_namespace(provider, dimension, chunking_policy)
        child_records = _child_records(materialized)
        metadata = FaissMetadata(
            index_version=FAISS_INDEX_VERSION,
            embedding_model=provider.model,
            embedding_dimension=dimension,
            section_count=len(materialized),
            source_ids=source_ids,
            section_hashes={source_id: section_hash(section) for source_id, section in zip(source_ids, materialized)},
            child_records=child_records,
            namespace=artifact_namespace.to_dict(),
        )
        if pending_cache_activation is not None:
            namespace, vectors = pending_cache_activation
            try:
                embedding_cache.replace_cached_vectors(
                    materialized,
                    vectors,
                    namespace,
                    cache_path,
                    activation_check=activation_check,
                )
            except (embedding_cache.EmbeddingCacheError, sqlite3.Error, OSError) as exc:
                raise FaissIndexError("Embedding cache could not be updated safely") from exc
        return cls(index=index, metadata=metadata)

    @classmethod
    def load(
        cls,
        directory: Path,
        sections: Sequence[object],
        provider: EmbeddingProvider,
        *,
        chunking_policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
    ) -> "FaissSectionIndex":
        materialized = list(sections)
        expected = _expected_metadata(
            materialized,
            provider,
            _provider_dimension(provider),
            chunking_policy,
        )
        index, metadata = _load_validated_generation(Path(directory), expected)
        return cls(index=index, metadata=metadata)

    def save(
        self,
        directory: Path,
        *,
        activation_check: Callable[[], bool] | None = None,
    ) -> None:
        directory = Path(directory)
        parent = directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        _reject_unsafe_activation_target(directory)
        candidate = parent / f".{directory.name}.tmp-{uuid.uuid4().hex}"
        try:
            os.mkdir(candidate, mode=0o700)
            faiss = _require_faiss()
            index_path = candidate / FAISS_INDEX_FILENAME
            faiss.write_index(self.index, str(index_path))
            _fsync_file(index_path)

            children_bytes = _canonical_json_bytes(self.metadata.child_records)
            children_path = candidate / FAISS_CHILDREN_FILENAME
            _write_fsynced(children_path, children_bytes)

            index_bytes = _read_regular_bytes(index_path, MAX_INDEX_BYTES, "FAISS index")
            manifest = _manifest_for(self.metadata, index_bytes, children_bytes)
            manifest_bytes = _canonical_json_bytes(manifest)
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise FaissIndexError("FAISS manifest exceeds the safety limit")
            _write_fsynced(candidate / FAISS_MANIFEST_FILENAME, manifest_bytes)
            _fsync_directory(candidate)

            # Validate the complete candidate, including native FAISS invariants,
            # before it can replace the active generation.
            _load_validated_generation(candidate, self.metadata)
            if activation_check is not None and not activation_check():
                raise FaissIndexError(
                    "FAISS candidate corpus changed before artifact activation"
                )

            if directory.exists():
                # RENAME_EXCHANGE keeps one complete generation at the active
                # path throughout activation. A two-step backup/replace has a
                # crash window where the active path does not exist.
                _atomic_exchange_directories(candidate, directory)
                _fsync_directory(parent)
                try:
                    # The exchange has already made the validated candidate
                    # active. Failure to reap the displaced generation must not
                    # report activation failure or disable the valid runtime.
                    shutil.rmtree(candidate)
                    _fsync_directory(parent)
                except OSError:
                    pass
            else:
                os.replace(candidate, directory)
                _fsync_directory(parent)
        except FaissIndexError:
            if candidate.exists():
                shutil.rmtree(candidate, ignore_errors=True)
            raise
        except OSError as exc:
            if candidate.exists():
                shutil.rmtree(candidate, ignore_errors=True)
            if directory.exists():
                raise FaissIndexError("Could not atomically activate FAISS generation") from exc
            raise FaissIndexError("Could not persist or activate FAISS generation") from exc

    def is_compatible(
        self,
        sections: Sequence[object],
        provider_or_model: EmbeddingProvider | str | None = None,
        *,
        chunking_policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
    ) -> bool:
        expected = _expected_metadata(
            sections,
            provider_or_model or self.metadata.embedding_model,
            self.metadata.embedding_dimension,
            chunking_policy,
        )
        return metadata_compatible(self.metadata, expected)

    def search(self, query: str, provider: EmbeddingProvider, k: int = 3) -> list[tuple[str, float]]:
        if k <= 0 or self.metadata.section_count <= 0:
            return []
        vector = _normalized_matrix([provider.embed_query(query)])
        if vector.shape[1] != self.metadata.embedding_dimension:
            raise FaissIndexError("Query embedding dimension does not match FAISS index")
        scores, indices = self.index.search(vector, min(k, self.metadata.section_count))
        ranked: list[tuple[str, float]] = []
        for score, index_position in zip(scores[0].tolist(), indices[0].tolist()):
            if index_position < 0 or index_position >= len(self.source_ids):
                continue
            if not math.isfinite(score) or score <= 0:
                continue
            ranked.append((self.source_ids[index_position], float(score)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]


def metadata_compatible(actual: FaissMetadata, expected: FaissMetadata) -> bool:
    return (
        actual.index_version == expected.index_version
        and actual.embedding_model == expected.embedding_model
        and actual.embedding_dimension == expected.embedding_dimension
        and actual.section_count == expected.section_count
        and actual.source_ids == expected.source_ids
        and actual.section_hashes == expected.section_hashes
        and (not expected.child_records or actual.child_records == expected.child_records)
        and (not expected.namespace or actual.namespace == expected.namespace)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _child_records(sections: Sequence[object]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for section in sections:
        chunk_id = str(getattr(section, "chunk_id", getattr(section, "id", "")))
        source_id = str(getattr(section, "source_id", _parent_source_id(chunk_id)))
        digest = section_hash(section)
        content_hash = str(getattr(section, "content_hash", "") or digest)
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            content_hash = digest
        records.append(
            {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "file": str(getattr(section, "file", "")),
                "content_hash": content_hash,
                "section_hash": digest,
            }
        )
    return records


def _corpus_fingerprint(records: list[dict[str, str]]) -> str:
    return _sha256(_canonical_json_bytes(records))


def _manifest_for(
    metadata: FaissMetadata,
    index_bytes: bytes,
    children_bytes: bytes,
) -> dict[str, object]:
    try:
        namespace = embedding_cache.EmbeddingNamespace.from_mapping(metadata.namespace)
    except embedding_cache.EmbeddingCacheError as exc:
        raise FaissIndexError("FAISS metadata has an invalid embedding namespace") from exc
    namespace_payload = namespace.to_dict()
    provider = cast(dict[str, Any], namespace_payload["provider"])
    vector = cast(dict[str, Any], namespace_payload["vector"])
    child_policy = namespace_payload["child_policy"]
    index_sha = _sha256(index_bytes)
    children_sha = _sha256(children_bytes)
    return {
        "schema": FAISS_MANIFEST_SCHEMA,
        "manifest_version": FAISS_MANIFEST_VERSION,
        "index_version": metadata.index_version,
        "provider": {
            "kind": provider["kind"],
            "model": provider["embedding_model"],
            "base_url": provider["base_url"],
            "requested_dimension": provider["requested_dimension"],
            "resolved_dimension": provider["resolved_dimension"],
            "config_fingerprint": namespace.fingerprint,
        },
        "index": {
            "dimension": metadata.embedding_dimension,
            "normalization": vector["normalization"],
            "metric": str(vector["metric"]).replace("-", "_"),
            "distance_strategy": vector["distance_strategy"],
            "type": "IndexFlatIP",
        },
        "chunking_policy": child_policy,
        "counts": {
            "chunks": metadata.section_count,
            "sections": len({record["source_id"] for record in metadata.child_records}),
            "files": len({record["file"] for record in metadata.child_records}),
        },
        "corpus_fingerprint": _corpus_fingerprint(metadata.child_records),
        "ordered_mapping_sha256": children_sha,
        "payloads": {
            FAISS_INDEX_FILENAME: {"sha256": index_sha, "bytes": len(index_bytes)},
            FAISS_CHILDREN_FILENAME: {"sha256": children_sha, "bytes": len(children_bytes)},
        },
    }


def _reject_unsafe_activation_target(directory: Path) -> None:
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise FaissIndexError("FAISS activation target must be a real directory")


def _preflight_directory(directory: Path) -> None:
    directory_fd = _open_generation_dirfd(directory)
    try:
        _preflight_directory_fd(directory_fd)
    finally:
        os.close(directory_fd)


def _open_generation_dirfd(directory: Path) -> int:
    """Open one immutable directory generation without following symlinks.

    Every artifact read is subsequently relative to this descriptor, so a
    concurrent rename of the public path cannot mix files from two generations.
    Platforms without safe directory-relative opens reject FAISS persistence and
    leave the caller on the deterministic local retrieval path.
    """

    if os.open not in os.supports_dir_fd:
        raise FaissIndexError("Safe directory-relative FAISS loading is unavailable")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        path_info = directory.lstat()
        if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode):
            raise FaissIndexError("FAISS artifact path must be a real directory")
        directory_fd = os.open(directory, flags)
        opened_info = os.fstat(directory_fd)
    except FaissIndexError:
        raise
    except OSError as exc:
        raise FaissIndexError("FAISS artifact directory is missing or inaccessible") from exc
    if (
        not stat.S_ISDIR(opened_info.st_mode)
        or (opened_info.st_dev, opened_info.st_ino)
        != (path_info.st_dev, path_info.st_ino)
    ):
        os.close(directory_fd)
        raise FaissIndexError("FAISS artifact directory changed while opening")
    return directory_fd


def _preflight_directory_fd(directory_fd: int) -> None:
    try:
        entries = set(os.listdir(directory_fd))
    except OSError as exc:
        raise FaissIndexError("FAISS artifact layout cannot be inspected safely") from exc
    if entries != _EXPECTED_ARTIFACTS:
        raise FaissIndexError("FAISS artifact layout is incomplete or unexpected")


def _open_regular_fd(
    path: Path | str,
    max_bytes: int,
    label: str,
    *,
    directory_fd: int | None = None,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        if directory_fd is None:
            fd = os.open(path, flags)
        else:
            fd = os.open(path, flags, dir_fd=directory_fd)
        info = os.fstat(fd)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise FaissIndexError(f"{label} is missing or unsafe") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > max_bytes:
        os.close(fd)
        raise FaissIndexError(f"{label} is not a bounded regular file")
    return fd


def _read_fd(fd: int, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > max_bytes:
        raise FaissIndexError(f"{label} exceeds the safety limit")
    return payload


def _read_regular_bytes(path: Path, max_bytes: int, label: str) -> bytes:
    fd = _open_regular_fd(path, max_bytes, label)
    try:
        return _read_fd(fd, max_bytes, label)
    finally:
        os.close(fd)


def _read_regular_bytes_at(
    directory_fd: int, filename: str, max_bytes: int, label: str
) -> bytes:
    fd = _open_regular_fd(
        filename,
        max_bytes,
        label,
        directory_fd=directory_fd,
    )
    try:
        return _read_fd(fd, max_bytes, label)
    finally:
        os.close(fd)


def _strict_json(payload: bytes, label: str) -> object:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid constant: {value}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FaissIndexError(f"{label} is not strict JSON") from exc


def _exact_mapping(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise FaissIndexError(f"{label} has an unexpected schema")
    return value


def _bounded_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise FaissIndexError(f"{label} is outside the safety bounds")
    return value


def _validate_child_records(value: object, expected_count: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise FaissIndexError("FAISS ordered child metadata count is invalid")
    records: list[dict[str, str]] = []
    expected_keys = frozenset({"chunk_id", "source_id", "file", "content_hash", "section_hash"})
    seen: set[str] = set()
    for raw in value:
        item = _exact_mapping(raw, expected_keys, "FAISS child record")
        record: dict[str, str] = {}
        for key in ("chunk_id", "source_id", "file", "content_hash", "section_hash"):
            field_value = item[key]
            if not isinstance(field_value, str) or not field_value or len(field_value.encode("utf-8")) > 8192:
                raise FaissIndexError(f"FAISS child {key} is invalid")
            record[key] = field_value
        if record["chunk_id"] in seen:
            raise FaissIndexError("FAISS ordered child metadata has duplicate IDs")
        if not re.fullmatch(r"[0-9a-f]{64}", record["content_hash"]) or not re.fullmatch(
            r"[0-9a-f]{64}", record["section_hash"]
        ):
            raise FaissIndexError("FAISS child hashes are invalid")
        file_path = record["file"]
        if (
            "\\" in file_path
            or file_path.startswith("/")
            or re.match(r"^[A-Za-z]:", file_path)
            or any(part in {"", ".", ".."} for part in file_path.split("/"))
        ):
            raise FaissIndexError("FAISS child file path is unsafe")
        seen.add(record["chunk_id"])
        records.append(record)
    return records


def _validate_manifest(
    value: object,
    *,
    index_bytes: bytes,
    children_bytes: bytes,
    expected: FaissMetadata,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    manifest = _exact_mapping(
        value,
        frozenset(
            {
                "schema", "manifest_version", "index_version", "provider", "index",
                "chunking_policy", "counts", "corpus_fingerprint", "ordered_mapping_sha256",
                "payloads",
            }
        ),
        "FAISS manifest",
    )
    if manifest["schema"] != FAISS_MANIFEST_SCHEMA:
        raise FaissIndexError("FAISS manifest schema is unsupported")
    if manifest["manifest_version"] != FAISS_MANIFEST_VERSION or manifest["index_version"] != FAISS_INDEX_VERSION:
        raise FaissIndexError("FAISS manifest version is unsupported")

    provider = _exact_mapping(
        manifest["provider"],
        frozenset({"kind", "model", "base_url", "requested_dimension", "resolved_dimension", "config_fingerprint"}),
        "FAISS provider manifest",
    )
    index_config = _exact_mapping(
        manifest["index"],
        frozenset({"dimension", "normalization", "metric", "distance_strategy", "type"}),
        "FAISS index manifest",
    )
    counts = _exact_mapping(
        manifest["counts"], frozenset({"chunks", "sections", "files"}), "FAISS counts"
    )
    payloads = _exact_mapping(
        manifest["payloads"], frozenset({FAISS_INDEX_FILENAME, FAISS_CHILDREN_FILENAME}), "FAISS payloads"
    )
    index_payload = _exact_mapping(
        payloads[FAISS_INDEX_FILENAME], frozenset({"sha256", "bytes"}), "FAISS index payload"
    )
    children_payload = _exact_mapping(
        payloads[FAISS_CHILDREN_FILENAME], frozenset({"sha256", "bytes"}), "FAISS children payload"
    )

    child_count = _bounded_int(counts["chunks"], "FAISS child count", minimum=0, maximum=1_000_000)
    dimension = _bounded_int(index_config["dimension"], "FAISS dimension", minimum=1, maximum=65_536)
    _bounded_int(counts["sections"], "FAISS section count", minimum=0, maximum=child_count)
    _bounded_int(counts["files"], "FAISS file count", minimum=0, maximum=child_count)
    if index_payload["bytes"] != len(index_bytes) or children_payload["bytes"] != len(children_bytes):
        raise FaissIndexError("FAISS payload size does not match manifest")
    index_sha = _sha256(index_bytes)
    children_sha = _sha256(children_bytes)
    if index_payload["sha256"] != index_sha or children_payload["sha256"] != children_sha:
        raise FaissIndexError("FAISS payload checksum does not match manifest")
    if manifest["ordered_mapping_sha256"] != children_sha:
        raise FaissIndexError("FAISS ordered mapping checksum is invalid")

    records = _validate_child_records(_strict_json(children_bytes, "FAISS child metadata"), child_count)
    if manifest["corpus_fingerprint"] != _corpus_fingerprint(records):
        raise FaissIndexError("FAISS corpus fingerprint is invalid")
    if counts["sections"] != len({record["source_id"] for record in records}) or counts["files"] != len(
        {record["file"] for record in records}
    ):
        raise FaissIndexError("FAISS manifest counts are inconsistent")

    try:
        expected_namespace = embedding_cache.EmbeddingNamespace.from_mapping(expected.namespace)
    except embedding_cache.EmbeddingCacheError as exc:
        raise FaissIndexError("Expected embedding namespace is invalid") from exc
    expected_payload = expected_namespace.to_dict()
    expected_provider = cast(dict[str, Any], expected_payload["provider"])
    expected_vector = cast(dict[str, Any], expected_payload["vector"])
    if provider != {
        "kind": expected_provider["kind"],
        "model": expected_provider["embedding_model"],
        "base_url": expected_provider["base_url"],
        "requested_dimension": expected_provider["requested_dimension"],
        "resolved_dimension": expected_provider["resolved_dimension"],
        "config_fingerprint": expected_namespace.fingerprint,
    }:
        raise FaissIndexError("FAISS provider identity is stale or incompatible")
    if index_config != {
        "dimension": expected.embedding_dimension,
        "normalization": expected_vector["normalization"],
        "metric": str(expected_vector["metric"]).replace("-", "_"),
        "distance_strategy": expected_vector["distance_strategy"],
        "type": "IndexFlatIP",
    } or manifest["chunking_policy"] != expected_payload["child_policy"]:
        raise FaissIndexError("FAISS vector configuration is stale or incompatible")
    if dimension != expected.embedding_dimension or records != expected.child_records:
        raise FaissIndexError("FAISS corpus metadata is stale or incompatible")
    return manifest, records


def _native_deserialize_index(payload: bytes):
    """Parse only bytes already bounded and checksum-validated by preflight."""

    try:
        import numpy as np

        faiss = _require_faiss()
        serialized = np.frombuffer(payload, dtype="uint8")
        return faiss.deserialize_index(serialized)
    except FaissIndexError:
        raise
    except Exception as exc:
        raise FaissIndexError("FAISS native index could not be parsed safely") from exc


def _load_validated_generation(
    directory: Path, expected: FaissMetadata
) -> tuple[object, FaissMetadata]:
    directory_fd = _open_generation_dirfd(directory)
    try:
        _preflight_directory_fd(directory_fd)
        manifest_bytes = _read_regular_bytes_at(
            directory_fd,
            FAISS_MANIFEST_FILENAME,
            MAX_MANIFEST_BYTES,
            "FAISS manifest",
        )
        raw_manifest = _strict_json(manifest_bytes, "FAISS manifest")
        partial = _exact_mapping(
            raw_manifest,
            frozenset(
                {
                    "schema", "manifest_version", "index_version", "provider", "index",
                    "chunking_policy", "counts", "corpus_fingerprint", "ordered_mapping_sha256",
                    "payloads",
                }
            ),
            "FAISS manifest",
        )
        index_config = _exact_mapping(
            partial["index"],
            frozenset({"dimension", "normalization", "metric", "distance_strategy", "type"}),
            "FAISS index manifest",
        )
        counts = _exact_mapping(
            partial["counts"], frozenset({"chunks", "sections", "files"}), "FAISS counts"
        )
        child_count = _bounded_int(
            counts["chunks"], "FAISS child count", minimum=0, maximum=1_000_000
        )
        dimension = _bounded_int(
            index_config["dimension"], "FAISS dimension", minimum=1, maximum=65_536
        )
        children_limit = min(
            MAX_CHILDREN_BYTES, 4096 + child_count * MAX_CHILD_RECORD_BYTES
        )
        index_limit = min(
            MAX_INDEX_BYTES, 1024 * 1024 + child_count * dimension * 4
        )
        children_bytes = _read_regular_bytes_at(
            directory_fd,
            FAISS_CHILDREN_FILENAME,
            children_limit,
            "FAISS child metadata",
        )
        index_bytes = _read_regular_bytes_at(
            directory_fd,
            FAISS_INDEX_FILENAME,
            index_limit,
            "FAISS index",
        )
        _manifest, records = _validate_manifest(
            partial,
            index_bytes=index_bytes,
            children_bytes=children_bytes,
            expected=expected,
        )
        metadata = FaissMetadata(
            index_version=FAISS_INDEX_VERSION,
            embedding_model=expected.embedding_model,
            embedding_dimension=expected.embedding_dimension,
            section_count=len(records),
            source_ids=[record["chunk_id"] for record in records],
            section_hashes={
                record["chunk_id"]: record["section_hash"] for record in records
            },
            child_records=records,
            namespace=expected.namespace,
        )
        if not metadata_compatible(metadata, expected):
            raise FaissIndexError("FAISS metadata is stale or incompatible")
        # Deserialize the exact bytes read from this anchored generation only
        # after all schema, bound, and checksum checks have succeeded.
        index = _native_deserialize_index(index_bytes)
    finally:
        os.close(directory_fd)

    faiss = _require_faiss()
    if index.d != metadata.embedding_dimension:
        raise FaissIndexError("FAISS native index dimension does not match manifest")
    if index.ntotal != metadata.section_count:
        raise FaissIndexError("FAISS native index count does not match manifest")
    if type(index) is not faiss.IndexFlatIP:
        raise FaissIndexError("FAISS index type is not the required exact flat-IP type")
    if getattr(index, "metric_type", None) != faiss.METRIC_INNER_PRODUCT:
        raise FaissIndexError("FAISS index metric is not inner product")
    return index, metadata


def _write_fsynced(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    fd = _open_regular_fd(path, MAX_INDEX_BYTES, "FAISS index")
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_exchange_directories(source: Path, destination: Path) -> None:
    """Atomically exchange two existing directory entries on Linux."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = getattr(os, "AT_FDCWD", -100)
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_exchange,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _parent_source_id(dense_id: str) -> str:
    marker = "::chunk-"
    return dense_id.rsplit(marker, 1)[0] if marker in dense_id else dense_id


def section_text(section: object) -> str:
    embedding_text = getattr(section, "embedding_text", None)
    if embedding_text is not None:
        return str(embedding_text)
    heading_path = " > ".join(str(part) for part in getattr(section, "heading_path", []) or [])
    content = str(getattr(section, "content", ""))
    return f"{heading_path}\n\n{content}".strip()


def section_hash(section: object) -> str:
    payload = {
        "id": getattr(section, "chunk_id", getattr(section, "id", "")),
        "source_id": getattr(section, "source_id", getattr(section, "id", "")),
        "file": getattr(section, "file", ""),
        "heading": getattr(section, "heading", ""),
        "heading_path": list(getattr(section, "heading_path", []) or []),
        "content": getattr(section, "content", ""),
        # Follow the exact vector input and every versioned child field that can
        # alter it. A child's embedding payload is heading-prefixed and is not
        # necessarily identical to its public ``content``.
        "embedding_text": section_text(section),
        "content_hash": getattr(section, "content_hash", None),
        "splitter_version": getattr(section, "splitter_version", None),
        "heading_prefix_version": getattr(section, "heading_prefix_version", None),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _expected_metadata(
    sections: Sequence[object],
    provider_or_model: EmbeddingProvider | str,
    dimension: int,
    chunking_policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> FaissMetadata:
    source_ids = [
        str(getattr(section, "chunk_id", getattr(section, "id")))
        for section in sections
    ]
    if isinstance(provider_or_model, str):
        model = provider_or_model
        namespace: dict[str, object] = {}
    else:
        model = provider_or_model.model
        namespace = _provider_namespace(
            provider_or_model, dimension, chunking_policy
        ).to_dict()
    return FaissMetadata(
        index_version=FAISS_INDEX_VERSION,
        embedding_model=model,
        embedding_dimension=dimension,
        section_count=len(source_ids),
        source_ids=source_ids,
        section_hashes={source_id: section_hash(section) for source_id, section in zip(source_ids, sections)},
        child_records=_child_records(sections),
        namespace=namespace,
    )


def _normalized_matrix(embeddings: list[list[float]]):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency check
        raise FaissIndexError("numpy is required for FAISS embeddings") from exc
    if not embeddings:
        return np.zeros((0, 1), dtype="float32")
    try:
        if isinstance(embeddings[0], (str, bytes)):
            raise TypeError
        dimension = len(embeddings[0])
        if any(
            isinstance(vector, (str, bytes)) or len(vector) != dimension
            for vector in embeddings
        ):
            raise FaissIndexError("Embedding vectors must share one dimension")
    except TypeError:
        raise FaissIndexError("Embedding vectors must be numeric sequences") from None
    if dimension <= 0:
        raise FaissIndexError("Embedding vectors must not be empty")
    try:
        matrix = np.array(embeddings, dtype="float32")
    except (TypeError, ValueError, OverflowError):
        raise FaissIndexError("Embedding vectors must contain only finite numbers") from None
    if matrix.ndim != 2 or matrix.shape != (len(embeddings), dimension):
        raise FaissIndexError("Embedding vectors must form a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise FaissIndexError("Embedding vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency check
        raise FaissIndexError("faiss-cpu is required for FAISS retrieval") from exc
    return faiss


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]
