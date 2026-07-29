from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .safe_io import UnsafeFileError, atomic_write_regular_file, read_regular_file


ALGORITHM = "local-tfidf-cosine"
VERSION = 1
TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_VECTOR_INDEX_BYTES = 64 * 1024 * 1024
ROOT_FIELDS = frozenset(
    {"algorithm", "version", "section_count", "vocabulary", "idf", "records"}
)
RECORD_REQUIRED_FIELDS = frozenset(
    {"id", "file", "heading", "heading_path", "token_count", "norm", "vector"}
)
RECORD_OPTIONAL_FIELDS = frozenset({"chunk_id", "source_id", "token_hash"})
RECORD_FIELDS = RECORD_REQUIRED_FIELDS | RECORD_OPTIONAL_FIELDS
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


@dataclass(frozen=True)
class VectorRecord:
    id: str
    source_id: str
    file: str
    heading: str
    heading_path: list[str]
    token_count: int
    token_hash: str
    vector: dict[str, float]
    norm: float

    def to_dict(self) -> dict:
        return {
            # Keep the historical parent-facing ``id`` while recording the
            # actual dense identity explicitly. Retrieval itself uses chunk_id.
            "id": self.source_id,
            "chunk_id": self.id,
            "source_id": self.source_id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "token_count": self.token_count,
            "token_hash": self.token_hash,
            "norm": self.norm,
            "vector": {token: self.vector[token] for token in sorted(self.vector)},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VectorRecord":
        if not isinstance(payload, dict):
            raise ValueError("Vector record must be an object")
        try:
            unknown_fields = set(payload) - RECORD_FIELDS
            if unknown_fields:
                raise ValueError(
                    "Vector record contains unknown fields: "
                    f"{_format_fields(unknown_fields)}"
                )
            missing_fields = RECORD_REQUIRED_FIELDS - set(payload)
            if missing_fields:
                raise ValueError(
                    "Vector record is missing required fields: "
                    f"{_format_fields(missing_fields)}"
                )

            persisted_id = payload["id"]
            record_id = payload.get("chunk_id", persisted_id)
            source_id_is_explicit = "source_id" in payload
            source_id = payload.get("source_id", persisted_id)
            file_name = payload["file"]
            heading = payload["heading"]
            heading_path = payload["heading_path"]
            token_count = payload["token_count"]
            token_hash = payload.get("token_hash", "")
            norm = _finite_float(payload["norm"], "vector record norm")
            vector_payload = payload["vector"]

            if not isinstance(persisted_id, str) or not persisted_id:
                raise ValueError("Persisted vector record id must be a non-empty string")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("Vector record id must be a non-empty string")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Vector record source_id must be a non-empty string")
            if source_id_is_explicit and persisted_id != source_id:
                raise ValueError(
                    "Persisted vector record id must match its explicit source_id"
                )
            if not isinstance(file_name, str):
                raise ValueError("Vector record file must be a string")
            if not isinstance(heading, str):
                raise ValueError("Vector record heading must be a string")
            if not isinstance(heading_path, list):
                raise ValueError("Vector record heading_path must be a list")
            if any(not isinstance(part, str) for part in heading_path):
                raise ValueError("Vector record heading_path entries must be strings")
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError("Vector record token_count must be an integer")
            if token_count < 0:
                raise ValueError("Vector record token_count must be non-negative")
            if not isinstance(token_hash, str):
                raise ValueError("Vector record token_hash must be a string")
            if not isinstance(vector_payload, dict):
                raise ValueError("Vector record vector must be an object")
            if any(
                not isinstance(token, str) or not token for token in vector_payload
            ):
                raise ValueError("Vector record vector keys must be non-empty strings")
            if norm < 0:
                raise ValueError("Vector record norm must be non-negative")
            vector = {
                token: _finite_float(weight, "vector record weight")
                for token, weight in vector_payload.items()
            }
            if any(weight < 0 for weight in vector.values()):
                raise ValueError("Vector record weights must be non-negative")
            return cls(
                id=record_id,
                source_id=source_id,
                file=file_name,
                heading=heading,
                heading_path=list(heading_path),
                token_count=token_count,
                token_hash=token_hash,
                vector=vector,
                norm=norm,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed vector record: {exc}") from exc


class LocalVectorIndex:
    """Deterministic child-level TF-IDF/cosine index.

    Child embeddings use ``ChildChunk.embedding_text`` so deterministic heading
    context participates in dense matching. Legacy section objects remain
    accepted for compatibility with callers that build a standalone index.
    """

    def __init__(self, idf: dict[str, float] | None = None, records: Sequence[VectorRecord] | None = None):
        self.idf = dict(idf or {})
        self.records = list(records or [])
        self._records_by_id = {record.id: record for record in self.records}
        if len(self._records_by_id) != len(self.records):
            raise ValueError("Vector index contains duplicate record ids")

    @classmethod
    def build(cls, sections: Iterable[object]) -> "LocalVectorIndex":
        materialized_sections = list(sections)
        section_counts: list[tuple[object, Counter[str]]] = []
        document_frequency: Counter[str] = Counter()

        for section in materialized_sections:
            counts = Counter(_embedding_tokens(section))
            section_counts.append((section, counts))
            document_frequency.update(counts.keys())

        total_sections = len(materialized_sections)
        idf = {
            token: math.log((total_sections + 1) / (frequency + 1)) + 1.0
            for token, frequency in sorted(document_frequency.items())
        }

        records: list[VectorRecord] = []
        for section, counts in section_counts:
            vector = _tf_idf_vector(counts, idf)
            records.append(
                VectorRecord(
                    id=str(getattr(section, "chunk_id", getattr(section, "id"))),
                    source_id=str(
                        getattr(section, "source_id", getattr(section, "id"))
                    ),
                    file=getattr(section, "file", ""),
                    heading=getattr(section, "heading", ""),
                    heading_path=list(getattr(section, "heading_path", []) or []),
                    token_count=sum(counts.values()),
                    token_hash=_token_hash(counts),
                    vector=vector,
                    norm=_norm(vector),
                )
            )

        records.sort(key=lambda record: record.id)
        return cls(idf=idf, records=records)

    @classmethod
    def load(cls, path: Path) -> "LocalVectorIndex":
        snapshot = read_regular_file(path, max_bytes=MAX_VECTOR_INDEX_BYTES)
        try:
            payload = json.loads(
                snapshot.data.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except RecursionError as exc:
            raise ValueError("Vector index JSON nesting is too deep") from exc
        idf, records = _validate_payload(payload)
        return cls(idf=idf, records=records)

    def save(self, path: Path) -> None:
        path = Path(path)
        try:
            payload = {
                "algorithm": ALGORITHM,
                "version": VERSION,
                "section_count": len(self.records),
                "vocabulary": sorted(self.idf),
                "idf": {token: self.idf[token] for token in sorted(self.idf)},
                "records": [record.to_dict() for record in self.records],
            }
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Malformed in-memory vector index") from exc

        _validate_payload(payload)
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise ValueError("Vector index payload is not JSON serializable") from exc
        if len(encoded) > MAX_VECTOR_INDEX_BYTES:
            raise UnsafeFileError("vector index output exceeds the configured size limit")
        atomic_write_regular_file(
            path,
            encoded,
            create_parents=True,
            file_mode=0o600,
        )

    @property
    def section_ids(self) -> set[str]:
        return set(self._records_by_id)

    def is_compatible(self, sections_or_ids: Iterable[object]) -> bool:
        expected = list(sections_or_ids)
        if not expected:
            return not self.records

        if all(isinstance(item, str) for item in expected):
            expected_ids = [str(item) for item in expected]
            return len(expected_ids) == len(self.records) and set(expected_ids) == self.section_ids

        if any(not isinstance(getattr(section, "id", None), str) or not getattr(section, "id") for section in expected):
            return False

        try:
            expected_index = LocalVectorIndex.build(expected)
        except (TypeError, ValueError):
            return False

        return self._matches(expected_index)

    def _matches(self, expected_index: "LocalVectorIndex") -> bool:
        if self.section_ids != expected_index.section_ids:
            return False
        if not _same_float_dict(self.idf, expected_index.idf):
            return False

        for expected_record in expected_index.records:
            record = self._records_by_id.get(expected_record.id)
            if record is None:
                return False
            if record.source_id != expected_record.source_id:
                return False
            if record.file != expected_record.file:
                return False
            if record.heading != expected_record.heading:
                return False
            if record.heading_path != expected_record.heading_path:
                return False
            if record.token_count != expected_record.token_count:
                return False
            if record.token_hash != expected_record.token_hash:
                return False
            if not _same_float(record.norm, expected_record.norm):
                return False
            if not _same_float_dict(record.vector, expected_record.vector):
                return False
        return True

    def search(self, query_tokens: Sequence[str], k: int = 3) -> list[tuple[str, float]]:
        if k <= 0 or not query_tokens or not self.records:
            return []

        query_counts = Counter(token for token in query_tokens if token in self.idf)
        query_vector = _tf_idf_vector(query_counts, self.idf)
        query_norm = _norm(query_vector)
        if query_norm <= 0:
            return []

        ranked: list[tuple[str, float]] = []
        query_terms = set(query_vector)
        for record in self.records:
            if record.norm <= 0:
                continue
            dot_product = sum(query_vector[token] * record.vector.get(token, 0.0) for token in query_terms)
            score = dot_product / (query_norm * record.norm) if dot_product > 0 else 0.0
            if score > 0 and math.isfinite(score):
                ranked.append((record.id, score))

        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]


def _token_hash(counts: Counter[str]) -> str:
    payload = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _embedding_tokens(section: object) -> list[str]:
    """Tokenize the exact text embedded for a child chunk.

    Parent ``Section`` objects remain supported for standalone callers and
    safe loading/rebuilding of indexes written before child chunking.
    """

    embedding_text = getattr(section, "embedding_text", None)
    if embedding_text is None:
        heading_path = " > ".join(
            str(part) for part in getattr(section, "heading_path", []) or []
        )
        content = str(getattr(section, "content", ""))
        embedding_text = f"{heading_path}\n\n{content}".strip()
    return [
        token
        for token in TOKEN_RE.findall(str(embedding_text).lower())
        if token not in STOP_WORDS
    ]


def _tf_idf_vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    if not counts:
        return {}
    max_frequency = max(counts.values()) or 1
    return {
        token: (frequency / max_frequency) * idf[token]
        for token, frequency in sorted(counts.items())
        if token in idf and frequency > 0
    }


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _format_fields(fields: Iterable[object]) -> str:
    return ", ".join(sorted(repr(field) for field in fields))


def _validate_payload(payload: object) -> tuple[dict[str, float], list[VectorRecord]]:
    if not isinstance(payload, dict):
        raise ValueError("Vector index payload must be an object")

    unknown_fields = set(payload) - ROOT_FIELDS
    if unknown_fields:
        raise ValueError(
            "Vector index payload contains unknown fields: "
            f"{_format_fields(unknown_fields)}"
        )
    missing_fields = ROOT_FIELDS - set(payload)
    if missing_fields:
        raise ValueError(
            "Vector index payload is missing required fields: "
            f"{_format_fields(missing_fields)}"
        )

    if payload["algorithm"] != ALGORITHM:
        raise ValueError(
            f"Unsupported vector index algorithm: {payload['algorithm']!r}"
        )
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != VERSION:
        raise ValueError(f"Unsupported vector index version: {version!r}")

    records_payload = payload["records"]
    idf_payload = payload["idf"]
    vocabulary_payload = payload["vocabulary"]
    section_count = payload["section_count"]
    if not isinstance(records_payload, list):
        raise ValueError("Vector index records must be a list")
    if not isinstance(idf_payload, dict):
        raise ValueError("Vector index idf must be an object")
    if not isinstance(vocabulary_payload, list):
        raise ValueError("Vector index vocabulary must be a list")
    if isinstance(section_count, bool) or not isinstance(section_count, int):
        raise ValueError("Vector index section_count must be an integer")
    if section_count < 0:
        raise ValueError("Vector index section_count must be non-negative")

    if any(
        not isinstance(token, str) or not token for token in vocabulary_payload
    ):
        raise ValueError(
            "Vector index vocabulary entries must be non-empty strings"
        )
    vocabulary = list(vocabulary_payload)
    if vocabulary != sorted(vocabulary) or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Vector index vocabulary must be sorted and unique")
    if any(not isinstance(token, str) or not token for token in idf_payload):
        raise ValueError("Vector index idf keys must be non-empty strings")

    idf = {
        token: _finite_float(weight, "vector index idf weight")
        for token, weight in idf_payload.items()
    }
    if any(weight < 0 for weight in idf.values()):
        raise ValueError("Vector index idf weights must be non-negative")
    if set(vocabulary) != set(idf):
        raise ValueError("Vector index vocabulary and idf keys differ")

    records = [VectorRecord.from_dict(record) for record in records_payload]
    if section_count != len(records):
        raise ValueError("Vector index section_count does not match records")
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Vector index contains duplicate record ids")

    for record in records:
        if any(token not in idf for token in record.vector):
            raise ValueError("Vector record contains token outside vocabulary")
        if not _same_float(record.norm, _norm(record.vector)):
            raise ValueError("Vector record norm does not match vector weights")
    return idf, records


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _same_float(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _same_float_dict(left: dict[str, float], right: dict[str, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(_same_float(left[token], right[token]) for token in left)
