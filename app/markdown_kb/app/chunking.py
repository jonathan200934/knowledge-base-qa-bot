from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Protocol


DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")
SPLITTER_VERSION = "recursive-character-v1"
HEADING_PREFIX_VERSION = "heading-path-v1"


class SectionLike(Protocol):
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str


@dataclass(frozen=True)
class ChunkingPolicy:
    """Versioned, vector-affecting child-splitting configuration."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    splitter_version: str = SPLITTER_VERSION
    heading_prefix_version: str = HEADING_PREFIX_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_size, int) or isinstance(self.chunk_size, bool) or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(self.chunk_overlap, int) or isinstance(self.chunk_overlap, bool) or self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not isinstance(self.separators, tuple) or not self.separators:
            raise ValueError("separators must be a non-empty tuple")
        if any(not isinstance(separator, str) for separator in self.separators):
            raise ValueError("separators must contain only strings")
        if not self.splitter_version or not self.heading_prefix_version:
            raise ValueError("splitter and heading-prefix versions must be non-empty")

    def to_dict(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
            "splitter_version": self.splitter_version,
            "heading_prefix_version": self.heading_prefix_version,
        }


DEFAULT_CHUNKING_POLICY = ChunkingPolicy()


@dataclass(frozen=True)
class ChildChunk:
    """An internal dense-index record whose public identity remains its parent."""

    chunk_id: str
    source_id: str
    file: str
    heading: str
    heading_path: list[str]
    chunk_index: int
    chunk_count: int
    content: str
    embedding_text: str
    content_hash: str
    splitter_version: str
    heading_prefix_version: str

    @property
    def id(self) -> str:
        """Return the internal identity used by dense indexes."""

        return self.chunk_id

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "content_hash": self.content_hash,
            "splitter_version": self.splitter_version,
            "heading_prefix_version": self.heading_prefix_version,
        }


def normalize_embedding_text(text: str) -> str:
    """Canonicalize Unicode and line endings before splitting or hashing."""

    normalized_newlines = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines)


def heading_context(section: SectionLike) -> str:
    """Render the stable v1 heading-path prefix repeated in every child."""

    parts = [normalize_embedding_text(str(part)).strip() for part in section.heading_path]
    parts = [part for part in parts if part]
    if not parts:
        heading = normalize_embedding_text(str(section.heading)).strip()
        if heading:
            parts = [heading]
    return " > ".join(parts)


def embedded_text(section: SectionLike, child_content: str) -> str:
    context = heading_context(section)
    return f"{context}\n\n{child_content}" if context else child_content


def embedded_content_hash(text: str) -> str:
    """Hash the exact normalized text submitted to an embedding backend."""

    normalized = normalize_embedding_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_content(content: str, policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY) -> list[str]:
    """Split normalized text deterministically with bounded character overlap."""

    text = normalize_embedding_text(content)
    if len(text) <= policy.chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + policy.chunk_size, len(text))
        end = hard_end if hard_end == len(text) else _preferred_split_end(text, start, hard_end, policy)
        chunks.append(text[start:end])
        if end == len(text):
            break
        next_start = end - policy.chunk_overlap
        if next_start <= start:  # Protected by policy validation and split selection.
            raise RuntimeError("child splitter did not make forward progress")
        start = next_start
    return chunks


def chunk_section(
    section: SectionLike,
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> list[ChildChunk]:
    """Create deterministic internal children without mutating the parent."""

    source_id = str(section.id)
    if not source_id:
        raise ValueError("parent section id must be non-empty")
    contents = split_content(section.content, policy)
    chunk_count = len(contents)
    chunks: list[ChildChunk] = []
    for chunk_index, content in enumerate(contents):
        text = embedded_text(section, content)
        chunks.append(
            ChildChunk(
                chunk_id=f"{source_id}::chunk-{chunk_index}",
                source_id=source_id,
                file=str(section.file),
                heading=str(section.heading),
                heading_path=[str(part) for part in section.heading_path],
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                content=content,
                embedding_text=text,
                content_hash=embedded_content_hash(text),
                splitter_version=policy.splitter_version,
                heading_prefix_version=policy.heading_prefix_version,
            )
        )
    return chunks


def chunk_sections(
    sections: Iterable[SectionLike],
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> list[ChildChunk]:
    """Materialize children in canonical parent and ordinal order."""

    chunks = [child for section in sections for child in chunk_section(section, policy)]
    chunk_ids = [child.chunk_id for child in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("child chunk ids must be unique")
    return chunks


def _preferred_split_end(
    text: str,
    start: int,
    hard_end: int,
    policy: ChunkingPolicy,
) -> int:
    # Prefer a configured semantic boundary in the latter half of the window.
    # If none exists, the hard character bound is deterministic and always safe.
    minimum = max(start + policy.chunk_overlap + 1, start + policy.chunk_size // 2)
    for separator in policy.separators:
        if not separator:
            continue
        position = text.rfind(separator, minimum, hard_end)
        if position >= 0:
            boundary = position + len(separator)
            if boundary > start + policy.chunk_overlap:
                return boundary
    return hard_end
