import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import embedding_cache, hybrid
from .chunking import (
    DEFAULT_CHUNKING_POLICY,
    ChildChunk,
    ChunkingPolicy,
    chunk_sections,
)
from .faiss_index import (
    EmbeddingProvider,
    FaissIndexError,
    FaissSectionIndex,
    OpenAIEmbeddingProvider,
    provider_config_fingerprint,
)
from .safe_io import UnsafeFileError, read_regular_file
from .vector_index import LocalVectorIndex


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
MAX_MARKDOWN_BYTES = 10 * 1024 * 1024
MAX_INDEX_JSON_BYTES = 64 * 1024 * 1024
INDEX_PATH = REPO_ROOT / ".kb" / "index.json"
INDEX_DB_PATH = REPO_ROOT / ".kb" / "index.sqlite3"
VECTOR_INDEX_FILENAME = "vector_index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
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


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
child_chunks: list[ChildChunk] = []
_child_chunking_policy = DEFAULT_CHUNKING_POLICY
section_vector_index: LocalVectorIndex | None = None
faiss_section_index: FaissSectionIndex | None = None
embedding_provider_override: EmbeddingProvider | None = None
_embedding_provider_cache: EmbeddingProvider | None = None
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0
last_index_stats: dict = {}
last_faiss_status: dict = {}

# Corpus lookups are runtime state, not per-query work. Holding the current
# list references allows a constant-time freshness check for replacement or an
# in-place length change without hashing the corpus on every search.
_section_by_id: dict[str, Section] = {}
_child_by_id: dict[str, ChildChunk] = {}
_runtime_sections_ref = sections
_runtime_sections_len = len(sections)
_runtime_child_chunks_ref = child_chunks
_runtime_child_chunks_len = len(child_chunks)
_runtime_generation = 0

# A dense index is validated once for a runtime corpus generation. Repeated
# queries can then avoid section hashing and full compatibility reconstruction.
_vector_runtime_generation: int | None = None
_faiss_runtime_generation: int | None = None
_faiss_runtime_model: str | None = None
_faiss_runtime_config_fingerprint: str | None = None
# A failed FAISS attempt is an explicit local-fallback state for one immutable
# corpus generation. Queries retain that reason and stay local until the corpus
# changes or an explicit build/load is requested.
_faiss_fallback_generation: int | None = None

EXPECTED_FAISS_FAILURES = (
    FaissIndexError,
    OSError,
    ImportError,
    json.JSONDecodeError,
    UnicodeError,
)
FAISS_PROVIDER_UNAVAILABLE_REASON = "FAISS embedding provider is unavailable"
FAISS_BUILD_UNAVAILABLE_REASON = "FAISS index build is unavailable"
FAISS_ARTIFACT_UNAVAILABLE_REASON = "FAISS index artifacts are unavailable"
FAISS_QUERY_FAILED_REASON = "FAISS query failed"


def _mark_faiss_local_fallback() -> None:
    global _faiss_fallback_generation
    _faiss_fallback_generation = _runtime_generation


def _clear_faiss_local_fallback() -> None:
    global _faiss_fallback_generation
    _faiss_fallback_generation = None


def refresh_runtime_state() -> None:
    """Rebuild parent/child maps and advance the corpus generation."""

    global _section_by_id, _child_by_id
    global _runtime_sections_ref, _runtime_sections_len
    global _runtime_child_chunks_ref, _runtime_child_chunks_len
    global _runtime_generation

    _section_by_id = {section.id: section for section in sections}
    _child_by_id = {child.chunk_id: child for child in child_chunks}
    _runtime_sections_ref = sections
    _runtime_sections_len = len(sections)
    _runtime_child_chunks_ref = child_chunks
    _runtime_child_chunks_len = len(child_chunks)
    _runtime_generation += 1


def _ensure_runtime_state_current() -> None:
    if (
        sections is not _runtime_sections_ref
        or len(sections) != _runtime_sections_len
        or child_chunks is not _runtime_child_chunks_ref
        or len(child_chunks) != _runtime_child_chunks_len
    ):
        refresh_runtime_state()


def _corpus_activation_snapshot() -> str:
    """Hash the exact parent/child corpus and chunk policy used by dense retrieval."""

    payload = {
        "sections": [section.to_dict() for section in sections],
        "children": [
            {
                "chunk_id": child.chunk_id,
                "source_id": child.source_id,
                "file": child.file,
                "heading": child.heading,
                "heading_path": child.heading_path,
                "chunk_index": child.chunk_index,
                "chunk_count": child.chunk_count,
                "content": child.content,
                "embedding_text": child.embedding_text,
                "content_hash": child.content_hash,
                "splitter_version": child.splitter_version,
                "heading_prefix_version": child.heading_prefix_version,
            }
            for child in child_chunks
        ],
        "chunking_policy": _child_chunking_policy.to_dict(),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def create_child_chunks(
    section_list: list[Section],
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> list[ChildChunk]:
    """Create internal dense children and retain their vector policy identity."""

    global _child_chunking_policy
    chunks = chunk_sections(section_list, policy)
    _child_chunking_policy = policy
    return chunks


def vector_index_path() -> Path:
    return INDEX_PATH.parent / VECTOR_INDEX_FILENAME


def faiss_index_dir() -> Path:
    return INDEX_PATH.parent / "faiss_index"


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider_cache

    if embedding_provider_override is not None:
        return embedding_provider_override
    if _embedding_provider_cache is None:
        _embedding_provider_cache = OpenAIEmbeddingProvider()
    return _embedding_provider_cache


def rebuild_vector_index(persist: bool = True) -> LocalVectorIndex:
    global section_vector_index, _vector_runtime_generation

    _ensure_runtime_state_current()
    section_vector_index = LocalVectorIndex.build(child_chunks)
    _vector_runtime_generation = _runtime_generation
    if persist:
        section_vector_index.save(vector_index_path())
    return section_vector_index


def load_or_rebuild_vector_index(persist: bool = True) -> LocalVectorIndex | None:
    global section_vector_index, _vector_runtime_generation

    _ensure_runtime_state_current()
    if not child_chunks:
        section_vector_index = None
        _vector_runtime_generation = None
        return None

    path = vector_index_path()
    if path.exists():
        try:
            loaded_index = LocalVectorIndex.load(path)
            if loaded_index.is_compatible(child_chunks):
                section_vector_index = loaded_index
                _vector_runtime_generation = _runtime_generation
                return section_vector_index
        except (OSError, ValueError, json.JSONDecodeError):
            # Stale or corrupt vector metadata should not block startup. Rebuild
            # from the persisted section index, then overwrite the local artifact.
            pass

    return rebuild_vector_index(persist=persist)


def ensure_vector_index_loaded() -> LocalVectorIndex | None:
    global _vector_runtime_generation

    _ensure_runtime_state_current()
    if not child_chunks:
        return None
    if section_vector_index is not None:
        if _vector_runtime_generation == _runtime_generation:
            return section_vector_index
        if section_vector_index.is_compatible(child_chunks):
            _vector_runtime_generation = _runtime_generation
            return section_vector_index
    return load_or_rebuild_vector_index()


def build_faiss_index(persist: bool = True) -> FaissSectionIndex | None:
    global faiss_section_index, last_faiss_status
    global _faiss_runtime_generation, _faiss_runtime_model
    global _faiss_runtime_config_fingerprint

    _ensure_runtime_state_current()
    candidate_generation = _runtime_generation
    candidate_snapshot = _corpus_activation_snapshot()
    candidate_children = list(child_chunks)

    def candidate_is_current() -> bool:
        _ensure_runtime_state_current()
        return (
            _runtime_generation == candidate_generation
            and _corpus_activation_snapshot() == candidate_snapshot
        )

    if not candidate_children:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {"available": False, "status": "empty", "reason": "No indexed sections"}
        return None

    try:
        provider = get_embedding_provider()
    except EXPECTED_FAISS_FAILURES:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "unavailable",
            "reason": FAISS_PROVIDER_UNAVAILABLE_REASON,
        }
        return None

    try:
        built_index = FaissSectionIndex.build(
            candidate_children,
            provider,
            cache_path=embedding_cache.CACHE_PATH,
            chunking_policy=_child_chunking_policy,
            activation_check=candidate_is_current,
        )
        if persist:
            built_index.save(
                faiss_index_dir(), activation_check=candidate_is_current
            )
        if not candidate_is_current():
            raise FaissIndexError(
                "FAISS candidate corpus changed before runtime activation"
            )
        faiss_section_index = built_index
        _faiss_runtime_generation = _runtime_generation
        _faiss_runtime_model = provider.model
        _faiss_runtime_config_fingerprint = provider_config_fingerprint(
            provider,
            built_index.metadata.embedding_dimension,
            _child_chunking_policy,
        )
        last_faiss_status = {
            "available": True,
            "status": "built",
            "embedding_model": built_index.metadata.embedding_model,
            "embedding_dimension": built_index.metadata.embedding_dimension,
            "section_count": built_index.metadata.section_count,
        }
        _clear_faiss_local_fallback()
        return faiss_section_index
    except EXPECTED_FAISS_FAILURES:
        # Expected provider/dependency/artifact failures leave keyword and local
        # vector retrieval available. Status text is static so provider payloads
        # or credentials can never leak through the public index response.
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "unavailable",
            "reason": FAISS_BUILD_UNAVAILABLE_REASON,
        }
        _mark_faiss_local_fallback()
        return None


def load_faiss_index() -> FaissSectionIndex | None:
    global faiss_section_index, last_faiss_status
    global _faiss_runtime_generation, _faiss_runtime_model
    global _faiss_runtime_config_fingerprint

    _ensure_runtime_state_current()
    if not child_chunks:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {"available": False, "status": "empty", "reason": "No indexed sections"}
        return None

    try:
        provider = get_embedding_provider()
    except EXPECTED_FAISS_FAILURES:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "unavailable",
            "reason": FAISS_PROVIDER_UNAVAILABLE_REASON,
        }
        return None

    try:
        loaded_index = FaissSectionIndex.load(
            faiss_index_dir(),
            child_chunks,
            provider,
            chunking_policy=_child_chunking_policy,
        )
        faiss_section_index = loaded_index
        _faiss_runtime_generation = _runtime_generation
        _faiss_runtime_model = provider.model
        _faiss_runtime_config_fingerprint = provider_config_fingerprint(
            provider,
            loaded_index.metadata.embedding_dimension,
            _child_chunking_policy,
        )
        last_faiss_status = {
            "available": True,
            "status": "loaded",
            "embedding_model": loaded_index.metadata.embedding_model,
            "embedding_dimension": loaded_index.metadata.embedding_dimension,
            "section_count": loaded_index.metadata.section_count,
        }
        return faiss_section_index
    except EXPECTED_FAISS_FAILURES:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "unavailable",
            "reason": FAISS_ARTIFACT_UNAVAILABLE_REASON,
        }
        return None


def ensure_faiss_index_loaded() -> FaissSectionIndex | None:
    global faiss_section_index, last_faiss_status
    global _faiss_runtime_generation, _faiss_runtime_model
    global _faiss_runtime_config_fingerprint

    _ensure_runtime_state_current()
    if not child_chunks:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "empty",
            "reason": "No indexed sections",
        }
        return None
    if (
        faiss_section_index is None
        and _faiss_fallback_generation == _runtime_generation
    ):
        return None
    try:
        provider = get_embedding_provider()
    except EXPECTED_FAISS_FAILURES:
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
        last_faiss_status = {
            "available": False,
            "status": "unavailable",
            "reason": FAISS_PROVIDER_UNAVAILABLE_REASON,
        }
        return None

    if faiss_section_index is not None:
        try:
            current_config_fingerprint = provider_config_fingerprint(
                provider,
                faiss_section_index.metadata.embedding_dimension,
                _child_chunking_policy,
            )
        except EXPECTED_FAISS_FAILURES:
            faiss_section_index = None
            _faiss_runtime_generation = None
            _faiss_runtime_model = None
            _faiss_runtime_config_fingerprint = None
            last_faiss_status = {
                "available": False,
                "status": "unavailable",
                "reason": FAISS_PROVIDER_UNAVAILABLE_REASON,
            }
            return None
        if (
            _faiss_runtime_generation == _runtime_generation
            and _faiss_runtime_config_fingerprint == current_config_fingerprint
        ):
            return faiss_section_index
        if faiss_section_index.is_compatible(
            child_chunks,
            provider,
            chunking_policy=_child_chunking_policy,
        ):
            _faiss_runtime_generation = _runtime_generation
            _faiss_runtime_model = provider.model
            _faiss_runtime_config_fingerprint = current_config_fingerprint
            return faiss_section_index
        faiss_section_index = None
        _faiss_runtime_generation = None
        _faiss_runtime_model = None
        _faiss_runtime_config_fingerprint = None
    return load_faiss_index()


def parse_markdown(path: Path) -> list[Section]:
    snapshot = read_regular_file(path, max_bytes=MAX_MARKDOWN_BYTES)
    return parse_markdown_text(
        path.name,
        snapshot.data.decode("utf-8"),
    )


def parse_markdown_text(file_name: str, text: str) -> list[Section]:
    parsed_sections: list[Section] = []
    heading_by_level: dict[int, str] = {}
    source_counts: Counter[str] = Counter()
    current_heading: str | None = None
    current_heading_path: list[str] = []
    current_lines: list[str] = []

    def flush_current_section() -> None:
        nonlocal current_heading, current_heading_path, current_lines
        if current_heading is None:
            current_lines = []
            return

        content = "\n".join(current_lines).strip()
        current_lines = []
        if not content:
            return

        slug = slugify(current_heading)
        source_counts[slug] += 1
        suffix = "" if source_counts[slug] == 1 else f"-{source_counts[slug]}"
        section_id = f"{file_name}#{slug}{suffix}"
        token_text = "\n".join([*current_heading_path, content])
        parsed_sections.append(
            Section(
                id=section_id,
                file=file_name,
                heading=current_heading,
                heading_path=list(current_heading_path),
                content=content,
                tokens=tokenize(token_text),
            )
        )

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush_current_section()
            level = len(match.group(1))
            heading = match.group(2).strip().strip("#").strip()

            for existing_level in list(heading_by_level):
                if existing_level >= level:
                    del heading_by_level[existing_level]
            heading_by_level[level] = heading
            path_parts = [heading_by_level[i] for i in sorted(heading_by_level) if i <= level]

            if level == 1:
                current_heading = None
                current_heading_path = []
            else:
                current_heading = heading
                current_heading_path = path_parts
            continue

        if current_heading is not None:
            current_lines.append(line)

    flush_current_section()
    return parsed_sections


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sections (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            heading TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens TEXT NOT NULL,
            FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            section_id UNINDEXED,
            file_path UNINDEXED,
            heading,
            heading_path,
            content
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sections_file_path ON sections(file_path)")


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    snapshot = read_regular_file(path, max_bytes=MAX_MARKDOWN_BYTES)
    return (
        snapshot.size,
        snapshot.mtime_ns,
        hashlib.sha256(snapshot.data).hexdigest(),
    )


def delete_file_sections(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute(
        "DELETE FROM sections_fts WHERE section_id IN (SELECT id FROM sections WHERE file_path = ?)",
        (file_path,),
    )
    conn.execute("DELETE FROM sections WHERE file_path = ?", (file_path,))


def insert_file_sections(
    conn: sqlite3.Connection,
    file_path: str,
    parsed_sections: list[Section],
    size: int,
    mtime_ns: int,
    content_hash: str,
) -> None:
    indexed_at = time.time_ns()
    delete_file_sections(conn, file_path)
    conn.execute(
        """
        INSERT INTO files(path, size, mtime_ns, content_hash, indexed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size = excluded.size,
            mtime_ns = excluded.mtime_ns,
            content_hash = excluded.content_hash,
            indexed_at = excluded.indexed_at
        """,
        (file_path, size, mtime_ns, content_hash, indexed_at),
    )
    for section in parsed_sections:
        conn.execute(
            """
            INSERT INTO sections(id, file_path, heading, heading_path, content, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                section.id,
                file_path,
                section.heading,
                json.dumps(section.heading_path, ensure_ascii=False),
                section.content,
                json.dumps(section.tokens, ensure_ascii=False),
            ),
        )
        conn.execute(
            """
            INSERT INTO sections_fts(section_id, file_path, heading, heading_path, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                section.id,
                file_path,
                section.heading,
                " > ".join(section.heading_path),
                section.content,
            ),
        )


def section_from_row(row: sqlite3.Row) -> Section:
    return Section(
        id=row["id"],
        file=row["file_path"],
        heading=row["heading"],
        heading_path=json.loads(row["heading_path"]),
        content=row["content"],
        tokens=json.loads(row["tokens"]),
    )


def sections_snapshot(section_list: list[Section]) -> list[dict]:
    return [section.to_dict() for section in section_list]


def read_sections_from_db(index_db_path: Path | None = None) -> list[Section] | None:
    index_db_path = index_db_path or INDEX_DB_PATH
    if not index_db_path.exists():
        return None

    with sqlite3.connect(index_db_path) as conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT id, file_path, heading, heading_path, content, tokens
            FROM sections
            ORDER BY file_path, id
            """
        ).fetchall()
    return [section_from_row(row) for row in rows]


def load_index_db(index_db_path: Path | None = None) -> tuple[int, int]:
    global sections

    db_sections = read_sections_from_db(index_db_path)
    if db_sections is None:
        sections = []
        rebuild_stats()
        load_or_rebuild_vector_index()
        load_faiss_index()
        return 0, 0

    sections = db_sections
    rebuild_stats()
    load_or_rebuild_vector_index()
    load_faiss_index()
    return files_indexed, len(sections)


def write_index_json(index_path: Path | None = None) -> None:
    index_path = index_path or INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": sections_snapshot(sections),
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
            "doc_freq": dict(sorted(doc_freq.items())),
            **last_index_stats,
        },
    }
    temporary_path = index_path.with_name(f".{index_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(index_path)


def rebuild_stats() -> None:
    global child_chunks, doc_freq, avg_doc_len, files_indexed

    files_indexed = len({section.file for section in sections})
    child_chunks = create_child_chunks(sections)
    doc_freq = Counter()
    for section in sections:
        doc_freq.update(set(section.tokens))
    avg_doc_len = (
        sum(len(section.tokens) for section in sections) / len(sections)
        if sections
        else 0.0
    )
    refresh_runtime_state()


def load_index_json(index_path: Path | None = None) -> tuple[int, int]:
    global sections

    index_path = index_path or INDEX_PATH
    if not index_path.exists():
        return load_index_db()

    try:
        snapshot = read_regular_file(index_path, max_bytes=MAX_INDEX_JSON_BYTES)
        payload = json.loads(snapshot.data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Index JSON payload must be an object")
        json_sections = [Section(**item) for item in payload.get("sections", [])]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return load_index_db()

    db_sections = read_sections_from_db()
    if db_sections is not None and sections_snapshot(db_sections) != sections_snapshot(json_sections):
        sections = db_sections
    else:
        sections = json_sections
    rebuild_stats()
    load_or_rebuild_vector_index()
    load_faiss_index()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global last_index_stats

    docs_dir = Path(docs_dir)
    INDEX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    changed_files = 0
    skipped_files = 0
    deleted_files = 0

    with sqlite3.connect(INDEX_DB_PATH) as conn:
        ensure_schema(conn)
        existing = {
            row[0]: {"size": row[1], "mtime_ns": row[2], "content_hash": row[3]}
            for row in conn.execute("SELECT path, size, mtime_ns, content_hash FROM files")
        }
        seen_files: set[str] = set()

        for path in sorted(docs_dir.glob("*.md")):
            try:
                snapshot = read_regular_file(path, max_bytes=MAX_MARKDOWN_BYTES)
            except UnsafeFileError:
                continue

            file_path = path.name
            seen_files.add(file_path)
            size = snapshot.size
            mtime_ns = snapshot.mtime_ns
            content_hash = hashlib.sha256(snapshot.data).hexdigest()
            current = existing.get(file_path)
            if current == {"size": size, "mtime_ns": mtime_ns, "content_hash": content_hash}:
                skipped_files += 1
                continue

            parsed_sections = parse_markdown_text(
                file_path,
                snapshot.data.decode("utf-8"),
            )
            insert_file_sections(
                conn,
                file_path,
                parsed_sections,
                size,
                mtime_ns,
                content_hash,
            )
            changed_files += 1

        for deleted_path in sorted(set(existing) - seen_files):
            delete_file_sections(conn, deleted_path)
            conn.execute("DELETE FROM files WHERE path = ?", (deleted_path,))
            deleted_files += 1

    load_index_db()
    build_faiss_index()
    last_index_stats = {
        "changed_files": changed_files,
        "skipped_files": skipped_files,
        "deleted_files": deleted_files,
        "index_db": str(INDEX_DB_PATH),
        "vector_index": str(vector_index_path()),
        "faiss_index": str(faiss_index_dir()),
        "faiss_status": last_faiss_status,
    }
    write_index_json()

    try:
        from . import filing

        wiki_path = filing.write_wiki_index(sections)
        last_index_stats["wiki_index"] = str(wiki_path)
        write_index_json()
    except Exception:
        # Wiki generation is useful but should not break core indexing.
        pass

    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not sections or avg_doc_len <= 0:
        return 0.0

    token_counts = Counter(section.tokens)
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    doc_len = max(len(section.tokens), 1)
    total_sections = len(sections)
    score = 0.0

    for token in set(query_tokens):
        term_frequency = token_counts[token]
        if term_frequency == 0:
            continue

        frequency = doc_freq[token]
        inverse_doc_frequency = math.log(1 + (total_sections - frequency + 0.5) / (frequency + 0.5))
        denominator = term_frequency + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += inverse_doc_frequency * (term_frequency * (k1 + 1) / denominator)
        if token in heading_tokens:
            score += inverse_doc_frequency * 0.25

    return score


def fts_query_for_tokens(tokens: list[str]) -> str:
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def sqlite_candidate_sections(query_tokens: list[str], limit: int) -> list[Section]:
    if not query_tokens or not INDEX_DB_PATH.exists():
        return []

    query = fts_query_for_tokens(query_tokens)
    if not query:
        return []

    with sqlite3.connect(INDEX_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT s.id, s.file_path, s.heading, s.heading_path, s.content, s.tokens
            FROM sections_fts f
            JOIN sections s ON s.id = f.section_id
            WHERE sections_fts MATCH ?
            ORDER BY bm25(sections_fts), s.id
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [section_from_row(row) for row in rows]


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    candidates = sqlite_candidate_sections(query_tokens, limit=max(k * 8, 20))
    if not candidates:
        candidates = sections

    ranked = [(section, bm25_score(query_tokens, section)) for section in candidates]
    ranked.sort(key=lambda item: (-item[1], item[0].id))
    return [(section, score) for section, score in ranked[:k] if score > 0]


def vector_search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    global faiss_section_index, last_faiss_status
    global _faiss_runtime_generation, _faiss_runtime_model
    global _faiss_runtime_config_fingerprint

    _ensure_runtime_state_current()
    if k <= 0 or not sections or not child_chunks:
        return []

    child_k = min(len(child_chunks), max(k * 4, k + 1, 10))

    def aggregate_parent_hits(
        ranked_ids: list[tuple[str, float]],
    ) -> list[tuple[Section, float]]:
        best_by_parent: dict[str, float] = {}
        for dense_id, score in ranked_ids:
            child = _child_by_id.get(dense_id)
            # A parent identity is accepted only for compatibility with an
            # already-loaded pre-child index; current indexes always resolve a
            # ChildChunk identity here.
            source_id = child.source_id if child is not None else dense_id
            if source_id not in _section_by_id or not math.isfinite(score) or score <= 0:
                continue
            previous = best_by_parent.get(source_id)
            if previous is None or score > previous:
                best_by_parent[source_id] = score

        ranked_parents = [
            (_section_by_id[source_id], score)
            for source_id, score in best_by_parent.items()
        ]
        ranked_parents.sort(key=lambda item: (-item[1], item[0].id))
        return ranked_parents

    def search_until_parent_cutoff_is_complete(search_children) -> list[tuple[Section, float]]:
        requested = child_k
        while True:
            ranked_ids = search_children(requested)
            ranked_parents = aggregate_parent_hits(ranked_ids)
            exhausted = requested >= len(child_chunks) or len(ranked_ids) < requested

            if exhausted:
                return ranked_parents[:k]
            if len(ranked_parents) >= k:
                kth_parent_score = ranked_parents[k - 1][1]
                finite_scores = [score for _dense_id, score in ranked_ids if math.isfinite(score)]
                boundary_score = min(finite_scores) if finite_scores else float("-inf")
                # A strict gap proves that unseen children cannot introduce or
                # reorder a parent at the kth-parent cutoff. Equality requires
                # another bounded fetch so deterministic parent-ID tie-breaking
                # sees the complete cutoff tie.
                if boundary_score < kth_parent_score:
                    return ranked_parents[:k]

            next_requested = min(len(child_chunks), requested * 2)
            if next_requested == requested:
                return ranked_parents[:k]
            requested = next_requested

    faiss_idx = ensure_faiss_index_loaded()
    if faiss_idx is not None:
        try:
            provider = get_embedding_provider()
            return search_until_parent_cutoff_is_complete(
                lambda requested: faiss_idx.search(query, provider, k=requested)
            )
        except EXPECTED_FAISS_FAILURES:
            faiss_section_index = None
            _faiss_runtime_generation = None
            _faiss_runtime_model = None
            _faiss_runtime_config_fingerprint = None
            last_faiss_status = {
                "available": False,
                "status": "query_failed",
                "reason": FAISS_QUERY_FAILED_REASON,
            }

    vector_idx = ensure_vector_index_loaded()
    if vector_idx is None:
        return []

    query_tokens = tokenize(query)
    return search_until_parent_cutoff_is_complete(
        lambda requested: vector_idx.search(query_tokens, k=requested)
    )


def hybrid_rrf_search(query: str, k: int = 3) -> list[hybrid.HybridResult]:
    if not sections:
        return []

    candidate_k = max(k * 4, 10)
    bm25_results = search(query, k=candidate_k)
    vector_results = vector_search(query, k=candidate_k)
    return hybrid.merge_rrf(bm25_results, vector_results, k=k)


def hybrid_search(query: str, k: int = 3) -> list[hybrid.HybridResult]:
    if not sections:
        return []
    from . import reranker

    candidate_k = max(k * 4, 10)
    fused_results = hybrid_rrf_search(query, k=candidate_k)
    return reranker.rerank_hybrid_results(query, fused_results, k=k)
