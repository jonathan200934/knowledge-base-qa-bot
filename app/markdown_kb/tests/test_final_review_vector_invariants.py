import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app import indexer, safe_io
from app.faiss_index import FaissIndexError, FaissSectionIndex
from app.vector_index import LocalVectorIndex


class FixedDimensionProvider:
    model = "fixed-dimension-final-review"
    provider_kind = "local-test"
    base_url = "https://local.invalid/v1"

    def __init__(self, *, dimension: int, document_vector: list[float]):
        self.dimension = dimension
        self.requested_dimension = dimension
        self.document_vector = list(document_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self.document_vector) for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return list(self.document_vector)


def _child(content: str = "Focused vector invariant content."):
    parent = indexer.Section(
        id="guide.md#invariant",
        file="guide.md",
        heading="Invariant",
        heading_path=["Guide", "Invariant"],
        content=content,
        tokens=indexer.tokenize(content),
    )
    return indexer.create_child_chunks([parent])[0]


def _local_index(identifier: str, content: str) -> LocalVectorIndex:
    section = SimpleNamespace(
        id=identifier,
        file=f"{identifier}.md",
        heading=identifier,
        heading_path=[identifier],
        content=content,
    )
    return LocalVectorIndex.build([section])


def test_faiss_non_cached_build_rejects_provider_dimension_mismatch():
    provider = FixedDimensionProvider(dimension=3, document_vector=[1.0, 0.0])

    with pytest.raises(FaissIndexError, match="provider dimension"):
        FaissSectionIndex.build([_child()], provider)


def test_empty_faiss_generation_round_trips_at_provider_dimension(tmp_path):
    provider = FixedDimensionProvider(dimension=7, document_vector=[1.0] * 7)
    directory = tmp_path / "faiss-empty"

    built = FaissSectionIndex.build([], provider)
    assert built.index.d == provider.dimension
    assert built.index.ntotal == 0
    assert built.metadata.embedding_dimension == provider.dimension

    built.save(directory)
    loaded = FaissSectionIndex.load(directory, [], provider)

    assert loaded.index.d == provider.dimension
    assert loaded.index.ntotal == 0
    assert loaded.source_ids == []
    assert loaded.metadata.embedding_dimension == provider.dimension


def test_local_vector_load_rejects_zero_norm_for_nonzero_vector(tmp_path):
    artifact = tmp_path / "vector-index.json"
    built = _local_index("nonzero", "meaningful searchable terms")
    built.save(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["records"][0]["vector"]
    assert payload["records"][0]["norm"] > 0
    payload["records"][0]["norm"] = 0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="norm does not match"):
        LocalVectorIndex.load(artifact)


def test_concurrent_local_saves_use_unique_same_parent_temps_and_fsync(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "vector-index.json"
    indexes = [
        _local_index("alpha", "alpha searchable content"),
        _local_index("beta", "beta different content"),
    ]
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    replacements: list[tuple[str, str, int | None, int | None]] = []
    fsynced_modes: list[int] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def synchronized_replace(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        with lock:
            replacements.append(
                (str(source), str(destination), src_dir_fd, dst_dir_fd)
            )
        barrier.wait(timeout=5)
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def recording_fsync(fd):
        with lock:
            fsynced_modes.append(stat.S_IFMT(os.fstat(fd).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr(safe_io.os, "replace", synchronized_replace)
    monkeypatch.setattr(safe_io.os, "fsync", recording_fsync)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(candidate.save, artifact) for candidate in indexes]
        for future in futures:
            future.result(timeout=10)

    assert len(replacements) == 2
    assert len({source for source, _target, _source_fd, _target_fd in replacements}) == 2
    for source, destination, source_fd, target_fd in replacements:
        assert source.startswith(".kb-write-")
        assert source.endswith(".tmp")
        assert destination == artifact.name
        assert source_fd is not None and source_fd == target_fd
    assert fsynced_modes.count(stat.S_IFREG) >= 2
    assert fsynced_modes.count(stat.S_IFDIR) >= 2
    assert not list(tmp_path.glob(".kb-write-*.tmp"))

    loaded = LocalVectorIndex.load(artifact)
    assert loaded.section_ids in (
        indexes[0].section_ids,
        indexes[1].section_ids,
    )


def test_failed_local_atomic_replace_preserves_target_and_cleans_temp(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "vector-index.json"
    _local_index("old", "old complete generation").save(artifact)
    original_bytes = artifact.read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated activation failure")

    monkeypatch.setattr(safe_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="activation failure"):
        _local_index("new", "new candidate generation").save(artifact)

    assert artifact.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".kb-write-*.tmp"))


def test_local_save_rejects_symlink_target_without_touching_destination(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("outside remains unchanged", encoding="utf-8")
    artifact = tmp_path / "vector-index.json"
    artifact.symlink_to(outside)

    with pytest.raises(ValueError, match="regular non-symlink"):
        _local_index("safe", "safe candidate content").save(artifact)

    assert artifact.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside remains unchanged"
    assert not list(tmp_path.glob(".kb-write-*.tmp"))


def test_local_save_rejects_symlink_parent(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    artifact = linked_parent / "vector-index.json"

    with pytest.raises(ValueError, match="ancestors"):
        _local_index("safe", "safe candidate content").save(artifact)

    assert not (real_parent / artifact.name).exists()
    assert not list(real_parent.glob(".kb-write-*.tmp"))
