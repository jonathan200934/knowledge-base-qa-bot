import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from app import faiss_index, indexer
from app.faiss_index import FaissIndexError, FaissSectionIndex, HashEmbeddingProvider


def _parent_and_child(content: str = "Safe persisted answer."):
    parent = indexer.Section(
        id="guide.md#answer",
        file="guide.md",
        heading="Answer",
        heading_path=["Guide", "Answer"],
        content=content,
        tokens=indexer.tokenize(content),
    )
    return parent, indexer.create_child_chunks([parent])[0]


def _save_generation(directory: Path, content: str = "Safe persisted answer."):
    _parent, child = _parent_and_child(content)
    provider = HashEmbeddingProvider("safe-artifact-test")
    built = FaissSectionIndex.build([child], provider)
    built.save(directory)
    return child, provider, built


def _assert_rejected_before_native_parse(monkeypatch, directory, children, provider):
    calls = []
    real_faiss = faiss_index._require_faiss()

    class GuardedFaiss:
        def __getattr__(self, name):
            return getattr(real_faiss, name)

        def read_index(self, *_args, **_kwargs):
            calls.append("read_index")
            raise AssertionError("unsafe artifact reached faiss.read_index")

        def deserialize_index(self, *_args, **_kwargs):
            calls.append("deserialize_index")
            raise AssertionError("unsafe artifact reached FAISS deserialization")

    monkeypatch.setattr(faiss_index, "_require_faiss", lambda: GuardedFaiss())
    with pytest.raises(FaissIndexError):
        FaissSectionIndex.load(directory, children, provider)
    assert calls == []


def test_save_writes_strict_non_pickle_generation_and_checksums(tmp_path):
    directory = tmp_path / "faiss_index"
    child, _provider, _built = _save_generation(directory)

    assert {entry.name for entry in directory.iterdir()} == {
        "index.faiss",
        "children.json",
        "manifest.json",
    }
    assert not list(directory.rglob("*.pkl"))
    children_bytes = (directory / "children.json").read_bytes()
    child_payload = json.loads(children_bytes)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert child_payload == [
        {
            "chunk_id": child.chunk_id,
            "content_hash": child.content_hash,
            "file": child.file,
            "section_hash": faiss_index.section_hash(child),
            "source_id": child.source_id,
        }
    ]
    assert manifest["schema"] == "markdown-kb-faiss-manifest"
    assert manifest["index_version"] == faiss_index.FAISS_INDEX_VERSION
    assert manifest["provider"]["model"] == "safe-artifact-test"
    assert "api_key" not in json.dumps(manifest).lower()
    assert manifest["index"]["type"] == "IndexFlatIP"
    assert manifest["index"]["metric"] == "inner_product"
    assert manifest["index"]["normalization"] == "l2"
    assert manifest["counts"] == {"chunks": 1, "files": 1, "sections": 1}
    assert manifest["payloads"]["children.json"]["sha256"] == hashlib.sha256(children_bytes).hexdigest()
    assert manifest["payloads"]["index.faiss"]["sha256"] == hashlib.sha256(
        (directory / "index.faiss").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "unsafe_case",
    ["directory_symlink", "file_symlink", "non_regular_file", "unexpected_entry"],
)
def test_unsafe_layout_is_rejected_before_faiss_parse(monkeypatch, tmp_path, unsafe_case):
    safe_directory = tmp_path / "safe"
    child, provider, _built = _save_generation(safe_directory)
    directory = safe_directory

    if unsafe_case == "directory_symlink":
        directory = tmp_path / "linked"
        directory.symlink_to(safe_directory, target_is_directory=True)
    elif unsafe_case == "file_symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("[]", encoding="utf-8")
        (safe_directory / "children.json").unlink()
        (safe_directory / "children.json").symlink_to(outside)
    elif unsafe_case == "non_regular_file":
        manifest_path = safe_directory / "manifest.json"
        manifest_path.unlink()
        os.mkfifo(manifest_path)
        real_open = os.open

        def require_nonblocking_open(path, flags, *args, **kwargs):
            if Path(path) == manifest_path and not flags & os.O_NONBLOCK:
                raise AssertionError("non-regular artifacts must never be opened in blocking mode")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(faiss_index.os, "open", require_nonblocking_open)
    else:
        (safe_directory / "unexpected.bin").write_bytes(b"unexpected")

    _assert_rejected_before_native_parse(monkeypatch, directory, [child], provider)


@pytest.mark.parametrize("unsafe_case", ["oversized_manifest", "duplicate_key", "nonfinite_json", "checksum_mismatch"])
def test_unsafe_metadata_is_rejected_before_faiss_parse(monkeypatch, tmp_path, unsafe_case):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory)
    manifest_path = directory / "manifest.json"
    children_path = directory / "children.json"

    if unsafe_case == "oversized_manifest":
        manifest_path.write_bytes(b" " * (faiss_index.MAX_MANIFEST_BYTES + 1))
    elif unsafe_case == "duplicate_key":
        manifest_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    elif unsafe_case == "nonfinite_json":
        children_path.write_text('[{"score":NaN}]', encoding="utf-8")
    else:
        index_bytes = (directory / "index.faiss").read_bytes()
        (directory / "index.faiss").write_bytes(index_bytes + b"tampered")

    _assert_rejected_before_native_parse(monkeypatch, directory, [child], provider)


def test_manifest_compatibility_mismatches_are_rejected_before_faiss_parse(monkeypatch, tmp_path):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory)

    changed = replace(child, embedding_text=child.embedding_text + " changed")
    _assert_rejected_before_native_parse(monkeypatch, directory, [changed], provider)


@pytest.mark.parametrize("mutation", ["missing_field", "unknown_field", "provider", "count"])
def test_strict_manifest_mutations_are_rejected_before_faiss_parse(
    monkeypatch, tmp_path, mutation
):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing_field":
        del manifest["corpus_fingerprint"]
    elif mutation == "unknown_field":
        manifest["unexpected"] = True
    elif mutation == "provider":
        manifest["provider"]["model"] = "stale-model"
    else:
        manifest["counts"]["sections"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _assert_rejected_before_native_parse(monkeypatch, directory, [child], provider)


def test_unsafe_child_file_path_is_rejected_before_faiss_parse(monkeypatch, tmp_path):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory)
    children_path = directory / "children.json"
    manifest_path = directory / "manifest.json"
    records = json.loads(children_path.read_text(encoding="utf-8"))
    records[0]["file"] = "../outside.md"
    children_bytes = faiss_index._canonical_json_bytes(records)
    children_path.write_bytes(children_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    children_sha = hashlib.sha256(children_bytes).hexdigest()
    manifest["ordered_mapping_sha256"] = children_sha
    manifest["corpus_fingerprint"] = children_sha
    manifest["payloads"]["children.json"] = {
        "sha256": children_sha,
        "bytes": len(children_bytes),
    }
    manifest_path.write_bytes(faiss_index._canonical_json_bytes(manifest))

    _assert_rejected_before_native_parse(monkeypatch, directory, [child], provider)


def test_native_index_type_metric_dimension_and_count_are_checked_after_parse(monkeypatch, tmp_path):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory)
    real_faiss = faiss_index._require_faiss()
    valid = real_faiss.read_index(str(directory / "index.faiss"))

    class InvalidIndex:
        d = valid.d + 1
        ntotal = valid.ntotal
        metric_type = valid.metric_type

    class ParsingFaiss:
        METRIC_INNER_PRODUCT = real_faiss.METRIC_INNER_PRODUCT
        IndexFlatIP = real_faiss.IndexFlatIP

        @staticmethod
        def deserialize_index(_payload):
            return InvalidIndex()

    monkeypatch.setattr(faiss_index, "_require_faiss", lambda: ParsingFaiss())
    with pytest.raises(FaissIndexError, match="dimension"):
        FaissSectionIndex.load(directory, [child], provider)


def test_failed_atomic_activation_preserves_previous_complete_generation(monkeypatch, tmp_path):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory, "Old complete answer.")
    old_payloads = {path.name: path.read_bytes() for path in directory.iterdir()}

    _new_parent, new_child = _parent_and_child("New candidate answer.")
    candidate = FaissSectionIndex.build([new_child], provider)

    def fail_exchange(_source, _destination):
        raise OSError("simulated atomic exchange failure")

    monkeypatch.setattr(faiss_index, "_atomic_exchange_directories", fail_exchange)
    with pytest.raises(FaissIndexError, match="activate"):
        candidate.save(directory)

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == old_payloads
    loaded = FaissSectionIndex.load(directory, [child], provider)
    assert loaded.source_ids == [child.chunk_id]


@pytest.mark.parametrize("failure_phase", ["write", "flush", "validation"])
def test_pre_activation_failures_preserve_previous_generation(
    monkeypatch, tmp_path, failure_phase
):
    directory = tmp_path / "faiss_index"
    child, provider, _built = _save_generation(directory, "Old complete answer.")
    old_payloads = {path.name: path.read_bytes() for path in directory.iterdir()}
    _parent, new_child = _parent_and_child("New candidate answer.")
    candidate = FaissSectionIndex.build([new_child], provider)

    if failure_phase == "write":
        real_faiss = faiss_index._require_faiss()

        class FailingFaiss:
            def __getattr__(self, name):
                return getattr(real_faiss, name)

            def write_index(self, *_args, **_kwargs):
                raise OSError("simulated write failure")

        monkeypatch.setattr(faiss_index, "_require_faiss", lambda: FailingFaiss())
    elif failure_phase == "flush":
        real_fsync_directory = faiss_index._fsync_directory

        def fail_candidate_flush(path):
            if Path(path).name.startswith(f".{directory.name}.tmp-"):
                raise OSError("simulated flush failure")
            return real_fsync_directory(path)

        monkeypatch.setattr(faiss_index, "_fsync_directory", fail_candidate_flush)
    else:
        real_load = faiss_index._load_validated_generation

        def fail_candidate_validation(path, expected):
            if Path(path).name.startswith(f".{directory.name}.tmp-"):
                raise FaissIndexError("simulated candidate validation failure")
            return real_load(path, expected)

        monkeypatch.setattr(faiss_index, "_load_validated_generation", fail_candidate_validation)

    with pytest.raises(FaissIndexError):
        candidate.save(directory)

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == old_payloads
    assert not list(tmp_path.glob(f".{directory.name}.tmp-*"))
    loaded = FaissSectionIndex.load(directory, [child], provider)
    assert loaded.source_ids == [child.chunk_id]


def test_post_activation_cleanup_failure_keeps_new_generation_active(monkeypatch, tmp_path):
    directory = tmp_path / "faiss_index"
    _old_child, provider, _built = _save_generation(directory, "Old complete answer.")
    _new_parent, new_child = _parent_and_child("New complete answer.")
    candidate = FaissSectionIndex.build([new_child], provider)
    real_rmtree = faiss_index.shutil.rmtree

    def fail_old_generation_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(f".{directory.name}.tmp-"):
            raise OSError("simulated post-activation cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(faiss_index.shutil, "rmtree", fail_old_generation_cleanup)

    candidate.save(directory)

    loaded = FaissSectionIndex.load(directory, [new_child], provider)
    assert loaded.source_ids == [new_child.chunk_id]


def test_generation_load_is_bound_to_opened_directory_inode(monkeypatch, tmp_path):
    class AxisProvider:
        model = "same-provider-identity"
        dimension = 2
        requested_dimension = 2
        provider_kind = "local-axis-test"
        base_url = "https://local.invalid/v1"

        def __init__(self, document_vector):
            self.document_vector = document_vector

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [list(self.document_vector) for _text in texts]

        def embed_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    directory = tmp_path / "faiss_index"
    replacement = tmp_path / "replacement"
    displaced = tmp_path / "displaced"
    _parent, child = _parent_and_child("Same corpus metadata.")
    expected_provider = AxisProvider([1.0, 0.0])
    replacement_provider = AxisProvider([0.0, 1.0])
    FaissSectionIndex.build([child], expected_provider).save(directory)
    FaissSectionIndex.build([child], replacement_provider).save(replacement)
    real_open_generation = getattr(faiss_index, "_open_generation_dirfd")
    swapped = False

    def open_then_swap(path):
        nonlocal swapped
        fd = real_open_generation(path)
        if Path(path) == directory and not swapped:
            swapped = True
            os.replace(directory, displaced)
            os.replace(replacement, directory)
        return fd

    monkeypatch.setattr(faiss_index, "_open_generation_dirfd", open_then_swap)

    loaded = FaissSectionIndex.load(directory, [child], expected_provider)

    assert loaded.search("query", expected_provider, k=1) == [(child.chunk_id, 1.0)]


def test_corpus_change_during_candidate_validation_prevents_activation(monkeypatch):
    parent, child = _parent_and_child("Candidate corpus answer.")
    indexer.sections = [parent]
    indexer.child_chunks = [child]
    indexer.refresh_runtime_state()
    indexer.embedding_provider_override = HashEmbeddingProvider("activation-snapshot-test")
    real_load = faiss_index._load_validated_generation

    def validate_then_mutate(path, expected):
        loaded = real_load(path, expected)
        if Path(path).name.startswith(".faiss_index.tmp-"):
            changed_parent, changed_child = _parent_and_child("Changed during validation.")
            indexer.sections = [changed_parent]
            indexer.child_chunks = [changed_child]
            indexer.refresh_runtime_state()
        return loaded

    monkeypatch.setattr(faiss_index, "_load_validated_generation", validate_then_mutate)

    assert indexer.build_faiss_index(persist=True) is None
    assert indexer.faiss_section_index is None
    assert not indexer.faiss_index_dir().exists()
    assert indexer.last_faiss_status == {
        "available": False,
        "status": "unavailable",
        "reason": indexer.FAISS_BUILD_UNAVAILABLE_REASON,
    }
