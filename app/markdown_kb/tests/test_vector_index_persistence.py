import json
import os
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import safe_io, vector_index
from app.safe_io import UnsafeFileError
from app.vector_index import LocalVectorIndex, VectorRecord


def _local_index() -> LocalVectorIndex:
    section = SimpleNamespace(
        id="guide.md#topic",
        file="guide.md",
        heading="Topic",
        heading_path=["Guide", "Topic"],
        content="searchable topic content",
    )
    return LocalVectorIndex.build([section])


def _valid_payload() -> dict:
    return {
        "algorithm": vector_index.ALGORITHM,
        "version": vector_index.VERSION,
        "section_count": 1,
        "vocabulary": ["term"],
        "idf": {"term": 1.0},
        "records": [
            {
                "id": "guide.md#topic",
                "chunk_id": "guide.md#topic::chunk-0",
                "source_id": "guide.md#topic",
                "file": "guide.md",
                "heading": "Topic",
                "heading_path": ["Guide", "Topic"],
                "token_count": 1,
                "token_hash": "",
                "norm": 1.0,
                "vector": {"term": 1.0},
            }
        ],
    }


def _write_payload(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("bad_version", [True, 1.0, "1"])
def test_load_rejects_non_integer_version_types(tmp_path, bad_version):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["version"] = bad_version
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="version"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize("bad_count", [True, 1.0, "1"])
def test_load_rejects_coerced_section_count_types(tmp_path, bad_count):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["section_count"] = bad_count
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="section_count"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize("bad_count", [True, 1.0, "1"])
def test_load_rejects_coerced_record_token_count_types(tmp_path, bad_count):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["records"][0]["token_count"] = bad_count
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="Malformed vector record"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("id", True),
        ("chunk_id", True),
        ("source_id", True),
        ("heading", True),
        ("heading_path", ["Guide", True]),
    ],
)
def test_load_rejects_coerced_record_string_types(tmp_path, field, bad_value):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["records"][0][field] = bad_value
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="Malformed vector record"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize("bad_weight", [True, "1.0"])
def test_load_rejects_coerced_idf_weight_types(tmp_path, bad_weight):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["idf"]["term"] = bad_weight
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="finite number"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize("bad_weight", [True, "1.0"])
def test_load_rejects_coerced_vector_weight_types(tmp_path, bad_weight):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["records"][0]["vector"]["term"] = bad_weight
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="Malformed vector record"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize(
    ("vocabulary", "idf"),
    [
        ([1], {"1": 1.0}),
        ([""], {"": 1.0}),
        (["term", "term"], {"term": 1.0}),
    ],
)
def test_load_requires_unique_nonempty_string_vocabulary_entries(
    tmp_path, vocabulary, idf
):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["section_count"] = 0
    payload["records"] = []
    payload["vocabulary"] = vocabulary
    payload["idf"] = idf
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="vocabulary"):
        LocalVectorIndex.load(artifact)


def test_vector_record_rejects_non_string_vector_keys():
    record = deepcopy(_valid_payload()["records"][0])
    record["vector"] = {1: 1.0}

    with pytest.raises(ValueError, match="Malformed vector record"):
        VectorRecord.from_dict(record)


def test_load_accepts_finite_json_integer_and_float_weights(tmp_path):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["idf"]["term"] = 2
    payload["records"][0]["vector"]["term"] = 2.0
    payload["records"][0]["norm"] = 2
    _write_payload(artifact, payload)

    loaded = LocalVectorIndex.load(artifact)

    assert loaded.idf == {"term": 2.0}
    assert loaded.records[0].vector == {"term": 2.0}
    assert loaded.records[0].norm == 2.0


def test_load_preserves_intentional_legacy_record_defaults(tmp_path):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    legacy_record = payload["records"][0]
    legacy_record.pop("chunk_id")
    legacy_record.pop("source_id")
    legacy_record.pop("token_hash")
    _write_payload(artifact, payload)

    loaded = LocalVectorIndex.load(artifact)

    assert loaded.records[0].id == legacy_record["id"]
    assert loaded.records[0].source_id == legacy_record["id"]
    assert loaded.records[0].token_hash == ""


def test_load_rejects_modern_record_with_mismatched_parent_identity(tmp_path):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    payload["records"][0]["source_id"] = "other.md#topic"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="Malformed vector record"):
        LocalVectorIndex.load(artifact)


def test_compatibility_includes_parent_source_identity():
    section = SimpleNamespace(
        id="guide.md#topic",
        chunk_id="guide.md#topic::chunk-0",
        source_id="guide.md#topic",
        file="guide.md",
        heading="Topic",
        heading_path=["Guide", "Topic"],
        embedding_text="searchable topic content",
    )
    expected = LocalVectorIndex.build([section])
    corrupted_record = replace(expected.records[0], source_id="other.md#topic")
    corrupted = LocalVectorIndex(idf=expected.idf, records=[corrupted_record])

    assert not corrupted.is_compatible([section])


@pytest.mark.parametrize("location", ["root", "record"])
def test_load_rejects_unknown_schema_keys(tmp_path, location):
    artifact = tmp_path / "vector-index.json"
    payload = _valid_payload()
    target = payload if location == "root" else payload["records"][0]
    target["unexpected"] = "not part of the persisted schema"
    _write_payload(artifact, payload)

    with pytest.raises(ValueError, match="unknown"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize("location", ["root", "record", "vector"])
def test_load_rejects_duplicate_json_object_keys_at_any_depth(tmp_path, location):
    artifact = tmp_path / "vector-index.json"
    encoded = json.dumps(_valid_payload())
    if location == "root":
        encoded = encoded.replace("{", '{"version": 1, ', 1)
    elif location == "record":
        encoded = encoded.replace(
            '"records": [{',
            '"records": [{"id": "duplicate", ',
            1,
        )
    else:
        encoded = encoded.replace(
            '"vector": {',
            '"vector": {"term": 1.0, ',
            1,
        )
    artifact.write_text(encoded, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        LocalVectorIndex.load(artifact)


@pytest.mark.parametrize(
    ("location", "bad_number"),
    [
        ("idf", float("nan")),
        ("norm", float("inf")),
        ("vector", float("-inf")),
    ],
)
def test_save_rejects_nonfinite_in_memory_numbers_before_filesystem_mutation(
    tmp_path, location, bad_number
):
    artifact = tmp_path / "missing" / "vector-index.json"
    candidate = _local_index()
    record = candidate.records[0]
    if location == "idf":
        token = next(iter(candidate.idf))
        candidate.idf[token] = bad_number
    elif location == "norm":
        candidate.records = [replace(record, norm=bad_number)]
    else:
        token = next(iter(record.vector))
        candidate.records = [
            replace(record, vector={**record.vector, token: bad_number})
        ]

    with pytest.raises(ValueError, match="finite number"):
        candidate.save(artifact)

    assert not artifact.parent.exists()


def test_save_rejects_malformed_in_memory_record_before_filesystem_mutation(
    tmp_path,
):
    artifact = tmp_path / "missing" / "vector-index.json"
    candidate = _local_index()
    candidate.records = [replace(candidate.records[0], heading_path=["Guide", 7])]

    with pytest.raises(ValueError, match="Malformed vector record"):
        candidate.save(artifact)

    assert not artifact.parent.exists()


def test_save_rejects_oversized_encoding_before_creating_parents(monkeypatch, tmp_path):
    artifact = tmp_path / "missing" / "nested" / "vector-index.json"
    monkeypatch.setattr(vector_index, "MAX_VECTOR_INDEX_BYTES", 1)

    with pytest.raises(ValueError, match="size limit"):
        _local_index().save(artifact)

    assert not (tmp_path / "missing").exists()


def test_save_rejects_symlinked_ancestor_before_creating_descendants(tmp_path):
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    linked_ancestor = tmp_path / "linked"
    linked_ancestor.symlink_to(attacker_directory, target_is_directory=True)
    artifact = linked_ancestor / "new-parent" / "vector-index.json"

    with pytest.raises(UnsafeFileError, match="ancestors"):
        _local_index().save(artifact)

    assert list(attacker_directory.iterdir()) == []


def test_save_cannot_be_redirected_by_ancestor_swap(monkeypatch, tmp_path):
    requested_ancestor = tmp_path / "raced"
    requested_ancestor.mkdir()
    moved_ancestor = tmp_path / "moved"
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    artifact = requested_ancestor / "nested" / "vector-index.json"
    real_open = safe_io.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        path_text = os.fspath(path)
        if not swapped and dir_fd is not None and path_text == requested_ancestor.name:
            requested_ancestor.rename(moved_ancestor)
            requested_ancestor.symlink_to(attacker_directory, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_io.os, "open", racing_open)

    with pytest.raises(UnsafeFileError, match="ancestors"):
        _local_index().save(artifact)

    assert swapped
    assert list(attacker_directory.iterdir()) == []
