import errno
import os
import stat
from types import SimpleNamespace

import pytest

from app import filing, safe_io
from app.safe_io import UnsafeFileError, read_regular_file


def test_read_regular_file_rejects_ctime_only_identity_change(monkeypatch, tmp_path):
    document = tmp_path / "document.md"
    document.write_bytes(b"same-size contents")
    real_fstat = safe_io.os.fstat
    regular_file_stats = 0

    def ctime_changing_fstat(fd):
        nonlocal regular_file_stats
        result = real_fstat(fd)
        if not stat.S_ISREG(result.st_mode):
            return result

        regular_file_stats += 1
        if regular_file_stats == 2:
            # A same-size mutation followed by restoring mtime still changes
            # ctime. Model that identity change deterministically.
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_dev=result.st_dev,
                st_ino=result.st_ino,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_ctime_ns=result.st_ctime_ns + 1,
            )
        return result

    monkeypatch.setattr(safe_io.os, "fstat", ctime_changing_fstat)

    with pytest.raises(OSError, match="file changed while it was being read"):
        read_regular_file(document, max_bytes=1_024)


def test_read_regular_file_closes_target_before_parent_context_exit(
    monkeypatch, tmp_path
):
    document = tmp_path / "document.md"
    document.write_bytes(b"contents")
    target_fd = os.open(document, os.O_RDONLY)

    class ParentCloseError(OSError):
        pass

    class RaisingParentContext:
        def __enter__(self):
            return -1, document.name

        def __exit__(self, *_args):
            raise ParentCloseError("simulated parent close failure")

    monkeypatch.setattr(
        safe_io,
        "_open_parent_directory",
        lambda *_args, **_kwargs: RaisingParentContext(),
    )
    monkeypatch.setattr(
        safe_io,
        "_open_read_only_regular_at",
        lambda _directory_fd, _filename: target_fd,
    )

    try:
        with pytest.raises(ParentCloseError, match="simulated parent close failure"):
            read_regular_file(document, max_bytes=1_024)

        with pytest.raises(OSError) as closed_error:
            os.fstat(target_fd)
        assert closed_error.value.errno == errno.EBADF
    finally:
        try:
            os.close(target_fd)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise


def test_read_regular_file_cannot_be_redirected_by_ancestor_swap(monkeypatch, tmp_path):
    requested_parent = tmp_path / "raced"
    requested_parent.mkdir()
    (requested_parent / "document.md").write_bytes(b"trusted")
    moved_parent = tmp_path / "moved"

    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    (attacker_directory / "document.md").write_bytes(b"attacker-controlled")

    real_open = safe_io.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        path_text = os.fspath(path)
        opening_old_full_path = dir_fd is None and path_text == os.fspath(
            requested_parent / "document.md"
        )
        opening_new_relative_component = (
            dir_fd is not None and path_text == requested_parent.name
        )
        if not swapped and (opening_old_full_path or opening_new_relative_component):
            requested_parent.rename(moved_parent)
            requested_parent.symlink_to(attacker_directory, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_io.os, "open", racing_open)

    with pytest.raises(UnsafeFileError, match="ancestors"):
        read_regular_file(requested_parent / "document.md", max_bytes=1_024)
    assert swapped


def test_read_regular_file_rejects_parent_traversal(tmp_path):
    safe_directory = tmp_path / "safe"
    safe_directory.mkdir()
    (tmp_path / "secret.md").write_bytes(b"not reachable through dot-dot")

    with pytest.raises(UnsafeFileError, match=r"\.\."):
        read_regular_file(safe_directory / ".." / "secret.md", max_bytes=1_024)


def test_write_wiki_index_rejects_symlinked_output_directory(tmp_path):
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    linked_wiki = tmp_path / "wiki"
    linked_wiki.symlink_to(attacker_directory, target_is_directory=True)

    with pytest.raises(UnsafeFileError, match="ancestors"):
        filing.write_wiki_index([], wiki_dir=linked_wiki)
    assert not (attacker_directory / "index.md").exists()


def test_write_wiki_index_rejects_symlink_index_without_touching_target(tmp_path):
    wiki_directory = tmp_path / "wiki"
    wiki_directory.mkdir()
    attacker_target = tmp_path / "attacker-index.md"
    attacker_target.write_text("do not replace\n", encoding="utf-8")
    (wiki_directory / "index.md").symlink_to(attacker_target)

    with pytest.raises(UnsafeFileError, match="regular non-symlink"):
        filing.write_wiki_index([], wiki_dir=wiki_directory)
    assert attacker_target.read_text(encoding="utf-8") == "do not replace\n"


def test_write_wiki_index_uses_descriptor_relative_atomic_replace_and_fsync(
    monkeypatch, tmp_path
):
    real_replace = safe_io.os.replace
    replace_calls = []
    operations = []

    def recording_replace(source, destination, *args, **kwargs):
        result = real_replace(source, destination, *args, **kwargs)
        replace_calls.append((source, destination, kwargs))
        operations.append(("replace", kwargs["dst_dir_fd"]))
        return result

    real_fsync = safe_io.os.fsync
    fsynced_modes = []

    def recording_fsync(fd):
        mode = os.fstat(fd).st_mode
        result = real_fsync(fd)
        fsynced_modes.append(mode)
        operation = "fsync-directory" if stat.S_ISDIR(mode) else "fsync-file"
        operations.append((operation, fd))
        return result

    monkeypatch.setattr(safe_io.os, "replace", recording_replace)
    monkeypatch.setattr(safe_io.os, "fsync", recording_fsync)

    wiki_directory = tmp_path / "nested" / "wiki"
    output_path = filing.write_wiki_index([], wiki_dir=wiki_directory)

    assert output_path.read_text(encoding="utf-8").startswith(
        "# Knowledge Base Wiki Index\n"
    )
    assert len(replace_calls) == 1
    _, destination, replace_kwargs = replace_calls[0]
    assert destination == "index.md"
    assert replace_kwargs["src_dir_fd"] == replace_kwargs["dst_dir_fd"]
    assert replace_kwargs["src_dir_fd"] is not None
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
    replace_operation = ("replace", replace_kwargs["dst_dir_fd"])
    final_directory_fsync = ("fsync-directory", replace_kwargs["dst_dir_fd"])
    replace_index = operations.index(replace_operation)
    assert operations.index(final_directory_fsync, replace_index + 1) > replace_index
    assert operations[-1] == final_directory_fsync
    assert sorted(path.name for path in wiki_directory.iterdir()) == ["index.md"]


@pytest.mark.parametrize("failure_stage", ["write", "fsync"])
def test_atomic_write_preserves_primary_failure_when_temp_cleanup_fails(
    monkeypatch, tmp_path, failure_stage
):
    class PrimaryWriteError(OSError):
        pass

    class CleanupError(OSError):
        pass

    if failure_stage == "write":

        def fail_write(_fd, _data):
            raise PrimaryWriteError("primary write failure")

        monkeypatch.setattr(safe_io, "_write_all", fail_write)
    else:
        real_fsync = safe_io.os.fsync

        def fail_file_fsync(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                raise PrimaryWriteError("primary fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(safe_io.os, "fsync", fail_file_fsync)

    def fail_cleanup_unlink(_path, *, dir_fd):
        raise CleanupError(f"cleanup unlink failure in fd {dir_fd}")

    monkeypatch.setattr(safe_io.os, "unlink", fail_cleanup_unlink)

    with pytest.raises(PrimaryWriteError, match=f"primary {failure_stage} failure"):
        safe_io.atomic_write_regular_file(tmp_path / "output.md", b"contents")


def test_atomic_write_supports_destination_at_name_max(tmp_path):
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    destination = tmp_path / ("x" * name_max)

    safe_io.atomic_write_regular_file(destination, b"contents")

    assert destination.read_bytes() == b"contents"


def test_file_answer_rejects_symlinked_output_directory(tmp_path):
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    linked_answers = tmp_path / "answers"
    linked_answers.symlink_to(attacker_directory, target_is_directory=True)

    with pytest.raises(UnsafeFileError, match="ancestors"):
        filing.file_answer("Question?", "Answer.", [], answers_dir=linked_answers)
    assert list(attacker_directory.iterdir()) == []


def test_file_answer_uses_descriptor_relative_exclusive_create_and_fsync(
    monkeypatch, tmp_path
):
    real_open = safe_io.os.open
    create_calls = []
    operations = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT:
            create_calls.append((os.fspath(path), flags, dir_fd, fd))
            operations.append(("create", fd, dir_fd))
        return fd

    real_fsync = safe_io.os.fsync
    fsynced_modes = []

    def recording_fsync(fd):
        mode = os.fstat(fd).st_mode
        result = real_fsync(fd)
        fsynced_modes.append(mode)
        operation = "fsync-directory" if stat.S_ISDIR(mode) else "fsync-file"
        operations.append((operation, fd))
        return result

    monkeypatch.setattr(safe_io.os, "open", recording_open)
    monkeypatch.setattr(safe_io.os, "fsync", recording_fsync)
    monkeypatch.setattr(filing.secrets, "token_hex", lambda _length: "unique")

    answer_path = filing.file_answer(
        "Question?", "Answer.", [], answers_dir=tmp_path / "nested" / "answers"
    )

    assert answer_path.exists()
    assert len(create_calls) == 1
    created_name, create_flags, parent_fd, output_fd = create_calls[0]
    assert created_name == answer_path.name
    assert create_flags & os.O_EXCL
    assert create_flags & os.O_NOFOLLOW
    assert parent_fd is not None
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
    create_index = operations.index(("create", output_fd, parent_fd))
    output_fsync_index = operations.index(("fsync-file", output_fd), create_index + 1)
    final_directory_fsync = ("fsync-directory", parent_fd)
    assert (
        operations.index(final_directory_fsync, output_fsync_index + 1)
        > output_fsync_index
    )
    assert operations[-1] == final_directory_fsync


def test_file_answer_removes_partial_file_when_descriptor_write_fails(
    monkeypatch, tmp_path
):
    real_write = safe_io.os.write
    write_calls = 0

    def interrupted_write(fd, data):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(fd, data[:8])
        raise OSError("simulated write failure")

    monkeypatch.setattr(safe_io.os, "write", interrupted_write)
    monkeypatch.setattr(filing.secrets, "token_hex", lambda _length: "partial")
    answers_directory = tmp_path / "answers"

    with pytest.raises(OSError, match="simulated write failure"):
        filing.file_answer("Question?", "Answer.", [], answers_dir=answers_directory)

    assert answers_directory.is_dir()
    assert list(answers_directory.iterdir()) == []
