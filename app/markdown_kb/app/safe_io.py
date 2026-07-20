import errno
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NoReturn


_READ_CHUNK_BYTES = 1024 * 1024
_TEMP_FILE_CREATE_ATTEMPTS = 100
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)


class UnsafeFileError(ValueError):
    """Raised before unsafe or unbounded filesystem input is consumed."""


@dataclass(frozen=True)
class RegularFileSnapshot:
    data: bytes
    size: int
    mtime_ns: int


def _path_components(path: Path) -> tuple[bool, list[str]]:
    path = Path(path)
    parts = list(path.parts)
    absolute = path.is_absolute()
    if absolute:
        parts = parts[1:]

    if ".." in parts:
        raise UnsafeFileError("path traversal component '..' is not allowed")

    # pathlib removes ordinary ``.`` and empty components. Keep this filter for
    # clarity and for any alternate Path implementation passed by a caller.
    return absolute, [part for part in parts if part not in ("", ".")]


def _raise_unsafe_ancestor(exc: OSError) -> NoReturn:
    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
        raise UnsafeFileError("symlinked path ancestors are not allowed") from exc
    raise exc


def _open_directory_component(parent_fd: int, component: str) -> int:
    try:
        return os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        _raise_unsafe_ancestor(exc)


def _open_directory_components(
    *, absolute: bool, components: list[str], create: bool, mode: int
) -> int:
    anchor = os.sep if absolute else "."
    current_fd = os.open(anchor, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in components:
            try:
                next_fd = _open_directory_component(current_fd, component)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=mode, dir_fd=current_fd)
                except FileExistsError:
                    # Another creator won the race. Opening with O_NOFOLLOW
                    # below verifies that it created a real directory.
                    pass
                else:
                    # Persist each newly-created directory entry in its parent.
                    os.fsync(current_fd)
                next_fd = _open_directory_component(current_fd, component)

            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


@contextmanager
def open_safe_directory(
    path: Path, *, create: bool = False, mode: int = 0o777
) -> Iterator[int]:
    """Yield a directory fd reached without following any path-component symlink.

    Missing components are created descriptor-relatively when ``create`` is
    true. A ``..`` component is always rejected rather than changing the
    traversal root.
    """

    absolute, components = _path_components(Path(path))
    directory_fd = _open_directory_components(
        absolute=absolute,
        components=components,
        create=create,
        mode=mode,
    )
    try:
        yield directory_fd
    finally:
        os.close(directory_fd)


@contextmanager
def _open_parent_directory(
    path: Path, *, create: bool = False, mode: int = 0o777
) -> Iterator[tuple[int, str]]:
    absolute, components = _path_components(Path(path))
    if not components:
        raise UnsafeFileError("path must name a file")
    filename = components.pop()
    directory_fd = _open_directory_components(
        absolute=absolute,
        components=components,
        create=create,
        mode=mode,
    )
    try:
        yield directory_fd, filename
    finally:
        os.close(directory_fd)


def _open_read_only_regular_at(directory_fd: int, filename: str) -> int:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        return os.open(filename, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise UnsafeFileError("symlink input is not allowed") from exc
        raise


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError(errno.EIO, "write made no progress")
        remaining = remaining[written:]


def _create_exclusive_regular_at(directory_fd: int, filename: str, mode: int) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(filename, flags, mode, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafeFileError("created output must be a regular file")
    except BaseException:
        os.close(fd)
        try:
            os.unlink(filename, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    return fd


def _unlink_created_file(directory_fd: int, filename: str) -> None:
    try:
        os.unlink(filename, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    os.fsync(directory_fd)


def exclusive_write_regular_file(
    path: Path,
    data: bytes,
    *,
    create_parents: bool = False,
    directory_mode: int = 0o777,
    file_mode: int = 0o666,
) -> None:
    """Exclusively create, fully write, and durably file one regular file.

    The output and every ancestor are opened descriptor-relatively with
    ``O_NOFOLLOW``. Any failure after creation removes the partial file.
    """

    with _open_parent_directory(
        Path(path), create=create_parents, mode=directory_mode
    ) as (directory_fd, filename):
        fd = _create_exclusive_regular_at(directory_fd, filename, file_mode)
        created = True
        try:
            try:
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory_fd)
            created = False
        except BaseException:
            if created:
                try:
                    _unlink_created_file(directory_fd, filename)
                except OSError:
                    # Preserve the write/fsync exception. The cleanup used a
                    # pinned directory fd and never follows the output name.
                    pass
            raise


def _ensure_regular_replace_target(directory_fd: int, filename: str) -> None:
    try:
        target = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target.st_mode):
        raise UnsafeFileError("destination must be a regular non-symlink file")


def atomic_write_regular_file(
    path: Path,
    data: bytes,
    *,
    create_parents: bool = False,
    directory_mode: int = 0o777,
    file_mode: int = 0o666,
) -> None:
    """Durably replace a regular file using a same-directory temporary file."""

    with _open_parent_directory(
        Path(path), create=create_parents, mode=directory_mode
    ) as (directory_fd, filename):
        _ensure_regular_replace_target(directory_fd, filename)

        temporary_name = None
        temporary_fd = None
        for _ in range(_TEMP_FILE_CREATE_ATTEMPTS):
            candidate = f".kb-write-{secrets.token_hex(8)}.tmp"
            try:
                temporary_fd = _create_exclusive_regular_at(
                    directory_fd, candidate, file_mode
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd is None or temporary_name is None:
            raise FileExistsError(
                f"unable to create an atomic temporary file after "
                f"{_TEMP_FILE_CREATE_ATTEMPTS} attempts"
            )

        temporary_exists = True
        try:
            try:
                _write_all(temporary_fd, data)
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)

            # Re-check after the potentially long write. os.replace itself
            # cannot follow the destination: it replaces the directory entry.
            _ensure_regular_replace_target(directory_fd, filename)
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_exists = False
            os.fsync(directory_fd)
        except BaseException:
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except OSError:
                    # Preserve the write/fsync/replace exception. Cleanup is
                    # descriptor-relative and never follows the temporary name.
                    pass
            raise


def read_regular_file(path: Path, *, max_bytes: int) -> RegularFileSnapshot:
    """Read one immutable, bounded snapshot of a non-symlink regular file."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    with _open_parent_directory(Path(path)) as (directory_fd, filename):
        fd = _open_read_only_regular_at(directory_fd, filename)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise UnsafeFileError("input must be a regular file")
            if before.st_size > max_bytes:
                raise UnsafeFileError("input exceeds the configured size limit")

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise UnsafeFileError("input exceeds the configured size limit")

            after = os.fstat(fd)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            data = b"".join(chunks)
            if before_identity != after_identity or len(data) != after.st_size:
                raise OSError("file changed while it was being read")

            return RegularFileSnapshot(
                data=data,
                size=after.st_size,
                mtime_ns=after.st_mtime_ns,
            )
        finally:
            os.close(fd)
