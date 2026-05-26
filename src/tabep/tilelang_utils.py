from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
import warnings


_TILELANG_WARNINGS_CONFIGURED = False


def configure_tilelang_warnings() -> None:
    """Silence noisy TileLang/TVM duplicate-field warnings emitted at import/compile time."""
    global _TILELANG_WARNINGS_CONFIGURED
    if _TILELANG_WARNINGS_CONFIGURED:
        return

    warnings.filterwarnings("ignore", message=r"Field .* duplicates an ancestor field.*")
    warnings.filterwarnings("ignore", message=r".*Child types should not re-register inherited fields.*")

    logging.getLogger("TileLang").setLevel(logging.ERROR)
    logging.getLogger("tilelang").setLevel(logging.ERROR)

    _TILELANG_WARNINGS_CONFIGURED = True


class _TileLangStderrFilter:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def write(self, text: str) -> int:
        if "duplicates an ancestor field" in text or "Child types should not re-register inherited fields" in text:
            return len(text)
        return self.wrapped.write(text)

    def flush(self) -> None:
        self.wrapped.flush()

    def isatty(self) -> bool:
        return self.wrapped.isatty()

    def fileno(self) -> int:
        return self.wrapped.fileno()

    @property
    def encoding(self):
        return self.wrapped.encoding


def install_tilelang_stderr_filter() -> None:
    configure_tilelang_warnings()
    if not isinstance(sys.stderr, _TileLangStderrFilter):
        sys.stderr = _TileLangStderrFilter(sys.stderr)


@contextmanager
def silence_stderr_fd():
    """Temporarily silence low-level stderr writes, including C/C++ extension output."""
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)
