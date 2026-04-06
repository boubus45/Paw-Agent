from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from paw_agent.tools import ToolRuntime


def test_write_and_read_file() -> None:
    base = Path(".test_tmp") / "tools_rw"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    rt = ToolRuntime(base)
    rt.write_file("a/b.txt", "hello")
    out = rt.read_file("a/b.txt", 100)
    assert out == "hello"


def test_prevent_workspace_escape() -> None:
    base = Path(".test_tmp") / "tools_escape"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    rt = ToolRuntime(base)
    with pytest.raises(ValueError):
        rt.read_file("../secret.txt", 100)


def test_rollback_file_recovers_previous_content() -> None:
    base = Path(".test_tmp") / "tools_rollback"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    rt = ToolRuntime(base)
    rt.write_file("a.txt", "v1")
    rt.write_file("a.txt", "v2")
    out = rt.rollback_file("a.txt")
    assert "Rolled back" in out
    assert rt.read_file("a.txt", 100) == "v1"
