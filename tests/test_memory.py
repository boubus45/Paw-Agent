from __future__ import annotations

from pathlib import Path
import shutil

from paw_agent.memory import SkillStore


def test_capture_and_retrieve_skill() -> None:
    base = Path(".test_tmp") / "skills"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    store = SkillStore(base)
    store.capture(
        title="CSV Fix",
        trigger="when parsing malformed csv in windows repo",
        guidance="use read_file then search delimiter issues then write_file patch",
    )
    hits = store.retrieve("fix malformed csv parser in repo", limit=2)
    assert hits
    assert hits[0].title == "CSV Fix"
