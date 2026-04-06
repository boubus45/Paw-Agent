from __future__ import annotations

from pathlib import Path
import shutil

from paw_agent.agent import build_vector_context
from paw_agent.vector_store import VectorStore, resolve_vector_db_path


def test_build_vector_context_returns_relevant_chunks() -> None:
    workspace = Path(".test_tmp") / "agent_vec_ws"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "math_utils.py").write_text(
        "def multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b\n",
        encoding="utf-8",
    )
    db = resolve_vector_db_path(workspace, global_mode=False)
    store = VectorStore(db)
    store.index_files(workspace=workspace, rel_root=".", extensions=[".py"], max_files=20)
    store.close()

    ctx = build_vector_context(workspace, query="how to multiply numbers", top_k=3, max_chars=1200)
    assert ctx
    assert "math_utils.py" in ctx
