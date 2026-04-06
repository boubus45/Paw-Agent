from __future__ import annotations

from pathlib import Path
import shutil

from paw_agent.vector_store import VectorStore, resolve_vector_db_path


def test_vector_index_and_query() -> None:
    workspace = Path(".test_tmp") / "vector_ws"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "demo.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    db = resolve_vector_db_path(workspace, global_mode=False)
    store = VectorStore(db)
    result = store.index_files(
        workspace=workspace,
        rel_root=".",
        extensions=[".py"],
        max_files=20,
    )
    assert result["indexed_docs"] >= 1
    hits = store.query("how to add two numbers", top_k=3)
    store.close()
    assert hits
    assert "demo.py" in hits[0].file_path
