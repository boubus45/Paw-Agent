from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


GLOBAL_HOME = Path(os.getenv("PAW_AGENT_GLOBAL_HOME", Path.home() / ".paw-agent"))


@dataclass
class VectorHit:
    file_path: str
    chunk_index: int
    score: float
    text: str


def resolve_vector_db_path(workspace: Path, global_mode: bool) -> Path:
    if global_mode:
        root = GLOBAL_HOME / "vector"
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root / "global.sqlite3"
        except Exception:
            # Fallback when user home is restricted in sandboxed environments.
            fb = workspace / ".paw-agent" / "global-vector"
            fb.mkdir(parents=True, exist_ok=True)
            return fb / "global.sqlite3"
    app_home = workspace / ".paw-agent"
    app_home.mkdir(parents=True, exist_ok=True)
    return app_home / "vector.sqlite3"


class VectorStore:
    def __init__(self, db_path: Path, dims: int = 256):
        self.db_path = db_path
        self.dims = dims
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              file_path TEXT PRIMARY KEY,
              content_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_path TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL,
              vec TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            """
        )
        self.conn.commit()

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"docs": int(docs), "chunks": int(chunks), "db_path": str(self.db_path)}

    def index_files(
        self,
        workspace: Path,
        rel_root: str,
        extensions: Iterable[str],
        max_files: int = 2000,
        chunk_chars: int = 1200,
        overlap: int = 180,
    ) -> dict:
        workspace = workspace.resolve()
        root = (workspace / rel_root).resolve()
        if workspace not in root.parents and root != workspace:
            raise ValueError(f"Path escapes workspace: {rel_root}")
        exts = {e.lower().strip() for e in extensions if e.strip()}
        indexed_docs = 0
        indexed_chunks = 0
        skipped = 0
        for i, path in enumerate(root.rglob("*")):
            if i >= max_files:
                break
            if not path.is_file():
                continue
            if ".paw-agent" in path.parts:
                continue
            if exts and path.suffix.lower() not in exts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue
            rel = str(path.relative_to(workspace))
            doc_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            row = self.conn.execute(
                "SELECT content_hash FROM docs WHERE file_path = ?",
                (rel,),
            ).fetchone()
            if row and row[0] == doc_hash:
                continue
            self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))
            chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                vec = _embed_text(chunk, self.dims)
                self.conn.execute(
                    "INSERT INTO chunks(file_path, chunk_index, text, vec) VALUES(?,?,?,?)",
                    (rel, idx, chunk, json.dumps(vec)),
                )
            self.conn.execute(
                """
                INSERT INTO docs(file_path, content_hash, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(file_path) DO UPDATE SET
                  content_hash=excluded.content_hash,
                  updated_at=excluded.updated_at
                """,
                (rel, doc_hash, datetime.now(timezone.utc).isoformat()),
            )
            indexed_docs += 1
            indexed_chunks += len(chunks)
        self.conn.commit()
        return {
            "indexed_docs": indexed_docs,
            "indexed_chunks": indexed_chunks,
            "skipped": skipped,
            "db_path": str(self.db_path),
        }

    def query(self, query_text: str, top_k: int = 5) -> List[VectorHit]:
        qv = _embed_text(query_text, self.dims)
        rows = self.conn.execute("SELECT file_path, chunk_index, text, vec FROM chunks").fetchall()
        scored: List[VectorHit] = []
        for file_path, chunk_index, text, vec_json in rows:
            try:
                vec = json.loads(vec_json)
                score = _cosine(qv, vec)
            except Exception:
                continue
            scored.append(
                VectorHit(
                    file_path=str(file_path),
                    chunk_index=int(chunk_index),
                    score=float(score),
                    text=str(text),
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(1, min(top_k, 50))]


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]
    out: List[str] = []
    step = max(1, chunk_chars - overlap)
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_chars].strip()
        if piece:
            out.append(piece)
        start += step
    return out


def _embed_text(text: str, dims: int) -> List[float]:
    vec = [0.0] * dims
    # Hashing trick embedding: fast, deterministic, zero external deps.
    for tok in _tokens(text):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dims
        sign = -1.0 if (h[4] & 1) else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-", "."}:
            cur.append(ch)
        else:
            if len(cur) >= 2:
                out.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        out.append("".join(cur))
    return out


def _cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(n)))
