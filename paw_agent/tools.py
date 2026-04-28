from __future__ import annotations

import hashlib
import json
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


class ToolRuntime:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.history_dir = self.workspace / ".paw-agent" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, args: Dict[str, Any]) -> str:
        if name == "read_file":
            return self.read_file(args["path"], int(args.get("max_chars", 12000)))
        if name == "write_file":
            return self.write_file(args["path"], args["content"])
        if name == "replace_in_file":
            return self.replace_in_file(
                args["path"],
                args["old"],
                args["new"],
                int(args.get("count", 1)),
            )
        if name == "list_files":
            return self.list_files(args.get("path", "."), int(args.get("limit", 300)))
        if name == "search":
            return self.search(args["pattern"], args.get("path", "."))
        if name == "run_shell":
            return self.run_shell(args["command"], int(args.get("timeout_sec", 120)))
        if name == "run_cmd":
            return self.run_cmd(args["command"], int(args.get("timeout_sec", 120)))
        if name == "run_powershell":
            return self.run_powershell(args["command"], int(args.get("timeout_sec", 120)))
        if name == "web_search":
            return self.web_search(args["query"], int(args.get("limit", 5)))
        if name == "rollback_file":
            return self.rollback_file(args["path"])
        raise ValueError(f"Unknown tool: {name}")

    def read_file(self, rel_path: str, max_chars: int) -> str:
        path = self._resolve(rel_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]..."
        return text

    def write_file(self, rel_path: str, content: str) -> str:
        path = self._resolve(rel_path)
        self._snapshot_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def replace_in_file(self, rel_path: str, old: str, new: str, count: int) -> str:
        path = self._resolve(rel_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return f"Text to replace not found in {rel_path}"
        self._snapshot_file(path)
        replaced = text.replace(old, new, max(1, count))
        path.write_text(replaced, encoding="utf-8")
        return f"Replaced text in {rel_path}"

    def rollback_file(self, rel_path: str) -> str:
        path = self._resolve(rel_path)
        manifest = self._read_manifest(rel_path)
        versions = manifest.get("versions", [])
        if not versions:
            return f"No rollback snapshot found for {rel_path}"
        snap_name = versions.pop()
        snap_path = self.history_dir / snap_name
        if not snap_path.exists():
            self._write_manifest(rel_path, manifest)
            return f"Snapshot missing for {rel_path}: {snap_name}"
        data = json.loads(snap_path.read_text(encoding="utf-8"))
        existed = bool(data.get("existed", False))
        old_content = data.get("content", "")
        if existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(old_content), encoding="utf-8")
        else:
            if path.exists():
                path.unlink()
        self._write_manifest(rel_path, manifest)
        return f"Rolled back {rel_path} to previous snapshot"

    def list_files(self, rel_path: str, limit: int) -> str:
        root = self._resolve(rel_path)
        files: List[str] = []
        for p in root.rglob("*"):
            if len(files) >= limit:
                break
            if p.is_file():
                files.append(str(p.relative_to(self.workspace)))
        return "\n".join(files)

    def search(self, pattern: str, rel_path: str) -> str:
        root = self._resolve(rel_path)
        hits: List[str] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    if pattern.lower() in line.lower():
                        hits.append(f"{p.relative_to(self.workspace)}:{i}:{line[:220]}")
            except Exception:
                continue
            if len(hits) >= 300:
                break
        return "\n".join(hits) if hits else "No matches."

    def run_shell(self, command: str, timeout_sec: int) -> str:
        return self._run_subprocess(command, timeout_sec, shell=True)

    def run_cmd(self, command: str, timeout_sec: int) -> str:
        return self._run_subprocess(
            ["cmd.exe", "/d", "/s", "/c", command],
            timeout_sec,
            shell=False,
        )

    def run_powershell(self, command: str, timeout_sec: int) -> str:
        return self._run_subprocess(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            timeout_sec,
            shell=False,
        )

    def _run_subprocess(self, command: Any, timeout_sec: int, shell: bool) -> str:
        proc = subprocess.run(
            command,
            cwd=str(self.workspace),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (proc.stdout or "")[-8000:]
        err = (proc.stderr or "")[-4000:]
        return f"exit_code={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    def web_search(self, query: str, limit: int) -> str:
        q = query.strip()
        if not q:
            return "No query provided."
        limit = max(1, min(limit, 10))
        url = "https://duckduckgo.com/html/"
        try:
            resp = requests.post(url, data={"q": q}, timeout=12)
            resp.raise_for_status()
        except Exception as exc:
            return f"WEB_SEARCH_ERROR: {exc}"
        html = resp.text
        # Lightweight parse of result titles/links/snippets from DuckDuckGo HTML.
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?(?:<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>)?',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        out: List[str] = []
        for i, b in enumerate(blocks[:limit], start=1):
            link, title, sn1, sn2 = b
            snippet = sn1 or sn2 or ""
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            out.append(f"{i}. {title}\n   {link}\n   {snippet}")
        if not out:
            return "No web results parsed."
        return "\n".join(out)

    def _resolve(self, rel_path: str) -> Path:
        p = (self.workspace / rel_path).resolve()
        if self.workspace not in p.parents and p != self.workspace:
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return p

    def _manifest_path(self, rel_path: str) -> Path:
        key = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
        return self.history_dir / f"{key}.json"

    def _read_manifest(self, rel_path: str) -> Dict[str, Any]:
        p = self._manifest_path(rel_path)
        if not p.exists():
            return {"path": rel_path, "versions": []}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"path": rel_path, "versions": []}

    def _write_manifest(self, rel_path: str, manifest: Dict[str, Any]) -> None:
        p = self._manifest_path(rel_path)
        manifest["path"] = rel_path
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _snapshot_file(self, path: Path) -> None:
        rel_path = str(path.relative_to(self.workspace))
        manifest = self._read_manifest(rel_path)
        versions = manifest.get("versions", [])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        tag = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
        snap_name = f"{tag}-{stamp}.json"
        snap_path = self.history_dir / snap_name
        payload = {
            "existed": path.exists(),
            "content": path.read_text(encoding="utf-8", errors="replace") if path.exists() else "",
        }
        snap_path.write_text(json.dumps(payload), encoding="utf-8")
        versions.append(snap_name)
        # Keep bounded history per file.
        if len(versions) > 30:
            old = versions.pop(0)
            old_path = self.history_dir / old
            if old_path.exists():
                old_path.unlink()
        manifest["versions"] = versions
        self._write_manifest(rel_path, manifest)


TOOL_SPEC = """
Available tools:
- read_file(path, max_chars?)
- write_file(path, content)
- replace_in_file(path, old, new, count?)
- rollback_file(path)
- list_files(path?, limit?)
- search(pattern, path?)
- run_shell(command, timeout_sec?)
- run_cmd(command, timeout_sec?)
- run_powershell(command, timeout_sec?)
- web_search(query, limit?)
"""
