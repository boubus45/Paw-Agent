from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from paw_agent.llama_client import LlamaCppClient
from paw_agent.memory import SkillStore
from paw_agent.tools import TOOL_SPEC, ToolRuntime
from paw_agent.vector_store import VectorStore, resolve_vector_db_path


SYSTEM_PROMPT = """You are Paw-Agent, a local coding CLI agent.
You must solve tasks incrementally with tools.
Return ONLY valid JSON with this schema:
{
  "status": "continue" | "done",
  "assistant_response": "very short user-facing update",
  "tool_call": {
    "name": "read_file|write_file|rollback_file|list_files|search|run_shell|web_search",
    "args": { ... }
  } | null
}
Rules:
- No chit-chat, no motivational text, no filler.
- Keep assistant_response concise: max 2 short sentences.
- Prefer list_files/search before reading many files.
- Read only targeted files and keep outputs concise.
- For coding tasks, operate step-by-step:
  1) inspect files, 2) change one focused file or small set per step,
  3) run validation, 4) finalize.
- If writing code, verify with run_shell tests when reasonable.
- Use rollback_file(path) if a write was incorrect.
- If done, set status to "done" and tool_call to null, and include a concise summary
  of changed files and verification status in assistant_response.
"""


@dataclass
class AgentResult:
    final_response: str
    steps: int
    tool_calls: int
    session_path: Path


class PawAgent:
    def __init__(self, cfg: Dict[str, Any], workspace: Path):
        self.cfg = cfg
        self.workspace = workspace.resolve()
        model_cfg = cfg["model"]
        selected_model = str(model_cfg.get("model", "auto"))
        if selected_model.lower() in {"", "auto"}:
            selected_model = LlamaCppClient.discover_model_id(
                base_url=model_cfg["base_url"],
                timeout_sec=5,
            )
        self.client = LlamaCppClient(
            base_url=model_cfg["base_url"],
            model=selected_model,
            temperature=float(model_cfg["temperature"]),
            top_p=float(model_cfg["top_p"]),
            max_tokens=int(model_cfg["max_tokens"]),
            request_timeout_sec=int(model_cfg.get("request_timeout_sec", 300)),
        )
        self.tools = ToolRuntime(self.workspace)
        self.skills = SkillStore()
        self.max_steps = int(cfg["agent"].get("max_steps", 24))
        self.auto_vector_retrieval = bool(cfg["agent"].get("auto_vector_retrieval", True))
        self.vector_top_k = int(cfg["agent"].get("vector_top_k", 4))
        self.vector_max_chars = int(cfg["agent"].get("vector_max_chars", 2600))
        self.vector_include_global = bool(cfg["agent"].get("vector_include_global", False))

    def run(self, user_goal: str) -> AgentResult:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        skill_context = self._skill_context(user_goal)
        if skill_context:
            messages.append({"role": "system", "content": skill_context})
        vector_context = self._vector_context(user_goal)
        if vector_context:
            messages.append({"role": "system", "content": vector_context})
        messages.append({"role": "system", "content": TOOL_SPEC})
        messages.append({"role": "user", "content": user_goal})

        transcript: List[Dict[str, Any]] = []
        final_response = ""
        tool_calls = 0

        for step in range(1, self.max_steps + 1):
            client = self.client
            raw = client.chat(messages)
            action = self._parse_action(raw)
            assistant_response = self._compact_response(str(action.get("assistant_response", "")).strip())
            final_response = assistant_response or final_response

            transcript.append(
                {
                    "step": step,
                    "model": client.model,
                    "raw": raw,
                    "action": action,
                }
            )
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})

            status = str(action.get("status", "continue")).lower()
            tool_call = action.get("tool_call")
            if status == "done" and not tool_call:
                final_response = self._with_auto_change_summary(final_response, transcript)
                session_path = self._save_session(user_goal, transcript, final_response, tool_calls, step)
                self._capture_skill_if_needed(user_goal, transcript, tool_calls, final_response)
                return AgentResult(
                    final_response=final_response or "Done.",
                    steps=step,
                    tool_calls=tool_calls,
                    session_path=session_path,
                )

            if not tool_call:
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool call provided. Continue with a valid JSON response and tool_call when needed.",
                    }
                )
                continue

            try:
                name = str(tool_call["name"])
                args = dict(tool_call.get("args", {}))
                result = self.tools.run(name, args)
                tool_calls += 1
            except Exception as exc:
                result = f"TOOL_ERROR: {exc}\n{traceback.format_exc(limit=1)}"
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            transcript[-1]["tool_result"] = result

        session_path = self._save_session(user_goal, transcript, final_response, tool_calls, self.max_steps)
        return AgentResult(
            final_response=self._with_auto_change_summary(final_response or "Stopped after max steps.", transcript),
            steps=self.max_steps,
            tool_calls=tool_calls,
            session_path=session_path,
        )

    def _parse_action(self, raw: str) -> Dict[str, Any]:
        try:
            obj = LlamaCppClient.parse_json_block(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {
            "status": "continue",
            "assistant_response": self._compact_response("Recovering from malformed model output."),
            "tool_call": None,
        }

    def _compact_response(self, text: str, max_chars: int = 420) -> str:
        t = " ".join(text.split())
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 3].rstrip() + "..."

    def _with_auto_change_summary(self, text: str, transcript: List[Dict[str, Any]]) -> str:
        changed_files: List[str] = []
        rollback_files: List[str] = []
        tests_run = False
        for item in transcript:
            action = item.get("action", {})
            call = action.get("tool_call") if isinstance(action, dict) else None
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", ""))
            args = call.get("args", {})
            if name == "write_file" and isinstance(args, dict) and args.get("path"):
                changed_files.append(str(args["path"]))
            if name == "rollback_file" and isinstance(args, dict) and args.get("path"):
                rollback_files.append(str(args["path"]))
            if name == "run_shell":
                tests_run = True
        if not changed_files and not rollback_files and not tests_run:
            return self._compact_response(text)
        uniq_changed = list(dict.fromkeys(changed_files))[:8]
        uniq_rollback = list(dict.fromkeys(rollback_files))[:8]
        parts = [self._compact_response(text)]
        if uniq_changed:
            parts.append("Changed: " + ", ".join(uniq_changed))
        if uniq_rollback:
            parts.append("Rolled back: " + ", ".join(uniq_rollback))
        parts.append("Validation: " + ("run_shell executed" if tests_run else "not run"))
        return self._compact_response(" | ".join(parts), max_chars=520)

    def _skill_context(self, user_goal: str) -> str:
        skills = self.skills.retrieve(user_goal, limit=2)
        if not skills:
            return ""
        lines = ["Relevant learned skills from prior successful sessions:"]
        for s in skills:
            lines.append(f"- {s.title}: when {s.trigger}; do {s.guidance}")
        return "\n".join(lines)

    def _save_session(
        self,
        goal: str,
        transcript: List[Dict[str, Any]],
        final_response: str,
        tool_calls: int,
        steps: int,
    ) -> Path:
        from paw_agent.config import SESSIONS_DIR

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = SESSIONS_DIR / f"session-{stamp}.json"
        payload = {
            "goal": goal,
            "final_response": final_response,
            "steps": steps,
            "tool_calls": tool_calls,
            "transcript": transcript,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _capture_skill_if_needed(
        self,
        user_goal: str,
        transcript: List[Dict[str, Any]],
        tool_calls: int,
        final_response: str,
    ) -> None:
        min_calls = int(self.cfg["agent"].get("skill_capture_min_tool_calls", 5))
        if tool_calls < min_calls:
            return
        tool_names: List[str] = []
        for item in transcript:
            act = item.get("action", {})
            call = act.get("tool_call") if isinstance(act, dict) else None
            if isinstance(call, dict) and call.get("name"):
                tool_names.append(str(call["name"]))
        if not tool_names:
            return
        title = "Tool-chain pattern"
        trigger = user_goal[:160]
        guidance = f"Prefer sequence: {' -> '.join(tool_names[:8])}. Final note: {final_response[:200]}"
        self.skills.capture(
            title=title,
            trigger=trigger,
            guidance=guidance,
            metadata={"tool_calls": tool_calls},
        )

    def _vector_context(self, user_goal: str) -> str:
        if not self.auto_vector_retrieval:
            return ""
        return build_vector_context(
            workspace=self.workspace,
            query=user_goal,
            top_k=self.vector_top_k,
            max_chars=self.vector_max_chars,
            include_global=self.vector_include_global,
        )


def build_vector_context(
    workspace: Path,
    query: str,
    top_k: int = 4,
    max_chars: int = 2600,
    include_global: bool = False,
) -> str:
    def _hits(global_mode: bool):
        try:
            db_path = resolve_vector_db_path(workspace.resolve(), global_mode=global_mode)
            store = VectorStore(db_path)
            out = store.query(query, top_k=top_k)
            store.close()
            return out
        except Exception:
            return []

    project_hits = _hits(False)
    global_hits = _hits(True) if include_global else []
    if not project_hits and not global_hits:
        return ""

    lines: List[str] = ["Retrieved relevant indexed context (higher priority: project):"]
    used = 0
    for h in project_hits:
        item = f"[project] {h.file_path}#{h.chunk_index} score={h.score:.3f}\n{h.text.strip()}\n"
        if used + len(item) > max_chars:
            break
        lines.append(item)
        used += len(item)
    for h in global_hits:
        item = f"[global] {h.file_path}#{h.chunk_index} score={h.score:.3f}\n{h.text.strip()}\n"
        if used + len(item) > max_chars:
            break
        lines.append(item)
        used += len(item)
    return "\n".join(lines).strip()
