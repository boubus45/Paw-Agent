from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol

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
    "name": "read_file|write_file|replace_in_file|rollback_file|list_files|search|run_shell|run_cmd|run_powershell|web_search",
    "args": { ... }
  } | null
}
Rules:
- Be concise and clear.
- Keep assistant_response reasonably short.
- Prefer list_files/search before reading many files.
- Read only targeted files and keep outputs concise.
- Prefer replace_in_file for focused edits to existing files.
- Use write_file mainly for new files or full rewrites when necessary.
- For coding tasks, operate step-by-step:
  1) inspect files, 2) change one focused file or small set per step,
  3) run validation, 4) finalize.
- After any file change, you must run a validation command before finishing.
- If validation fails, debug and fix the issue before finishing.
- Use run_cmd or run_powershell on Windows when the shell matters.
- If blocked on an unfamiliar error, use web_search and continue fixing.
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


@dataclass(frozen=True)
class ValidationCommand:
    tool_name: str
    command: str
    workdir: Path


class AgentObserver(Protocol):
    def on_step_start(self, step: int) -> None: ...
    def on_model_text(self, text: str) -> None: ...
    def on_model_done(self) -> None: ...
    def on_tool_call(self, name: str, args: Dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_info(self, message: str) -> None: ...


class PawAgent:
    def __init__(self, cfg: Dict[str, Any], workspace: Path, observer: AgentObserver | None = None):
        self.cfg = cfg
        self.workspace = workspace.resolve()
        self.observer = observer
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
            request_timeout_sec=int(model_cfg.get("request_timeout_sec", 32865)),
        )
        self.tools = ToolRuntime(self.workspace)
        self.skills = SkillStore()
        self.max_steps = int(cfg["agent"].get("max_steps", 24))
        self.auto_vector_retrieval = bool(cfg["agent"].get("auto_vector_retrieval", True))
        self.vector_top_k = int(cfg["agent"].get("vector_top_k", 4))
        self.vector_max_chars = int(cfg["agent"].get("vector_max_chars", 2600))
        self.vector_include_global = bool(cfg["agent"].get("vector_include_global", False))
        self.root_validation_commands = detect_validation_commands(self.workspace, [])

    def run(self, user_goal: str) -> AgentResult:
        system_parts: List[str] = [SYSTEM_PROMPT.strip()]
        skill_context = self._skill_context(user_goal)
        if skill_context:
            system_parts.append(skill_context.strip())
        vector_context = self._vector_context(user_goal)
        if vector_context:
            system_parts.append(vector_context.strip())
        if self.root_validation_commands:
            detected = "; ".join(
                f"{v.tool_name} `{v.command}` in `{v.workdir}`" for v in self.root_validation_commands[:3]
            )
            system_parts.append(
                "Detected project validation command(s): "
                f"{detected}. After edits, validate the changed subproject(s)."
            )
        system_parts.append(TOOL_SPEC.strip())
        messages: List[Dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        messages.append({"role": "user", "content": user_goal})

        transcript: List[Dict[str, Any]] = []
        final_response = ""
        tool_calls = 0
        changed_files: List[str] = []
        validation_attempted = False
        validation_succeeded = False

        for step in range(1, self.max_steps + 1):
            client = self.client
            self._notify_step_start(step)
            raw = client.chat_stream(messages, self._notify_model_text)
            self._notify_model_done()
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
                if changed_files and not validation_attempted:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You changed files but have not validated the project yet. "
                                "Run an appropriate validation command now using run_cmd, "
                                "run_powershell, or run_shell before finishing."
                            ),
                        }
                    )
                    continue
                if changed_files and not validation_succeeded:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Validation has not succeeded yet. Continue debugging, fixing, "
                                "and re-running validation. If you are stuck, use web_search."
                            ),
                        }
                    )
                    continue
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
                self._notify_tool_call(name, args)
                result = self.tools.run(name, args)
                tool_calls += 1
                if name in {"write_file", "replace_in_file"} and args.get("path"):
                    changed_files.append(str(args["path"]))
                    validation_attempted = False
                    validation_succeeded = False
                    auto_validation = self._auto_validate_after_change(changed_files)
                    if auto_validation:
                        validation_attempted = True
                        validation_succeeded = self._all_validations_succeeded(auto_validation)
                        self._notify_info("Automatic validation executed.")
                        self._notify_tool_result("automatic_validation", auto_validation)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Automatic validation result after file change:\n"
                                    f"{auto_validation}"
                                ),
                            }
                        )
                        transcript[-1]["auto_validation_result"] = auto_validation
                if name in {"run_shell", "run_cmd", "run_powershell"}:
                    validation_attempted = True
                    if self._tool_succeeded(result):
                        validation_succeeded = True
            except Exception as exc:
                result = f"TOOL_ERROR: {exc}\n{traceback.format_exc(limit=1)}"
            self._notify_tool_result(name, result)
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            transcript[-1]["tool_result"] = result
            if name in {"run_shell", "run_cmd", "run_powershell"} and changed_files and not self._tool_succeeded(result):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Validation failed. Inspect the error, fix the problem, and run "
                            "validation again. Use web_search if the error is unfamiliar."
                        ),
                    }
                )

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

    def _tool_succeeded(self, result: str) -> bool:
        first = result.splitlines()[0].strip().lower() if result else ""
        return first == "exit_code=0"

    def _notify_step_start(self, step: int) -> None:
        if self.observer:
            self.observer.on_step_start(step)

    def _notify_model_text(self, text: str) -> None:
        if self.observer:
            self.observer.on_model_text(text)

    def _notify_model_done(self) -> None:
        if self.observer:
            self.observer.on_model_done()

    def _notify_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        if self.observer:
            self.observer.on_tool_call(name, args)

    def _notify_tool_result(self, name: str, result: str) -> None:
        if self.observer:
            self.observer.on_tool_result(name, result)

    def _notify_info(self, message: str) -> None:
        if self.observer:
            self.observer.on_info(message)

    def _all_validations_succeeded(self, result: str) -> bool:
        if not result.strip():
            return False
        blocks = [b.strip() for b in result.split("\n\n===\n\n") if b.strip()]
        return bool(blocks) and all(self._tool_succeeded(block) for block in blocks)

    def _auto_validate_after_change(self, changed_files: List[str]) -> str:
        validations = detect_validation_commands(self.workspace, changed_files)
        if not validations:
            return ""
        outputs: List[str] = []
        for validation in validations:
            rel_workdir = str(validation.workdir.relative_to(self.workspace))
            original_workspace = self.tools.workspace
            try:
                self.tools.workspace = validation.workdir
                result = self.tools.run(validation.tool_name, {"command": validation.command, "timeout_sec": 300})
            except Exception as exc:
                result = f"TOOL_ERROR: automatic validation failed: {exc}"
            finally:
                self.tools.workspace = original_workspace
            outputs.append(
                f"validation_workdir={rel_workdir or '.'}\nvalidation_tool={validation.tool_name}\nvalidation_command={validation.command}\n{result}"
            )
            if not self._tool_succeeded(result):
                break
        return "\n\n===\n\n".join(outputs)

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
            if name in {"write_file", "replace_in_file"} and isinstance(args, dict) and args.get("path"):
                changed_files.append(str(args["path"]))
            if name == "rollback_file" and isinstance(args, dict) and args.get("path"):
                rollback_files.append(str(args["path"]))
            if name == "run_shell":
                tests_run = True
            if name == "run_cmd":
                tests_run = True
            if name == "run_powershell":
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


def detect_validation_commands(workspace: Path, changed_files: List[str]) -> List[ValidationCommand]:
    workspace = workspace.resolve()
    candidate_dirs: List[Path] = []
    if changed_files:
        for rel in changed_files:
            try:
                p = (workspace / rel).resolve()
            except Exception:
                continue
            start = p.parent if p.suffix else p
            candidate_dirs.append(start)
    else:
        candidate_dirs.append(workspace)

    commands: List[ValidationCommand] = []
    seen: set[tuple[str, str, str]] = set()
    for start in candidate_dirs:
        for current in [start, *start.parents]:
            if current == workspace.parent:
                break
            if workspace not in current.parents and current != workspace:
                continue
            validation = _detect_validation_command_in_dir(current, workspace)
            if not validation:
                continue
            key = (validation.tool_name, validation.command, str(validation.workdir))
            if key not in seen:
                seen.add(key)
                commands.append(validation)
            break
    if commands:
        return commands
    fallback = _detect_validation_command_in_dir(workspace, workspace)
    return [fallback] if fallback else []


def _detect_validation_command_in_dir(current: Path, workspace: Path) -> ValidationCommand | None:
    if (current / "package.json").exists():
        manager = _detect_js_package_manager(current, workspace)
        return ValidationCommand("run_cmd", f"{manager} test", current)
    if (current / "Cargo.toml").exists():
        return ValidationCommand("run_cmd", "cargo test", current)
    if (current / "go.mod").exists():
        return ValidationCommand("run_cmd", "go test ./...", current)
    if (current / "pom.xml").exists():
        return ValidationCommand("run_cmd", "mvn test", current)
    if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
        cmd = "gradlew.bat test" if (current / "gradlew.bat").exists() else "gradlew test"
        return ValidationCommand("run_cmd", cmd, current)
    if list(current.glob("*.sln")) or list(current.glob("*.csproj")):
        return ValidationCommand("run_cmd", "dotnet test", current)
    if (current / "pyproject.toml").exists():
        text = (current / "pyproject.toml").read_text(encoding="utf-8", errors="replace").lower()
        cmd = "pytest" if "pytest" in text else "python -m pytest"
        return ValidationCommand("run_cmd", cmd, current)
    if (current / "pytest.ini").exists() or (current / "conftest.py").exists():
        return ValidationCommand("run_cmd", "pytest", current)
    if (current / "requirements.txt").exists() or list(current.glob("test_*.py")) or (current / "tests").exists():
        return ValidationCommand("run_cmd", "pytest", current)
    return None


def _detect_js_package_manager(current: Path, workspace: Path) -> str:
    for directory in [current, *current.parents]:
        if workspace not in directory.parents and directory != workspace:
            continue
        if (directory / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (directory / "yarn.lock").exists():
            return "yarn"
        if (directory / "package-lock.json").exists():
            return "npm"
    return "npm"
