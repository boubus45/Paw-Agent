from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from paw_agent.agent import PawAgent
from paw_agent.config import CONFIG_PATH, SKILLS_DIR, SESSIONS_DIR, init_config, load_config, save_config
from paw_agent.llama_client import LlamaCppClient
from paw_agent.tools import ToolRuntime
from paw_agent.vector_store import VectorStore, resolve_vector_db_path


class ConsoleObserver:
    def __init__(self) -> None:
        self._in_model_block = False

    def on_step_start(self, step: int) -> None:
        if self._in_model_block:
            print()
            self._in_model_block = False
        print(f"\n[step {step}]")

    def on_model_text(self, text: str) -> None:
        if not self._in_model_block:
            print("[model] ", end="", flush=True)
            self._in_model_block = True
        print(text, end="", flush=True)

    def on_model_done(self) -> None:
        if self._in_model_block:
            print()
            self._in_model_block = False

    def on_tool_call(self, name: str, args: dict) -> None:
        print(f"[tool] {name} {json.dumps(args, ensure_ascii=False)}")

    def on_tool_result(self, name: str, result: str) -> None:
        print(f"[tool-result:{name}]")
        print(result)

    def on_info(self, message: str) -> None:
        print(f"[info] {message}")


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = init_config()
    print(f"Config file: {cfg_path}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    agent = PawAgent(cfg, workspace=workspace, observer=ConsoleObserver())
    info = LlamaCppClient.fetch_server_info(cfg["model"]["base_url"], timeout_sec=8)
    model_name = info.get("model") or agent.client.model
    ctx_value = info.get("ctx") if info.get("ctx") is not None else "unknown"
    try:
        started = time.perf_counter()
        result = agent.run(args.prompt)
        elapsed = time.perf_counter() - started
        print(result.final_response)
        print(
            f"[model={model_name} ctx={ctx_value} "
            f"steps={result.steps} tool_calls={result.tool_calls} time={elapsed:.1f}s]"
        )
        print(f"[session={result.session_path}]")
        return 0
    except requests.Timeout:
        print("Model server timed out while generating.")
        print(
            "Increase `model.request_timeout_sec` or lower `model.max_tokens` in config."
        )
        print("Use `paw doctor` to inspect current server-reported model/context.")
        return 2
    except requests.RequestException as exc:
        print(f"Model server connection failed: {exc}")
        print("Run `paw doctor` to inspect server status.")
        return 2


def cmd_repl(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    agent = PawAgent(cfg, workspace=workspace, observer=ConsoleObserver())
    session_id = args.session_id or _new_session_id()
    turns = _load_chat_turns(session_id)
    _print_repl_banner(cfg, workspace, session_id)
    print(f"chat_session: {session_id}")
    print("Type your prompt or `/help` for commands.\n")
    last_session_path = ""
    while True:
        try:
            prompt = input("paw> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if prompt.startswith("/"):
            handled = _handle_repl_command(prompt, cfg, workspace, session_id)
            if not handled:
                print("Unknown command. Use `/help`.")
            continue
        try:
            prepared_prompt = _prepare_prompt_with_history(turns, prompt)
            started = time.perf_counter()
            result = agent.run(prepared_prompt)
            elapsed = time.perf_counter() - started
            info = LlamaCppClient.fetch_server_info(cfg["model"]["base_url"], timeout_sec=4)
            model_name = info.get("model") or agent.client.model
            ctx_value = info.get("ctx") if info.get("ctx") is not None else "unknown"
            print(result.final_response)
            print(
                f"[model={model_name} ctx={ctx_value} "
                f"steps={result.steps} tool_calls={result.tool_calls} time={elapsed:.1f}s]"
            )
            print(f"[session={result.session_path}]")
            last_session_path = str(result.session_path)
            turns.append({"user": prompt, "assistant": result.final_response})
            _save_chat_turns(session_id, turns, workspace)
        except requests.Timeout:
            print("Model server timed out while generating.")
            print(
                "Increase `model.request_timeout_sec` or lower `model.max_tokens` in config."
            )
        except requests.RequestException as exc:
            print(f"Model server connection failed: {exc}")
            print("Use `paw doctor` then start llama-server.")
    print(f"Resume this chat with: paw resume {session_id}")
    if not turns and last_session_path:
        print(f"Last run session file: {last_session_path}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    ns = argparse.Namespace(workspace=args.workspace, session_id=args.session_id)
    return cmd_repl(ns)


def cmd_sessions(args: argparse.Namespace) -> int:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SESSIONS_DIR.glob("chat-*.json"), reverse=True)
    if not files:
        print("No chat sessions found.")
        return 0
    for p in files[: args.limit]:
        sid = p.stem.replace("chat-", "", 1)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            workspace = data.get("workspace", "")
            turns = len(data.get("turns", []))
        except Exception:
            workspace = "?"
            turns = 0
        print(f"{sid}  turns={turns}  workspace={workspace}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    runtime = ToolRuntime(workspace)
    try:
        out = runtime.rollback_file(args.path)
        print(out)
        return 0
    except Exception as exc:
        print(f"Rollback failed: {exc}")
        return 2


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config()
    model_cfg = cfg["model"]
    print(f"Config: {CONFIG_PATH}")
    print(f"Configured model hint: {model_cfg['model']}")
    print(f"Base URL: {model_cfg['base_url']}")
    print(f"Request timeout: {model_cfg.get('request_timeout_sec', 32865)} sec")
    info = LlamaCppClient.fetch_server_info(model_cfg["base_url"], timeout_sec=6)
    print(f"\nServer status: {info.get('status')}")
    print(f"Loaded model: {info.get('model') or 'unknown'}")
    print(f"Context size: {info.get('ctx') if info.get('ctx') is not None else 'unknown'}")
    if info.get("status") != "online":
        print("Start llama.cpp server first, then run `paw doctor` again.")
    return 0


def cmd_vector_init(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    db_path = resolve_vector_db_path(workspace, global_mode=bool(args.global_mode))
    store = VectorStore(db_path)
    stats = store.stats()
    store.close()
    mode = "global" if args.global_mode else "project"
    print(f"Vector store initialized ({mode}).")
    print(f"db={stats['db_path']} docs={stats['docs']} chunks={stats['chunks']}")
    return 0


def cmd_vector_index(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    db_path = resolve_vector_db_path(workspace, global_mode=bool(args.global_mode))
    store = VectorStore(db_path)
    exts = [x.strip() for x in args.extensions.split(",") if x.strip()]
    result = store.index_files(
        workspace=workspace,
        rel_root=args.path,
        extensions=exts,
        max_files=int(args.max_files),
    )
    store.close()
    mode = "global" if args.global_mode else "project"
    print(f"Indexed vector store ({mode}).")
    print(
        f"docs={result['indexed_docs']} chunks={result['indexed_chunks']} skipped={result['skipped']} db={result['db_path']}"
    )
    return 0


def cmd_vector_query(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    db_path = resolve_vector_db_path(workspace, global_mode=bool(args.global_mode))
    store = VectorStore(db_path)
    hits = store.query(args.query, top_k=int(args.top_k))
    store.close()
    if not hits:
        print("No vector hits found. Run `paw vector index` first.")
        return 0
    for i, h in enumerate(hits, start=1):
        preview = h.text.replace("\n", " ")[:220]
        print(f"{i}. score={h.score:.3f} file={h.file_path} chunk={h.chunk_index}")
        print(f"   {preview}")
    return 0


def cmd_vector_stats(args: argparse.Namespace) -> int:
    cfg = load_config()
    workspace = Path(args.workspace or cfg["agent"].get("workspace", ".")).resolve()
    db_path = resolve_vector_db_path(workspace, global_mode=bool(args.global_mode))
    store = VectorStore(db_path)
    stats = store.stats()
    store.close()
    mode = "global" if args.global_mode else "project"
    print(f"Vector stats ({mode}): db={stats['db_path']} docs={stats['docs']} chunks={stats['chunks']}")
    return 0


def _print_repl_banner(cfg: dict, workspace: Path, session_id: str) -> None:
    model_cfg = cfg["model"]
    info = LlamaCppClient.fetch_server_info(model_cfg["base_url"], timeout_sec=4)
    lines = [
        "Paw-Agent Interactive",
        f"session: {session_id}",
        f"model: {info.get('model') or model_cfg.get('model', 'unknown')}",
        f"provider: {model_cfg.get('provider', 'llamacpp')}",
        f"ctx: {info.get('ctx') if info.get('ctx') is not None else 'unknown'}",
        f"max_tokens: {model_cfg.get('max_tokens', 'n/a')}",
        f"workspace: {workspace}",
        f"server: {info.get('status')}",
    ]
    width = max(len(x) for x in lines) + 2
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"| {line.ljust(width - 1)}|")
    print("+" + "-" * width + "+")


def _handle_repl_command(cmd: str, cfg: dict, workspace: Path, session_id: str) -> bool:
    parts = cmd.strip().split(maxsplit=1)
    c = parts[0].lower()
    if c == "/help":
        print("Commands: /help, /status, /doctor, /skills, /sessions, /vector, /rollback <path>, /exit")
        return True
    if c == "/status":
        _print_repl_banner(cfg, workspace, session_id)
        return True
    if c == "/doctor":
        ns = argparse.Namespace()
        cmd_doctor(ns)
        return True
    if c == "/skills":
        _print_skills()
        return True
    if c == "/sessions":
        ns = argparse.Namespace(limit=20)
        cmd_sessions(ns)
        return True
    if c == "/vector":
        ns = argparse.Namespace(global_mode=False, workspace=str(workspace))
        cmd_vector_stats(ns)
        return True
    if c == "/rollback":
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /rollback <relative-file-path>")
            return True
        ns = argparse.Namespace(path=parts[1].strip(), workspace=str(workspace))
        cmd_rollback(ns)
        return True
    return False


def _print_skills() -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SKILLS_DIR.glob("*.json"), reverse=True)
    if not files:
        print("No learned skills yet.")
        return
    print(f"{len(files)} skills:")
    for p in files[:20]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            title = data.get("title", "untitled")
            trigger = data.get("trigger", "")
            print(f"- {p.stem}: {title} | trigger: {str(trigger)[:80]}")
        except Exception:
            print(f"- {p.stem}")


def _chat_session_path(session_id: str) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_")).strip()
    safe = safe or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"chat-{safe}.json"


def _new_session_id() -> str:
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")
    time_low = (ts_ms >> 16) & 0xFFFFFFFF
    time_mid = ts_ms & 0xFFFF
    time_hi_and_version = (0x7 << 12) | ((rand >> 68) & 0x0FFF)
    clock_seq_hi_and_reserved = 0x80 | ((rand >> 62) & 0x3F)
    clock_seq_low = (rand >> 54) & 0xFF
    node = rand & ((1 << 48) - 1)
    return (
        f"{time_low:08x}-{time_mid:04x}-{time_hi_and_version:04x}-"
        f"{clock_seq_hi_and_reserved:02x}{clock_seq_low:02x}-{node:012x}"
    )


def _load_chat_turns(session_id: str) -> list[dict]:
    p = _chat_session_path(session_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        if isinstance(turns, list):
            return [t for t in turns if isinstance(t, dict)]
    except Exception:
        return []
    return []


def _save_chat_turns(session_id: str, turns: list[dict], workspace: Path) -> None:
    p = _chat_session_path(session_id)
    payload = {
        "session_id": session_id,
        "workspace": str(workspace),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "turns": turns[-30:],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_prompt_with_history(turns: list[dict], prompt: str) -> str:
    if not turns:
        return prompt
    # Keep context lean for small local models.
    recent = turns[-4:]
    lines = ["Conversation history (most recent first):"]
    for t in reversed(recent):
        u = str(t.get("user", "")).strip().replace("\n", " ")
        a = str(t.get("assistant", "")).strip().replace("\n", " ")
        lines.append(f"User: {u[:160]}")
        lines.append(f"Assistant: {a[:160]}")
    lines.append(f"Current user request: {prompt}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paw-agent", description="Paw-Agent local coding CLI agent")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create config file")
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="Run one-shot task")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--workspace")
    p_chat.set_defaults(func=cmd_chat)

    p_repl = sub.add_parser("repl", help="Interactive mode")
    p_repl.add_argument("--workspace")
    p_repl.add_argument("--session-id")
    p_repl.set_defaults(func=cmd_repl)

    p_resume = sub.add_parser("resume", help="Resume an interactive chat session")
    p_resume.add_argument("session_id")
    p_resume.add_argument("--workspace")
    p_resume.set_defaults(func=cmd_resume)

    p_sessions = sub.add_parser("sessions", help="List saved chat sessions")
    p_sessions.add_argument("--limit", type=int, default=20)
    p_sessions.set_defaults(func=cmd_sessions)

    p_rollback = sub.add_parser("rollback", help="Rollback one file to previous saved snapshot")
    p_rollback.add_argument("path", help="Relative file path")
    p_rollback.add_argument("--workspace")
    p_rollback.set_defaults(func=cmd_rollback)

    p_doctor = sub.add_parser("doctor", help="Inspect running llama.cpp server details")
    p_doctor.set_defaults(func=cmd_doctor)

    p_vector = sub.add_parser("vector", help="Manage local vector store")
    p_vector_sub = p_vector.add_subparsers(dest="vector_cmd", required=True)

    v_init = p_vector_sub.add_parser("init", help="Initialize vector database")
    v_init.add_argument("--global", dest="global_mode", action="store_true")
    v_init.add_argument("--workspace")
    v_init.set_defaults(func=cmd_vector_init)

    v_index = p_vector_sub.add_parser("index", help="Index files into vector database")
    v_index.add_argument("--path", default=".")
    v_index.add_argument("--extensions", default=".py,.md,.txt,.json,.yaml,.yml,.toml,.js,.ts")
    v_index.add_argument("--max-files", type=int, default=2000)
    v_index.add_argument("--global", dest="global_mode", action="store_true")
    v_index.add_argument("--workspace")
    v_index.set_defaults(func=cmd_vector_index)

    v_query = p_vector_sub.add_parser("query", help="Query vector database")
    v_query.add_argument("query")
    v_query.add_argument("--top-k", type=int, default=5)
    v_query.add_argument("--global", dest="global_mode", action="store_true")
    v_query.add_argument("--workspace")
    v_query.set_defaults(func=cmd_vector_query)

    v_stats = p_vector_sub.add_parser("stats", help="Show vector database stats")
    v_stats.add_argument("--global", dest="global_mode", action="store_true")
    v_stats.add_argument("--workspace")
    v_stats.set_defaults(func=cmd_vector_stats)

    return parser


def main() -> int:
    parser = build_parser()
    known = {"init", "chat", "repl", "resume", "sessions", "rollback", "doctor", "vector"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        args = parser.parse_args(["chat", " ".join(argv)])
    else:
        args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["repl"])
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
