from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


APP_HOME = Path(os.getenv("PAW_AGENT_HOME", Path.cwd() / ".paw-agent"))
CONFIG_PATH = APP_HOME / "config.yaml"
SKILLS_DIR = APP_HOME / "skills"
SESSIONS_DIR = APP_HOME / "sessions"


DEFAULT_CONFIG: Dict[str, Any] = {
    "agent": {
        "name": "Paw-Agent",
        "max_steps": 22,
        "workspace": ".",
        "auto_reflect": True,
        "skill_capture_min_tool_calls": 5,
        "auto_vector_retrieval": True,
        "vector_top_k": 4,
        "vector_max_chars": 2600,
        "vector_include_global": False,
    },
    "model": {
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "auto",
        "temperature": 0.15,
        "top_p": 0.9,
        "max_tokens": 512,
        "request_timeout_sec": 300,
    },
}


def ensure_app_home() -> None:
    APP_HOME.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    ensure_app_home()
    cfg = yaml.safe_load(yaml.safe_dump(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _sanitize_user_config(user)
        _merge(cfg, user)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    ensure_app_home()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def init_config() -> Path:
    ensure_app_home()
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
    else:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        before = yaml.safe_dump(user, sort_keys=True)
        _sanitize_user_config(user)
        after = yaml.safe_dump(user, sort_keys=True)
        if before != after:
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                yaml.safe_dump(user, f, sort_keys=False, allow_unicode=False)
    return CONFIG_PATH


def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def _sanitize_user_config(user: Dict[str, Any]) -> None:
    user.pop("llamacpp", None)
    agent = user.get("agent")
    if isinstance(agent, dict):
        agent.pop("use_dual_model_router", None)
    model = user.get("model")
    if isinstance(model, dict):
        model.pop("fast_model", None)
        model.pop("strong_model", None)
