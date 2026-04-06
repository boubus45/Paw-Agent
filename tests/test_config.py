from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil


def test_init_config_creates_file() -> None:
    base = Path(".test_tmp") / "config_home"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    os.environ["PAW_AGENT_HOME"] = str(base / ".paw-agent")
    import paw_agent.config as config

    importlib.reload(config)
    path = config.init_config()
    assert path.exists()
    cfg = config.load_config()
    assert cfg["agent"]["name"] == "Paw-Agent"
    assert "llamacpp" not in cfg
    assert "model" in cfg
