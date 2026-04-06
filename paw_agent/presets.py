from __future__ import annotations

from typing import Dict


MODEL_PRESETS: Dict[str, Dict[str, object]] = {
    "qwen3.5-2b-instruct": {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 512,
    },
    "qwen3.5-4b-instruct": {
        "temperature": 0.12,
        "top_p": 0.92,
        "max_tokens": 512,
    },
    "gemma-4-e2b-it": {
        "temperature": 0.08,
        "top_p": 0.9,
        "max_tokens": 512,
    },
}


def apply_preset(cfg: Dict[str, object], model_name: str) -> bool:
    preset = MODEL_PRESETS.get(model_name)
    if not preset:
        return False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["model"] = model_name
    for k in ("temperature", "top_p", "max_tokens"):
        if k in preset:
            model_cfg[k] = preset[k]
    return True
