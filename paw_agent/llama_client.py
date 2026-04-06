from __future__ import annotations

import json
from typing import Any, Dict, List

import requests


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        request_timeout_sec: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        # Fast connect timeout + configurable read timeout for slower local inference.
        resp = requests.post(url, json=payload, timeout=(8, self.request_timeout_sec))
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def discover_model_id(base_url: str, timeout_sec: int = 5) -> str:
        info = LlamaCppClient.fetch_server_info(base_url, timeout_sec=timeout_sec)
        model = str(info.get("model", "")).strip()
        if model:
            return model
        raise ValueError("No model id returned by llama.cpp server")

    @staticmethod
    def fetch_server_info(base_url: str, timeout_sec: int = 5) -> Dict[str, Any]:
        out: Dict[str, Any] = {"status": "offline", "model": None, "ctx": None}
        root = base_url.rstrip("/")

        def _get_json(url: str) -> Dict[str, Any] | List[Any] | None:
            try:
                r = requests.get(url, timeout=timeout_sec)
                if not r.ok:
                    return None
                return r.json()
            except Exception:
                return None

        models = _get_json(f"{root}/models")
        if isinstance(models, dict):
            data = models.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    out["model"] = first.get("id") or out["model"]
                    out["status"] = "online"

        # llama.cpp native info endpoints (best effort; may vary by build).
        base_no_v1 = root[:-3] if root.endswith("/v1") else root
        props = _get_json(f"{base_no_v1}/props")
        if isinstance(props, dict):
            out["status"] = "online"
            out["ctx"] = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
            model_path = props.get("model_path")
            if not out.get("model") and isinstance(model_path, str) and model_path:
                out["model"] = model_path.split("\\")[-1].split("/")[-1]

        slots = _get_json(f"{base_no_v1}/slots")
        if isinstance(slots, list) and slots:
            out["status"] = "online"
            first_slot = slots[0]
            if isinstance(first_slot, dict):
                out["ctx"] = out.get("ctx") or first_slot.get("n_ctx")
                slot_model = first_slot.get("model")
                if not out.get("model") and isinstance(slot_model, str) and slot_model:
                    out["model"] = slot_model

        return out

    @staticmethod
    def parse_json_block(text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(stripped[start : end + 1])
