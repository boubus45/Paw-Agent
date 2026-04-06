from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from paw_agent.config import SKILLS_DIR


@dataclass
class SkillNote:
    path: Path
    title: str
    trigger: str
    guidance: str
    score: float = 0.0


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())}


class SkillStore:
    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = (skills_dir or SKILLS_DIR).resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        title: str,
        trigger: str,
        guidance: str,
        metadata: Dict[str, object] | None = None,
    ) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.strip().lower()).strip("-")[:48] or "skill"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.skills_dir / f"{stamp}-{safe}.json"
        payload = {
            "title": title.strip(),
            "trigger": trigger.strip(),
            "guidance": guidance.strip(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def retrieve(self, query: str, limit: int = 3) -> List[SkillNote]:
        q = _tokens(query)
        out: List[SkillNote] = []
        for p in sorted(self.skills_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            title = str(data.get("title", ""))
            trigger = str(data.get("trigger", ""))
            guidance = str(data.get("guidance", ""))
            base = f"{title}\n{trigger}\n{guidance}"
            t = _tokens(base)
            if not t:
                continue
            overlap = len(q & t)
            if overlap == 0:
                continue
            score = overlap / max(1, len(q))
            out.append(SkillNote(path=p, title=title, trigger=trigger, guidance=guidance, score=score))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]
