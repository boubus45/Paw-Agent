# Paw-Agent

Paw-Agent is a local-first CLI coding agent for low-spec laptops and small models served by `llama.cpp`.
It is designed to enforce concise, tool-driven execution: inspect files, plan step-by-step edits, validate, and summarize.

## Features

- Interactive CLI (`paw`) with Hermes-style workflow.
- Multi-step tool loop for coding tasks.
- Built-in tools: `read_file`, `write_file`, `rollback_file`, `list_files`, `search`, `run_shell`, `web_search`.
- Windows-aware command tools: `run_cmd`, `run_powershell`, plus generic `run_shell`.
- Auto vector retrieval from indexed project files.
- Per-file rollback snapshots before writes.
- Per-project sessions and skills (isolated by folder).
- Optional global vector database.

## Requirements

- Windows, macOS, or Linux with Python `>=3.10`.
- A running `llama.cpp` server exposing OpenAI-compatible endpoint (for example `http://127.0.0.1:8080/v1`).

## Quick Start

```powershell
git clone <YOUR_REPO_URL>
cd Paw-Agent
python -m pip install -e .
paw init
paw doctor
paw
```

If editable install is blocked, run directly:

```powershell
.\paw.ps1 init
.\paw.ps1
```

## Core Commands

```text
paw                       # interactive mode
paw "task prompt"         # one-shot task
paw doctor                # server/model/context introspection
paw sessions              # list saved chat sessions
paw resume <session-id>   # resume chat
paw rollback <path>       # rollback previous snapshot for one file
```

Inside interactive mode:

```text
/help
/status
/doctor
/skills
/sessions
/vector
/rollback <path>
/exit
```

## Configuration

Config is created in the current project at:

```text
./.paw-agent/config.yaml
```

Default model config:

- `base_url: http://127.0.0.1:8080/v1`
- `model: auto` (detect from server)
- `request_timeout_sec: 300`
- `max_tokens: 512`

This repo does not commit local config/state (`.paw-agent/` is ignored).

## Vector Database

Per-project vector DB (default):

```powershell
paw vector init
paw vector index --path . --extensions .py,.md,.txt,.json
paw vector query "where is resume implemented" --top-k 5
paw vector stats
```

Optional global DB:

```powershell
paw vector init --global
paw vector index --global --path .
paw vector query --global "shared coding pattern"
```

Auto retrieval is enabled by default in the agent loop. Paw-Agent injects top relevant chunks into context before planning and tool calls.

## Privacy and Safety

- Local state is stored under `./.paw-agent/` by default.
- File writes create rollback snapshots under `./.paw-agent/history/`.
- Web search is tool-based and only used when invoked by the agent plan.

## Launch From Any Folder

Add repo folder to `PATH` so `paw` works anywhere.

Then run `paw` from any project folder. Each folder gets isolated local state.

Inspect server-reported llama.cpp details:

```powershell
paw doctor
```
