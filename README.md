# JARVIS

> A genuinely autonomous agentic assistant, built on the Anthropic Claude API.

JARVIS is a real, working agent framework — not a chatbot wrapper. It plans,
acts with tools, verifies its own work, remembers across sessions, and can
delegate independent sub-tasks to parallel sub-agents. It runs an open-ended
agentic loop powered by Claude's adaptive thinking and tool use.

```
   ██  █████  ██████  ██    ██ ██ ███████
   ██ ██   ██ ██   ██ ██    ██ ██ ██
   ██ ███████ ██████  ██    ██ ██ ███████
██ ██ ██   ██ ██   ██  ██  ██  ██      ██
 ████  ██   ██ ██   ██   ████   ██ ███████
        Just A Rather Very Intelligent System
```

## What it can actually do

- **Reason and plan** — adaptive extended thinking, effort control, an explicit
  task list it maintains as it works.
- **Touch the filesystem** — read, write, edit, list, glob, and grep, all
  confined to a workspace root (path-traversal guarded).
- **Run commands** — a sandboxed `bash` tool and a Python execution tool, with a
  catastrophic-command guard and a permission gate for interactive mode.
- **Use the web** — Claude's server-side web search and web fetch (current info,
  citations, dynamic filtering).
- **Remember** — a persistent SQLite memory store with full-text search, so it
  carries lessons and facts across sessions.
- **Delegate** — spawn focused sub-agents for independent workstreams and
  collect their results.
- **Stay safe** — destructive actions are gated; a hard blocklist stops the
  truly catastrophic regardless of mode.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # or use `ant auth login`
```

## Use

Interactive REPL:

```bash
python -m jarvis
```

One-shot objective (autonomous):

```bash
python -m jarvis --autonomous "Audit this repo for TODOs and write a summary to REPORT.md"
```

Inspect / manage memory:

```bash
python -m jarvis mem list
python -m jarvis mem search "deployment"
```

## Configuration

Everything is a flag (run `python -m jarvis --help`) or an env var:

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--model` | `JARVIS_MODEL` | `claude-opus-4-8` | Claude model. Set to `claude-fable-5` for maximum capability. |
| `--effort` | `JARVIS_EFFORT` | `high` | `low` / `medium` / `high` / `xhigh` / `max` |
| `--workspace` | `JARVIS_WORKSPACE` | cwd | Root the agent is allowed to touch |
| `--autonomous` | — | off | Skip confirmation prompts (still blocks catastrophic commands) |
| `--no-web` | — | web on | Disable server-side web tools |
| `--max-iterations` | — | 50 | Agentic loop safety cap |

### Maximum capability

The default model is `claude-opus-4-8`. To run on Anthropic's most capable
model, Claude Fable 5 — the best choice for the hardest long-horizon work:

```bash
python -m jarvis --model claude-fable-5 "..."
```

JARVIS handles Fable 5's specifics automatically: thinking is always on, and a
server-side refusal fallback to `claude-opus-4-8` is enabled so a false-positive
safety refusal never fails the run outright.

## Architecture

```
jarvis/
  cli.py          argparse + interactive REPL
  agent.py        the agentic loop (plan → act → verify)
  llm.py          Anthropic client wrapper (streaming, thinking, effort, fallback)
  config.py       config + model capability resolution
  prompts.py      system prompt (tuned for autonomous operation)
  ui.py           rich terminal rendering
  memory/store.py SQLite + FTS persistent memory
  tools/          the tool surface
    filesystem.py read / write / edit / ls / glob / grep
    shell.py      sandboxed bash
    python_exec.py local Python execution
    web.py        server-side web search + fetch declarations
    memory_tool.py remember / recall / forget
    tasks.py      the working task list
    subagent.py   delegate to sub-agents
tests/            pytest suite
```

## Safety

JARVIS is powerful by design. It is meant to run in a workspace you control.

- All filesystem access is confined to the workspace root.
- A hard blocklist rejects catastrophic commands (`rm -rf /`, fork bombs,
  disk wipes, …) in every mode.
- In interactive (non-`--autonomous`) mode, writes, shell commands, and code
  execution require confirmation.
- Nothing leaves your machine except Claude API calls and any web
  search/fetch you enable.

## License

MIT
