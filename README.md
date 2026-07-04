# JARVIS

> A genuinely autonomous agentic assistant that runs **free** — on Groq, NVIDIA,
> Cerebras, OpenRouter, Mistral, or a fully local Ollama. No paid API key required.

JARVIS is a real, working agent framework — not a chatbot wrapper. It plans,
acts with tools, verifies its own work, remembers across sessions, and can
delegate independent sub-tasks to parallel sub-agents. It is **provider-agnostic**:
it speaks the OpenAI-compatible API, so it works with every major free LLM
provider, and it ships with keyless web search so the free path stays capable.

```
   ██  █████  ██████  ██    ██ ██ ███████
   ██ ██   ██ ██   ██ ██    ██ ██ ██
   ██ ███████ ██████  ██    ██ ██ ███████
██ ██ ██   ██ ██   ██  ██  ██  ██      ██
 ████  ██   ██ ██   ██   ████   ██ ███████
        Just A Rather Very Intelligent System
```

## Free from day one

Pick any of these — all have a free tier or run locally, none need a paid card:

| Provider | Free? | Key env var | Get a key |
|----------|-------|-------------|-----------|
| **Groq** | free tier | `GROQ_API_KEY` | https://console.groq.com/keys |
| **NVIDIA NIM** | free credits | `NVIDIA_API_KEY` | https://build.nvidia.com |
| **Cerebras** | free tier | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |
| **OpenRouter** | free models | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| **Mistral** | free tier | `MISTRAL_API_KEY` | https://console.mistral.ai |
| **Ollama** | 100% local | *(none)* | https://ollama.com |

JARVIS auto-detects whichever key you have. Set one and go:

```bash
export GROQ_API_KEY=gsk_...      # or any of the above
python -m jarvis --autonomous "Audit this repo for TODOs and write REPORT.md"
```

No key at all? Run a model locally with Ollama:

```bash
# one-time: install ollama, then
ollama pull llama3.1
python -m jarvis          # auto-uses the local model, fully offline
```

See everything JARVIS detected: `python -m jarvis --list-providers`.

## What it can actually do

- **Reason and plan** — an explicit task list it maintains as it works, plus live
  reasoning display (for models that expose it — `reasoning_content` or `<think>` tags).
- **Touch the filesystem** — read, write, edit, list, glob, and grep, all confined
  to a workspace root (path-traversal guarded).
- **Run commands** — a sandboxed `bash` tool and a Python execution tool, with a
  catastrophic-command guard and a permission gate for interactive mode.
- **Use the web — free and keyless** — `web_search` (via DuckDuckGo) and `web_fetch`
  (HTML → readable text). Standard library only, no API key.
- **Remember** — a persistent SQLite memory store with full-text search, carried
  across sessions.
- **Delegate** — spawn focused sub-agents for independent workstreams.
- **Stay safe** — destructive actions gated; a hard blocklist stops the truly
  catastrophic regardless of mode.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # openai + rich + prompt_toolkit
# optional, only for --provider anthropic:  pip install anthropic
```

## Use

Interactive REPL:

```bash
python -m jarvis
```

One-shot autonomous objective:

```bash
python -m jarvis --autonomous "Refactor utils.py and run the tests"
```

Choose a provider/model explicitly:

```bash
python -m jarvis --provider groq  --model llama-3.3-70b-versatile "..."
python -m jarvis --provider ollama --model qwen2.5 "..."
python -m jarvis --provider nvidia "..."
```

Inspect / manage memory:

```bash
python -m jarvis mem list
python -m jarvis mem search "deployment"
```

## Configuration

Flags (run `python -m jarvis --help`) or env vars:

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--provider` | `JARVIS_PROVIDER` | auto-detect → ollama | groq / nvidia / cerebras / openrouter / mistral / ollama / anthropic / custom |
| `--model` | `JARVIS_MODEL` | provider default | Model id |
| `--base-url` | `JARVIS_BASE_URL` | provider default | Custom / self-hosted OpenAI-compatible endpoint |
| `--api-key-env` | — | provider default | Env var holding the key (for custom endpoints) |
| `--workspace` | `JARVIS_WORKSPACE` | cwd | Root the agent may touch |
| `--autonomous` | — | off | Skip confirmation prompts (still blocks catastrophic commands) |
| `--no-web` | — | web on | Disable the keyless web tools |
| `--max-iterations` | — | 50 | Agentic loop safety cap |

### Any OpenAI-compatible endpoint

Anything that speaks `/v1/chat/completions` works — LM Studio, llama.cpp's server,
vLLM, Together, etc.:

```bash
python -m jarvis --provider custom \
  --base-url http://localhost:8080/v1 \
  --api-key-env MY_KEY \
  --model my-local-model "..."
```

### Anthropic (optional)

Claude is one backend among many, off the free path:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m jarvis --provider anthropic --model claude-opus-4-8 "..."
```

## Architecture

```
jarvis/
  cli.py             argparse + interactive REPL + provider listing
  agent.py           the agentic loop (plan → act → verify), provider-agnostic
  config.py          config + provider registry + auto-detection
  prompts.py         system prompt (tuned for autonomous operation)
  ui.py              rich terminal rendering
  memory/store.py    SQLite + FTS persistent memory
  providers/
    base.py          neutral types (AssistantTurn, ToolCall) + Backend interface
    openai_compat.py Groq/NVIDIA/Ollama/… backend (streaming, tools, reasoning)
    anthropic_backend.py  optional Claude backend
  tools/
    filesystem.py    read / write / edit / ls / glob / grep
    shell.py         sandboxed bash
    python_exec.py   local Python execution
    web_client.py    keyless DuckDuckGo search + page fetch
    memory_tool.py   remember / recall / forget
    tasks.py         the working task list
    subagent.py      delegate to sub-agents
tests/               pytest suite (53 tests, no network)
```

The agent talks to a provider-neutral `Backend`. Each backend owns its native
message history and translates to/from a small neutral surface, so adding a new
provider is one file.

## Safety

JARVIS is powerful by design. Run it in a workspace you control.

- All filesystem access is confined to the workspace root.
- A hard blocklist rejects catastrophic commands (`rm -rf /`, fork bombs, disk
  wipes, …) in every mode.
- In interactive mode, writes, shell commands, and code execution require
  confirmation. `--autonomous` skips the prompts but keeps the hard blocklist.

## License

MIT
