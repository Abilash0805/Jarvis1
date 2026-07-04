"""Command-line interface: one-shot runs, an interactive REPL, and memory ops."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import DETECT_ORDER, EFFORT_LEVELS, PROVIDERS, Config, detect_provider
from .memory import Memory
from .providers import ProviderError, make_backend
from .ui import Console


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — a fully autonomous agentic assistant. Runs free on Groq, "
        "NVIDIA, Cerebras, OpenRouter, Mistral, or a local Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Memory:  jarvis mem list | search <q> | add <text> | forget <id>\n"
        "Providers:  jarvis --list-providers",
    )
    p.add_argument("objective", nargs="*", help="Objective to run once, then exit. Omit for a REPL.")
    p.add_argument("--provider", default=None, help="Provider (default: auto-detect from your keys, else ollama)")
    p.add_argument("--model", default=None, help="Model id (default: the provider's default)")
    p.add_argument("--base-url", default=None, help="Override the provider endpoint (for custom/self-hosted)")
    p.add_argument("--api-key-env", default=None, help="Env var holding the API key (for custom providers)")
    p.add_argument("--effort", default=None, choices=EFFORT_LEVELS, help="Reasoning effort (Anthropic only)")
    p.add_argument("--workspace", default=None, help="Root directory the agent may touch (default: cwd)")
    p.add_argument("--autonomous", action="store_true", help="Skip confirmation prompts (still blocks catastrophic commands)")
    p.add_argument("--no-web", action="store_true", help="Disable keyless web search/fetch")
    p.add_argument("--no-subagents", action="store_true", help="Disable delegation to sub-agents")
    p.add_argument("--max-iterations", type=int, default=None, help="Agentic loop safety cap (default 50)")
    p.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per turn")
    p.add_argument("--quiet", action="store_true", help="Suppress tool-result panels")
    p.add_argument("--no-thinking", action="store_true", help="Do not stream the model's reasoning")
    p.add_argument("--list-providers", action="store_true", help="List known providers and exit")
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    return p


def build_mem_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis mem", description="Inspect or edit persistent memory.")
    p.add_argument("action", choices=["list", "search", "add", "forget"])
    p.add_argument("args", nargs="*", help="Operation arguments")
    p.add_argument("--workspace", default=None, help="Workspace whose memory to use (default: cwd)")
    return p


def make_config(ns: argparse.Namespace) -> Config:
    return Config(
        provider=ns.provider,
        model=ns.model,
        base_url=ns.base_url,
        api_key_env=ns.api_key_env,
        effort=ns.effort or "high",
        max_tokens=ns.max_tokens or 4096,
        workspace=Path(ns.workspace) if ns.workspace else Path.cwd(),
        autonomous=bool(ns.autonomous),
        enable_web=not ns.no_web,
        enable_subagents=not ns.no_subagents,
        max_iterations=ns.max_iterations or 50,
    )


# -- listings ---------------------------------------------------------------

def list_providers(console: Console) -> int:
    detected = detect_provider()
    console.rich.print("[bold]Available providers[/bold] (auto-detect order shown first):\n")
    order = DETECT_ORDER + [n for n in PROVIDERS if n not in DETECT_ORDER]
    for name in order:
        p = PROVIDERS[name]
        key = f"needs {p.env_key}" if p.env_key else "keyless"
        free = "[green]free[/green]" if p.free else "[yellow]paid[/yellow]"
        mark = " [cyan]← detected[/cyan]" if name == detected else ""
        console.rich.print(f"  [bold]{name:11}[/bold] {free:16} {key:22} {p.default_model}{mark}")
    console.rich.print("\nUse:  jarvis --provider <name> [--model <id>]")
    console.rich.print("Set the key first, e.g.:  export GROQ_API_KEY=...")
    return 0


# -- memory subcommand ------------------------------------------------------

def run_mem(ns: argparse.Namespace, cfg: Config, console: Console) -> int:
    cfg.ensure_dirs()
    mem = Memory(cfg.memory_path)
    action = ns.action
    if action == "list":
        recs = mem.list(limit=200)
        if not recs:
            console.info("Memory is empty.")
        for r in recs:
            console.rich.print(r.render())
    elif action == "search":
        query = " ".join(ns.args)
        if not query:
            console.error("Usage: jarvis mem search <query>")
            return 2
        for r in mem.search(query, limit=20):
            console.rich.print(r.render())
    elif action == "add":
        text = " ".join(ns.args)
        if not text:
            console.error("Usage: jarvis mem add <text>")
            return 2
        console.info(f"Added memory #{mem.add(text)}.")
    elif action == "forget":
        if not ns.args or not ns.args[0].isdigit():
            console.error("Usage: jarvis mem forget <id>")
            return 2
        console.info("Forgotten." if mem.forget(int(ns.args[0])) else "No such memory.")
    mem.close()
    return 0


# -- agent entry points -----------------------------------------------------

def _make_agent(cfg: Config, console: Console) -> tuple[Agent, Memory]:
    cfg.ensure_dirs()
    mem = Memory(cfg.memory_path)
    backend = make_backend(cfg)
    return Agent(cfg, backend, mem, console), mem


def _preflight(agent: Agent, console: Console) -> bool:
    """Return True if the backend looks ready; print guidance and return False otherwise."""
    problem = agent.backend.health_check()
    if problem:
        console.error(problem)
        return False
    return True


def run_oneshot(objective: str, cfg: Config, console: Console) -> int:
    agent, mem = _make_agent(cfg, console)
    try:
        if not _preflight(agent, console):
            return 1
        agent.run(objective)
    except KeyboardInterrupt:
        console.warn("Interrupted.")
        return 130
    except ProviderError as exc:
        console.error(str(exc))
        return 1
    finally:
        mem.close()
    console.rule("done")
    return 0


def run_repl(cfg: Config, console: Console) -> int:
    console.banner(
        f"provider={cfg.provider} ({cfg.label})  model={cfg.model}  "
        f"mode={'autonomous' if cfg.autonomous else 'interactive'}  ws={cfg.workspace}"
    )
    console.info("Type your objective. Commands: /help /reset /tasks /memory /exit\n")

    agent, mem = _make_agent(cfg, console)
    if not _preflight(agent, console):
        mem.close()
        return 1
    session = _input_session()
    try:
        while True:
            try:
                line = session().strip()
            except (EOFError, KeyboardInterrupt):
                console.rich.print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(line, agent, console):
                    break
                continue
            try:
                agent.send(line)
            except KeyboardInterrupt:
                console.warn("(interrupted — send another message or /exit)")
            except ProviderError as exc:
                console.error(str(exc))
    finally:
        mem.close()
    console.info("Goodbye.")
    return 0


def _handle_command(line: str, agent: Agent, console: Console) -> bool:
    cmd = line[1:].strip().lower()
    if cmd in ("exit", "quit", "q"):
        return True
    if cmd in ("help", "h", "?"):
        console.rich.print(
            "[bold]Commands[/bold]\n"
            "  /reset   start a fresh conversation\n"
            "  /tasks   show the current task list\n"
            "  /memory  show recent persistent memories\n"
            "  /exit    quit"
        )
    elif cmd == "reset":
        agent.reset()
        console.info("Conversation reset.")
    elif cmd == "tasks":
        console.render_tasks(agent.state.get("tasks", []))
    elif cmd == "memory":
        recs = agent.memory.list(limit=20)
        if not recs:
            console.info("Memory is empty.")
        for r in recs:
            console.rich.print(r.render())
    else:
        console.warn(f"Unknown command: {line}")
    return False


def _input_session():
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import ANSI

        ps = PromptSession()
        return lambda: ps.prompt(ANSI("\033[1;32myou ›\033[0m "))
    except Exception:  # pragma: no cover
        return lambda: input("you › ")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "mem":
        ns = build_mem_parser().parse_args(argv[1:])
        ws = Path(ns.workspace) if ns.workspace else Path.cwd()
        return run_mem(ns, Config(workspace=ws, state_dir=ws / ".jarvis"), Console())

    ns = build_parser().parse_args(argv)
    console = Console(quiet=ns.quiet, show_thinking=not ns.no_thinking)
    if ns.list_providers:
        return list_providers(console)

    cfg = make_config(ns)
    objective = " ".join(ns.objective).strip()
    if objective:
        return run_oneshot(objective, cfg, console)
    return run_repl(cfg, console)


if __name__ == "__main__":
    sys.exit(main())
