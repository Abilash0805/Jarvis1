"""Command-line interface: one-shot runs, an interactive REPL, and memory ops."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from . import __version__
from .agent import Agent
from .config import DEFAULT_MODEL, EFFORT_LEVELS, Config
from .llm import LLMClient
from .memory import Memory
from .ui import Console


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — a fully autonomous agentic assistant built on the Claude API.",
    )
    p.add_argument("objective", nargs="*", help="Objective to run once, then exit. Omit for an interactive session.")
    p.add_argument("--model", default=None, help=f"Claude model (default {DEFAULT_MODEL}; use claude-fable-5 for max capability)")
    p.add_argument("--effort", default=None, choices=EFFORT_LEVELS, help="Reasoning effort (default high)")
    p.add_argument("--workspace", default=None, help="Root directory the agent may touch (default: cwd)")
    p.add_argument("--autonomous", action="store_true", help="Skip confirmation prompts (still blocks catastrophic commands)")
    p.add_argument("--no-web", action="store_true", help="Disable server-side web search/fetch")
    p.add_argument("--no-subagents", action="store_true", help="Disable delegation to sub-agents")
    p.add_argument("--max-iterations", type=int, default=None, help="Agentic loop safety cap (default 50)")
    p.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per turn (default 32000)")
    p.add_argument("--quiet", action="store_true", help="Suppress tool-result panels")
    p.add_argument("--no-thinking", action="store_true", help="Do not stream the model's reasoning")
    p.add_argument("--version", action="version", version=f"jarvis {__version__}")
    p.epilog = "Memory: `jarvis mem list | search <q> | add <text> | forget <id>`"
    return p


def build_mem_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis mem", description="Inspect or edit persistent memory.")
    p.add_argument("action", choices=["list", "search", "add", "forget"])
    p.add_argument("args", nargs="*", help="Operation arguments")
    p.add_argument("--workspace", default=None, help="Workspace whose memory to use (default: cwd)")
    return p


def make_config(ns: argparse.Namespace) -> Config:
    cfg = Config()
    if ns.model:
        cfg.model = ns.model
    if ns.effort:
        cfg.effort = ns.effort
    if ns.workspace:
        cfg.workspace = Path(ns.workspace)
    if ns.max_iterations:
        cfg.max_iterations = ns.max_iterations
    if ns.max_tokens:
        cfg.max_tokens = ns.max_tokens
    cfg.autonomous = bool(ns.autonomous)
    cfg.enable_web = not ns.no_web
    cfg.enable_subagents = not ns.no_subagents
    # Re-resolve derived fields (model caps, effort clamp, resolved paths).
    return Config(
        model=cfg.model,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
        workspace=cfg.workspace,
        state_dir=cfg.workspace / ".jarvis",
        autonomous=cfg.autonomous,
        enable_web=cfg.enable_web,
        enable_subagents=cfg.enable_subagents,
        max_iterations=cfg.max_iterations,
    )


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
        recs = mem.search(query, limit=20)
        if not recs:
            console.info("No matches.")
        for r in recs:
            console.rich.print(r.render())
    elif action == "add":
        text = " ".join(ns.args)
        if not text:
            console.error("Usage: jarvis mem add <text>")
            return 2
        mid = mem.add(text)
        console.info(f"Added memory #{mid}.")
    elif action == "forget":
        if not ns.args or not ns.args[0].isdigit():
            console.error("Usage: jarvis mem forget <id>")
            return 2
        ok = mem.forget(int(ns.args[0]))
        console.info("Forgotten." if ok else "No such memory.")
    mem.close()
    return 0


# -- agent entry points -----------------------------------------------------

def _make_agent(cfg: Config, console: Console) -> tuple[Agent, Memory]:
    cfg.ensure_dirs()
    mem = Memory(cfg.memory_path)
    llm = LLMClient(cfg)
    agent = Agent(cfg, llm, mem, console)
    return agent, mem


def _is_auth_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("authentication", "api_key", "auth_token", "x-api-key"))


def run_oneshot(objective: str, cfg: Config, console: Console) -> int:
    agent, mem = _make_agent(cfg, console)
    try:
        result = agent.run(objective)
    except KeyboardInterrupt:
        console.warn("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — surface a clean message, not a traceback
        if _is_auth_error(exc):
            _auth_help(console)
            return 1
        console.error(f"Run failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        mem.close()
    if result:
        console.rule("done")
    return 0


def run_repl(cfg: Config, console: Console) -> int:
    console.banner(
        f"model={cfg.model}  effort={cfg.effort}  "
        f"mode={'autonomous' if cfg.autonomous else 'interactive'}  ws={cfg.workspace}"
    )
    if not cfg.has_api_credentials():
        console.info("No ANTHROPIC_API_KEY in env — will use an `ant auth login` profile if present.")
    console.info("Type your objective. Commands: /help /reset /tasks /memory /exit\n")

    agent, mem = _make_agent(cfg, console)
    session = _input_session()
    try:
        while True:
            try:
                line = session()
            except (EOFError, KeyboardInterrupt):
                console.rich.print()
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if _handle_command(line, agent, cfg, console):
                    break
                continue
            try:
                agent.send(line)
            except KeyboardInterrupt:
                console.warn("(interrupted — send another message or /exit)")
            except Exception as exc:  # noqa: BLE001
                if _is_auth_error(exc):
                    _auth_help(console)
                    break
                console.error(f"{type(exc).__name__}: {exc}")
    finally:
        mem.close()
    console.info("Goodbye.")
    return 0


def _handle_command(line: str, agent: Agent, cfg: Config, console: Console) -> bool:
    """Handle a /command. Returns True if the REPL should exit."""
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
        agent.messages = []
        agent.state = {}
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
    """Return a callable that reads one line of input (prompt_toolkit if available)."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import ANSI

        ps = PromptSession()
        return lambda: ps.prompt(ANSI("\033[1;32myou ›\033[0m "))
    except Exception:
        return lambda: input("you › ")


def _auth_help(console: Console) -> None:
    console.error("Authentication failed.")
    console.info(
        "Set ANTHROPIC_API_KEY, or run `ant auth login` to use an OAuth profile. "
        "See https://platform.claude.com/ for an API key."
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Route the `mem` subcommand explicitly — argparse can't disambiguate a
    # free-form `objective` positional from a subparser.
    if argv and argv[0] == "mem":
        ns = build_mem_parser().parse_args(argv[1:])
        console = Console()
        cfg = Config(workspace=Path(ns.workspace) if ns.workspace else Path.cwd())
        cfg = Config(workspace=cfg.workspace, state_dir=cfg.workspace / ".jarvis")
        return run_mem(ns, cfg, console)

    ns = build_parser().parse_args(argv)
    console = Console(quiet=ns.quiet, show_thinking=not ns.no_thinking)
    cfg = make_config(ns)

    objective = " ".join(ns.objective).strip()
    if objective:
        return run_oneshot(objective, cfg, console)
    return run_repl(cfg, console)


if __name__ == "__main__":
    sys.exit(main())
