from jarvis.tools.shell import Bash, is_catastrophic


def test_catastrophic_patterns_blocked():
    assert is_catastrophic("rm -rf /")
    assert is_catastrophic("rm -rf /*")
    assert is_catastrophic(":(){ :|:& };:")
    assert is_catastrophic("dd if=/dev/zero of=/dev/sda")
    assert is_catastrophic("shutdown now")


def test_benign_commands_allowed():
    assert is_catastrophic("ls -la") is None
    assert is_catastrophic("rm -rf ./build") is None
    assert is_catastrophic("git status") is None


def test_bash_refuses_catastrophic(ctx):
    r = Bash().run({"command": "rm -rf /"}, ctx)
    assert r.is_error
    assert "Refused" in r.content


def test_bash_runs_benign(ctx):
    r = Bash().run({"command": "echo hello-jarvis"}, ctx)
    assert not r.is_error
    assert "hello-jarvis" in r.content


def test_bash_reports_nonzero_exit(ctx):
    r = Bash().run({"command": "exit 3"}, ctx)
    assert not r.is_error  # a non-zero exit is information, not a tool failure
    assert "exit 3" in r.content


def test_bash_permission_gate_denied(ctx):
    ctx.autonomous = False
    ctx.confirm = lambda action: False
    r = Bash().run({"command": "echo nope"}, ctx)
    assert r.is_error
    assert "denied" in r.content.lower()
