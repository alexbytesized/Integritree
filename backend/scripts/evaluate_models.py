"""Research evaluation: command guard only; implement the workflow in Phase 4."""

from integritree.commands import pending_command

if __name__ == "__main__":
    raise SystemExit(pending_command("Research evaluation", 4, "evaluate"))
