"""Batch prediction: command guard only; implement the workflow in Phase 5."""

from integritree.commands import pending_command

if __name__ == "__main__":
    raise SystemExit(pending_command("Batch prediction", 5, None))
