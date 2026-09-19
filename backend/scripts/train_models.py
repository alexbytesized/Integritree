"""Model training: command guard only; implement the workflow in Phase 3."""

from integritree.commands import pending_command

if __name__ == "__main__":
    raise SystemExit(pending_command("Model training", 3, "train"))
