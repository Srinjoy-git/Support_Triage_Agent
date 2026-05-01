"""Shared utility helpers for the triage agent."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
SUPPORT_TICKETS_DIR = ROOT_DIR / "support_tickets"
SUPPORT_TICKETS_PATH = SUPPORT_TICKETS_DIR / "support_tickets.csv"
OUTPUT_PATH = SUPPORT_TICKETS_DIR / "output.csv"
CONFIG_PATH = CODE_DIR / "rules.json"


def normalize_text(value: Any) -> str:
    """Convert raw values into clean single-spaced text."""
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def load_rules_config() -> dict[str, Any]:
    """Load keyword rules and thresholds from code/rules.json."""
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open(mode="r", encoding="utf-8") as config_file:
        return json.load(config_file)


def setup_logging() -> logging.Logger:
    """Configure simple console logging for the terminal tool."""
    logger = logging.getLogger("support_triage")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def split_ticket_issues(ticket_text: Any) -> list[str]:
    """Split tickets that contain multiple requests into issue fragments."""
    raw_ticket = "" if ticket_text is None else str(ticket_text).strip()

    if not raw_ticket:
        return []

    split_pattern = (
        r"(?:\s*[\r\n]+\s*)|"
        r"(?:\s*;\s*)|"
        r"(?:\s+-\s+)|"
        r"(?:\s+\d+[.)]\s+)|"
        r"(?:\s+(?:and also|also|additionally|plus)\s+)"
    )
    fragments = [
        normalize_text(fragment)
        for fragment in re.split(split_pattern, raw_ticket)
    ]
    meaningful = [fragment for fragment in fragments if len(fragment.split()) >= 2]

    return meaningful or [normalize_text(raw_ticket)]
