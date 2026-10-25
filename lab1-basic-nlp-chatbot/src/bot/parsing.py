"""Parsowanie argumentow komend - czyste funkcje, latwe do przetestowania bez Telegrama."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.errors import InvalidInputError
from src.storage.models import PRIORITIES

_PARENT = re.compile(r"(?:^|\s)--parent\s+(\S+)")
_PRIO = re.compile(r"(?:^|\s)--prio\s+(\S+)")


@dataclass(frozen=True)
class AddArgs:
    description: str
    parent: Optional[str]
    priority: str


def parse_add_args(raw: str) -> AddArgs:
    """'Kupic mleko --parent 550e8400 --prio high' -> AddArgs(...)."""
    parent_match = _PARENT.search(raw)
    prio_match = _PRIO.search(raw)
    priority = prio_match.group(1).lower() if prio_match else "medium"
    if priority not in PRIORITIES:
        raise InvalidInputError(f"Priorytet musi byc jednym z: {', '.join(PRIORITIES)}")
    description = _PARENT.sub(" ", _PRIO.sub(" ", raw))
    description = " ".join(description.split())
    if not description:
        raise InvalidInputError("Podaj tresc zadania, np. /add Kupic mleko")
    return AddArgs(description, parent_match.group(1) if parent_match else None, priority)


def split_first_word(raw: str) -> tuple[str, str]:
    """'abc def ghi' -> ('abc', 'def ghi'); brak drugiej czesci -> ''."""
    parts = raw.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0], (parts[1].strip() if len(parts) > 1 else "")
