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


# --------------------------------------------------------------- zadanie 2
# Telegram (i telefony) czesto zamieniaja " na cudzyslowy typograficzne.
_QUOTES = str.maketrans({"\u201e": '"', "\u201d": '"', "\u201c": '"', "\u00ab": '"',
                         "\u00bb": '"'})


@dataclass(frozen=True)
class ClassifierArgs:
    cleaning: str
    text: str
    repr_kind: str
    method: str


CLASSIFIER_USAGE = (
    'Uzycie: /classifier cleaning "tekst" bow|tfidf|embedding metoda\n'
    'np. /classifier cleaning "Napisac raport z laboratorium" tfidf adaboost'
)


def parse_classifier_args(raw: str) -> ClassifierArgs:
    """'cleaning "Kupic mleko" tfidf tree' -> ClassifierArgs.

    Tekst powinien byc w cudzyslowie, ale dzialamy tez bez niego: pierwszy wyraz to
    tryb czyszczenia, dwa ostatnie to reprezentacja i metoda, a reszta to tekst.
    """
    raw = raw.translate(_QUOTES).strip()
    quoted = re.fullmatch(r'(\S+)\s+"(.*)"\s+(\S+)\s+(\S+)', raw, flags=re.DOTALL)
    if quoted:
        cleaning, text, repr_kind, method = quoted.groups()
    else:
        parts = raw.split()
        if len(parts) < 4:
            raise InvalidInputError(CLASSIFIER_USAGE)
        cleaning, repr_kind, method = parts[0], parts[-2], parts[-1]
        text = " ".join(parts[1:-2]).strip('"')
    if not text.strip():
        raise InvalidInputError(CLASSIFIER_USAGE)
    return ClassifierArgs(cleaning.lower(), text.strip(), repr_kind.lower(), method.lower())


def parse_repr_method(raw: str, command: str) -> tuple[str, str]:
    """'tfidf adaboost' -> ('tfidf', 'adaboost')."""
    parts = raw.split()
    if len(parts) != 2:
        raise InvalidInputError(f"Uzycie: /{command} bow|tfidf|embedding metoda (np. tfidf tree)")
    return parts[0].lower(), parts[1].lower()


def parse_prepare_args(raw: str) -> tuple[str | None, bool]:
    """'todo --seed' -> ('todo', True); '' -> (None, False). Akceptuje tez 'a / b' z polecenia."""
    seed = "--seed" in raw.split()
    words = [w for w in raw.replace("/", " ").split() if w != "--seed"]
    if len(words) > 1:
        raise InvalidInputError(
            "Podaj jeden zbior: /prepare_data_for_classification [todo|agnews|20newsgroups] [--seed]"
        )
    return (words[0] if words else None), seed
