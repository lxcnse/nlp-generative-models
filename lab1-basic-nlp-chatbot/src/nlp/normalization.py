from __future__ import annotations

import re
import unicodedata

# Ujednolicenie "ozdobnych" znakow typograficznych do prostych odpowiednikow.
_CHAR_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
        "\u00a0": " ",
    }
)
# 'ł' nie rozkłada sie w Unicode (NFD), wiec mapujemy recznie.
_DIACRITIC_EXTRA = str.maketrans({"ł": "l", "Ł": "L"})
_WHITESPACE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    text = text.translate(_DIACRITIC_EXTRA)
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str, remove_diacritics: bool = False) -> str:
    """NFKC -> ujednolicenie znakow -> casefold -> biale znaki (-> bez ogonkow)."""
    text = unicodedata.normalize("NFKC", text)   # np. pełnoszerokie znaki -> zwykle
    text = text.translate(_CHAR_MAP)
    text = text.casefold()                        # agresywniejsze niz lower()
    if remove_diacritics:
        text = strip_diacritics(text)
    return _WHITESPACE.sub(" ", text).strip()
