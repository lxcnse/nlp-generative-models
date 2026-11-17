"""czyszczenie tekstu"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
# Wszystko co NIE jest litera (cyfry, interpunkcja, znaki specjalne, '_', emoji) -> spacja.
# [^\W\d_] oznacza "litera w Unicode", wiec polskie znaki (ą, ę, ł...) zostaja.
_NON_LETTERS = re.compile(r"[\W\d_]+", flags=re.UNICODE)
_SINGLE_CHAR = re.compile(r"\b\w\b", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, remove_single_chars: bool = True) -> str:
    """Zwraca oczyszczony tekst.

    Kroki (kolejnosc ma znaczenie):
      1. male litery,
      2. usuniecie adresow URL (zanim interpunkcja je 'pocieta'),
      3. usuniecie liczb, interpunkcji i znakow specjalnych,
      4. usuniecie pojedynczych znakow (np. 'z', 'w' - wylaczane parametrem),
      5. sciagniecie nadmiarowych bialych znakow.
    """
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _NON_LETTERS.sub(" ", text)
    if remove_single_chars:
        text = _SINGLE_CHAR.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
