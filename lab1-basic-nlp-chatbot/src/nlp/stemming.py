"""Stemming dla jezyka polskiego (Stempel - slownikowy stemmer z projektu Egothor)."""
from __future__ import annotations

import importlib.resources as _resources
from functools import lru_cache

# pystempel 2.0 importuje z importlib.resources nazwy 'Package' i 'Resource',
# ktore zostaly usuniete w Pythonie 3.13. W pystempel sluza tylko do adnotacji
# typow, wiec wystarczy je dodac PRZED importem biblioteki.
for _name in ("Package", "Resource"):
    if not hasattr(_resources, _name):
        setattr(_resources, _name, str)

from pystempel import Stemmer  # noqa: E402  (import celowo po shimie)


@lru_cache(maxsize=1)
def _stemmer() -> Stemmer:
    return Stemmer.default()  # ladowanie slownika ~2 MB, robione raz


def stem_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Zwraca pary (slowo, rdzen)."""
    stemmer = _stemmer()
    result = []
    for tok in tokens:
        low = tok.lower()
        result.append((tok, stemmer(low) or low))  # None -> zostaje slowo bez zmian
    return result