from __future__ import annotations

from nltk.util import ngrams


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n musi byc >= 1")
    return list(ngrams(tokens, n))  # pusta lista, gdy tokenow jest mniej niz n
