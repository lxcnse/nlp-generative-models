"""Pobieranie zasobow NLTK (model tokenizera 'punkt') w razie potrzeby."""
from __future__ import annotations

from functools import lru_cache

import nltk


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
