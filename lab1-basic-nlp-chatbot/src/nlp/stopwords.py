"""Usuwanie stopwords. NLTK nie ma polskiej listy, wiec uzywamy listy spaCy (381 slow)."""
from __future__ import annotations

from dataclasses import dataclass

from spacy.lang.pl.stop_words import STOP_WORDS

from src.nlp.tokenization import only_words, tokenize_spacy


@dataclass(frozen=True)
class StopwordsResult:
    original: str
    kept: list[str]
    removed: list[str]

    @property
    def text_without_stopwords(self) -> str:
        return " ".join(self.kept)


def remove_stopwords(text: str) -> StopwordsResult:
    tokens = only_words(tokenize_spacy(text))
    kept, removed = [], []
    for tok in tokens:
        (removed if tok.lower() in STOP_WORDS else kept).append(tok)
    return StopwordsResult(original=text, kept=kept, removed=removed)
