"""Tokenizacja dwiema metodami (NLTK i spaCy) + porownanie wynikow."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import spacy
from nltk.tokenize import word_tokenize

from src.nlp.resources import ensure_nltk_data


@lru_cache(maxsize=1)
def _spacy_blank():
    """Pusty pipeline z samym tokenizerem - szybki, nie wymaga pobierania modelu."""
    return spacy.blank("pl")


def tokenize_nltk(text: str) -> list[str]:
    ensure_nltk_data()
    return word_tokenize(text, language="polish")


def tokenize_spacy(text: str) -> list[str]:
    return [t.text for t in _spacy_blank()(text) if not t.is_space]


@dataclass(frozen=True)
class TokenizerComparison:
    nltk: list[str]
    spacy: list[str]
    only_nltk: list[str]    # tokeny, ktore wystepuja tylko u NLTK
    only_spacy: list[str]   # tokeny, ktore wystepuja tylko u spaCy

    @property
    def identical(self) -> bool:
        return self.nltk == self.spacy


def compare_tokenizers(text: str) -> TokenizerComparison:
    a, b = tokenize_nltk(text), tokenize_spacy(text)
    # Counter (multizbior) uwzglednia powtorzenia, zwykly set by je gubil.
    return TokenizerComparison(
        nltk=a,
        spacy=b,
        only_nltk=list((Counter(a) - Counter(b)).elements()),
        only_spacy=list((Counter(b) - Counter(a)).elements()),
    )


def only_words(tokens: list[str]) -> list[str]:
    """Odrzuca tokeny bez liter/cyfr (sama interpunkcja)."""
    return [t for t in tokens if any(ch.isalnum() for ch in t)]
