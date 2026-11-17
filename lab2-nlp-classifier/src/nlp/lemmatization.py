from __future__ import annotations

from functools import lru_cache

import spacy

from src.errors import BotError


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        # parser i ner nie sa potrzebne do lematyzacji - wylaczenie przyspiesza
        return spacy.load(model_name, exclude=["parser", "ner"])
    except OSError as exc:
        raise BotError(
            f"Brak modelu spaCy '{model_name}'. Uruchom: python scripts/setup_nlp.py"
        ) from exc


def lemmatize(text: str, model_name: str = "pl_core_news_sm") -> list[tuple[str, str]]:
    """Zwraca pary (slowo, lemat), pomijajac interpunkcje i biale znaki."""
    doc = _load_model(model_name)(text)
    return [(t.text, t.lemma_) for t in doc if not (t.is_punct or t.is_space)]
