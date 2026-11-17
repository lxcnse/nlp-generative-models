"""Reprezentacje tekstu dla klasyfikatorow: Bag of Words, TF-IDF, embedding.

Kazda reprezentacja to transformer scikit-learn (fit/transform), dzieki czemu
mozna ja wstawic do Pipeline razem z czyszczeniem tekstu i klasyfikatorem.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.errors import BotError, InvalidInputError
from src.nlp.cleaning import clean_text

REPRESENTATIONS: tuple[str, ...] = ("bow", "tfidf", "embedding")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --------------------------------------------------------------------- parametry
BOW_PARAMS: dict = dict(
    ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=20000, lowercase=True, binary=False,
)
TFIDF_PARAMS: dict = dict(
    ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=20000, lowercase=True,
    use_idf=True, smooth_idf=True, sublinear_tf=True, norm="l2",
)
EMBEDDING_PARAMS: dict = dict(
    model_name=DEFAULT_EMBEDDING_MODEL, normalize_embeddings=True, batch_size=64,
)

PARAM_DOCS: dict[str, str] = {
    "ngram_range": "(1, 2) = pojedyncze slowa + pary slow (np. 'raport kwartalny'); "
                   "bigramy lapia krotkie frazy, ktore niosa wiecej znaczenia niz slowa osobno",
    "min_df": "term musi wystapic w >= 2 dokumentach; odrzuca literowki i unikaty, "
              "ktore i tak nie pomoga uogolnic",
    "max_df": "odrzuca termy obecne w > 90% dokumentow (dzialaja jak stopwords)",
    "max_features": "limit slownika do 20 000 najczestszych termow (pamiec i szybkosc)",
    "lowercase": "male litery (tekst i tak jest juz po clean_text)",
    "binary": "False = liczba wystapien termu, a nie tylko 0/1",
    "use_idf": "waga IDF: rzadkie w korpusie slowa sa wazniejsze",
    "smooth_idf": "+1 w liczniku i mianowniku IDF, brak dzielenia przez zero",
    "sublinear_tf": "tf -> 1 + log(tf); slowo powtorzone 10x nie jest 10x wazniejsze",
    "norm": "normalizacja L2 wektora dokumentu - dlugie teksty nie dominuja",
    "model_name": "wielojezyczny model sentence-transformers (PL + EN), wektor 384 liczb; "
                  "zdania o podobnym znaczeniu maja bliskie wektory nawet bez wspolnych slow",
    "normalize_embeddings": "wektory o dlugosci 1 (iloczyn skalarny = podobienstwo kosinusowe)",
    "batch_size": "ile zdan liczy sie naraz (kompromis pamiec/szybkosc)",
}

REPR_DESCRIPTIONS: dict[str, str] = {
    "bow": "Bag of Words (CountVectorizer) - rzadki wektor liczby wystapien termow",
    "tfidf": "TF-IDF (TfidfVectorizer) - rzadki wektor wag tf * idf",
    "embedding": "Embedding (sentence-transformers) - gesty wektor znaczenia zdania",
}


def repr_params(kind: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    if kind == "bow":
        return dict(BOW_PARAMS)
    if kind == "tfidf":
        return dict(TFIDF_PARAMS)
    if kind == "embedding":
        return {**EMBEDDING_PARAMS, "model_name": embedding_model}
    raise InvalidInputError(
        f"Nieznana reprezentacja '{kind}'. Dostepne: {', '.join(REPRESENTATIONS)}"
    )


# ------------------------------------------------------------------ transformery
class TextCleaner(BaseEstimator, TransformerMixin):
    """Krok 'cleaning' w Pipeline - funkcja clean_text z zadania 1."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: Sequence[str]) -> list[str]:
        return [clean_text(x) for x in X]


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - zalezy od instalacji
        raise BotError(
            "Reprezentacja 'embedding' wymaga pakietu sentence-transformers "
            "(pip install sentence-transformers)."
        ) from exc
    return SentenceTransformer(model_name, device="cpu")


# Pamiec podreczna wektorow: ten sam tekst liczymy tylko raz (porownanie 2 metod
# na tej samej reprezentacji nie musi dwa razy przepuszczac korpusu przez siec).
_EMBED_CACHE: dict[tuple[str, str], np.ndarray] = {}


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Zamienia liste tekstow na macierz embeddingow (n_dokumentow x 384).

    Model nie jest atrybutem obiektu (tylko globalnym cache), wiec Pipeline
    mozna bez problemu zapisac joblibem - plik ma kilka kB, a nie 400 MB.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 normalize_embeddings: bool = True, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        texts = [x if x.strip() else "-" for x in X]  # pusty tekst -> neutralny token
        missing = list({t for t in texts if (self.model_name, t) not in _EMBED_CACHE})
        if missing:
            model = _load_sentence_model(self.model_name)
            vectors = model.encode(
                missing, batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings, show_progress_bar=False,
            )
            for text, vec in zip(missing, vectors):
                _EMBED_CACHE[(self.model_name, text)] = np.asarray(vec, dtype=np.float32)
        return np.vstack([_EMBED_CACHE[(self.model_name, t)] for t in texts])


def build_representation(kind: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    params = repr_params(kind, embedding_model)
    if kind == "bow":
        return CountVectorizer(**params)
    if kind == "tfidf":
        return TfidfVectorizer(**params)
    return SentenceEmbedder(**params)


def is_sparse_repr(kind: str) -> bool:
    """BoW i TF-IDF daja macierze rzadkie (scipy.sparse), embedding - gesta."""
    return kind in ("bow", "tfidf")


def describe_params(kind: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> str:
    lines = [REPR_DESCRIPTIONS[kind]]
    for key, value in repr_params(kind, embedding_model).items():
        lines.append(f"  {key} = {value}  # {PARAM_DOCS[key]}")
    return "\n".join(lines)
