"""Reprezentacje wektorowe tekstu: Bag of Words (CountVectorizer) i TF-IDF (TfidfVectorizer).

Wektoryzatory sa dopasowywane (fit) do KORPUSU = opisow wszystkich zadan uzytkownika,
a nastepnie transformowane jest zadanie wskazane w komendzie. Dzieki temu IDF ma sens
(slowo wystepujace w wielu zadaniach dostaje nizsza wage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.errors import BotError
from src.nlp.cleaning import clean_text

# Parametry + krotkie wyjasnienia (wyswietlane uzytkownikowi przez bota).
BOW_PARAMS: dict = dict(
    ngram_range=(1, 1), max_features=None, min_df=1, max_df=1.0, lowercase=True, binary=False
)
TFIDF_PARAMS: dict = dict(
    ngram_range=(1, 1), max_features=None, min_df=1, max_df=1.0,
    use_idf=True, smooth_idf=True, sublinear_tf=False,
)
PARAM_DOCS: dict[str, str] = {
    "ngram_range": "zakres dlugosci n-gramow; (1, 1) = same pojedyncze slowa, (1, 2) = slowa i pary",
    "max_features": "limit rozmiaru slownika (None = bez limitu; inaczej tylko N najczestszych)",
    "min_df": "ignoruje termy rzadsze niz w tylu dokumentach (1 = bierz wszystkie)",
    "max_df": "ignoruje termy czestsze niz ten ulamek dokumentow (1.0 = nie odrzucaj nic)",
    "lowercase": "zamiana na male litery przed tokenizacja",
    "binary": "True = tylko obecnosc slowa (0/1), False = liczba wystapien",
    "use_idf": "wlacza wage IDF (odwrotna czestosc dokumentowa)",
    "smooth_idf": "dodaje 1 do licznikow w IDF, by uniknac dzielenia przez zero",
    "sublinear_tf": "zamienia tf na 1 + log(tf), tlumiac wplyw bardzo czestych slow",
}


@dataclass(frozen=True)
class VectorResult:
    kind: str                       # "bow" | "tfidf"
    params: dict
    vocabulary_size: int
    terms: list[tuple[str, float]]  # niezerowe wspolrzedne wektora zadania


def _build(kind: str, corpus: Sequence[str], target: str) -> VectorResult:
    docs = [clean_text(d) for d in corpus]
    target_clean = clean_text(target)
    if target_clean not in docs:        # gwarantuje, ze zadanie jest w slowniku
        docs.append(target_clean)

    if kind == "bow":
        params = BOW_PARAMS
        vec = CountVectorizer(**params)
    else:
        params = TFIDF_PARAMS
        vec = TfidfVectorizer(**params)

    try:
        vec.fit(docs)
    except ValueError as exc:  # np. "empty vocabulary" gdy zadanie ma same znaki specjalne
        raise BotError(f"Nie da sie zbudowac reprezentacji: {exc}") from exc

    row = vec.transform([target_clean])
    features = vec.get_feature_names_out()
    coo = row.tocoo()
    terms = sorted(
        ((str(features[j]), float(v)) for j, v in zip(coo.col, coo.data)),
        key=lambda t: (-t[1], t[0]),
    )
    return VectorResult(kind, dict(params), len(features), terms)


def bag_of_words(corpus: Sequence[str], target: str) -> VectorResult:
    return _build("bow", corpus, target)


def tfidf(corpus: Sequence[str], target: str) -> VectorResult:
    return _build("tfidf", corpus, target)
