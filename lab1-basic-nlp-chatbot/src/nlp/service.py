"""Fasada NLP: jedno miejsce, ktore laczy wszystkie operacje i formatuje wynik jako tekst.

Handler Telegrama zna tylko ta klase - nie importuje NLTK/spaCy/sklearn bezposrednio.
"""
from __future__ import annotations

from typing import Callable, Sequence

from src.errors import InvalidInputError
from src.nlp import vectorizers
from src.nlp.cleaning import clean_text
from src.nlp.lemmatization import lemmatize
from src.nlp.ngrams import generate_ngrams
from src.nlp.normalization import normalize_text
from src.nlp.stemming import stem_tokens
from src.nlp.stopwords import remove_stopwords
from src.nlp.tokenization import compare_tokenizers, only_words, tokenize_nltk

ALIASES: dict[str, str] = {
    "token": "tokens", "tokenize": "tokens", "tokenization": "tokens",
    "clean_text": "clean", "normalization": "normalize", "norm": "normalize",
    "unigram": "unigram", "unigrams": "unigram", "1gram": "unigram",
    "bigrams": "bigram", "2gram": "bigram",
    "trigrams": "trigram", "3gram": "trigram",
    "bag_of_words": "bow", "count": "bow",
    "tf-idf": "tfidf", "tf_idf": "tfidf",
    "stemming": "stem", "stems": "stem",
    "lemmatization": "lemma", "lemmas": "lemma", "lemmatize": "lemma",
    "stopword": "stopwords", "stop": "stopwords",
}


class NlpService:
    def __init__(self, xgram_n: int, spacy_model: str = "pl_core_news_sm") -> None:
        self.xgram_n = xgram_n
        self.spacy_model = spacy_model
        # kolejnosc = kolejnosc w /ops oraz w operacji 'all'
        self._ops: dict[str, tuple[str, Callable[[str, Sequence[str]], str]]] = {
            "tokens": ("tokenizacja NLTK vs spaCy + porownanie", self._tokens),
            "clean": ("clean_text - czyszczenie tekstu", self._clean),
            "normalize": ("normalizacja (casefold, znaki, spacje, ogonki)", self._normalize),
            "unigram": ("1-gramy", lambda t, c: self._ngram(t, 1)),
            "bigram": ("2-gramy", lambda t, c: self._ngram(t, 2)),
            "trigram": ("3-gramy", lambda t, c: self._ngram(t, 3)),
            "xgram": (f"{xgram_n}-gramy (wg numeru albumu)", lambda t, c: self._ngram(t, xgram_n)),
            "bow": ("Bag of Words (CountVectorizer)", self._bow),
            "tfidf": ("TF-IDF (TfidfVectorizer)", self._tfidf),
            "stem": ("stemming", self._stem),
            "lemma": ("lematyzacja (spaCy)", self._lemma),
            "stopwords": ("tekst przed i po usunieciu stopwords", self._stopwords),
        }
        self._aliases = dict(ALIASES)
        self._aliases[f"{xgram_n}gram"] = "xgram"
        self._aliases["xgrams"] = "xgram"

    # ------------------------------------------------------------------- API
    def available_operations(self) -> dict[str, str]:
        ops = {name: desc for name, (desc, _) in self._ops.items()}
        ops["all"] = "wszystkie powyzsze naraz"
        return ops

    def run(self, operation: str, text: str, corpus: Sequence[str] = ()) -> str:
        name = self._aliases.get(operation.lower(), operation.lower())
        if name == "all":
            return "\n\n".join(self._section(n, text, corpus) for n in self._ops)
        if name not in self._ops:
            raise InvalidInputError(
                f"Nieznana operacja '{operation}'. Dostepne: {', '.join(self.available_operations())}"
            )
        return self._section(name, text, corpus)

    def _section(self, name: str, text: str, corpus: Sequence[str]) -> str:
        desc, func = self._ops[name]
        return f"=== {name.upper()} - {desc} ===\n{func(text, corpus)}"

    # ------------------------------------------------------------ pomocnicze
    @staticmethod
    def _words(text: str) -> list[str]:
        """Znormalizowane slowa bez interpunkcji - baza dla n-gramow i stemmingu."""
        return only_words(tokenize_nltk(normalize_text(text)))

    @staticmethod
    def _pairs(pairs: list[tuple[str, str]]) -> str:
        return "\n".join(f"{a} -> {b}" for a, b in pairs) or "(brak)"

    # ----------------------------------------------------------- operacje NLP
    def _tokens(self, text: str, corpus: Sequence[str]) -> str:
        cmp = compare_tokenizers(text)
        lines = [f"NLTK  ({len(cmp.nltk)}): {cmp.nltk}", f"spaCy ({len(cmp.spacy)}): {cmp.spacy}"]
        if cmp.identical:
            lines.append("Porownanie: obie metody daly identyczny wynik.")
        else:
            lines.append("Porownanie: wyniki sie roznia.")
            lines.append(f"  tylko NLTK : {cmp.only_nltk}")
            lines.append(f"  tylko spaCy: {cmp.only_spacy}")
        return "\n".join(lines)

    def _clean(self, text: str, corpus: Sequence[str]) -> str:
        return f"Przed: {text}\nPo   : {clean_text(text)}"

    def _normalize(self, text: str, corpus: Sequence[str]) -> str:
        return (
            f"Przed           : {text}\n"
            f"Znormalizowany  : {normalize_text(text)}\n"
            f"Bez ogonkow     : {normalize_text(text, remove_diacritics=True)}"
        )

    def _ngram(self, text: str, n: int) -> str:
        grams = generate_ngrams(self._words(text), n)
        if not grams:
            return f"(tekst ma za malo slow na {n}-gramy)"
        return f"Liczba: {len(grams)}\n{grams}"

    def _format_vector(self, res: vectorizers.VectorResult) -> str:
        params = "\n".join(
            f"  {k} = {v}  # {vectorizers.PARAM_DOCS[k]}" for k, v in res.params.items()
        )
        terms = "\n".join(f"  {term}: {value:.4f}" for term, value in res.terms) or "  (brak)"
        return (
            f"Parametry:\n{params}\n"
            f"Rozmiar slownika (korpus = Twoje zadania): {res.vocabulary_size}\n"
            f"Niezerowe wspolrzedne wektora zadania:\n{terms}"
        )

    def _bow(self, text: str, corpus: Sequence[str]) -> str:
        return self._format_vector(vectorizers.bag_of_words(corpus, text))

    def _tfidf(self, text: str, corpus: Sequence[str]) -> str:
        return self._format_vector(vectorizers.tfidf(corpus, text))

    def _stem(self, text: str, corpus: Sequence[str]) -> str:
        return self._pairs(stem_tokens(self._words(text)))

    def _lemma(self, text: str, corpus: Sequence[str]) -> str:
        return self._pairs(lemmatize(text, self.spacy_model))

    def _stopwords(self, text: str, corpus: Sequence[str]) -> str:
        res = remove_stopwords(text)
        return (
            f"Oryginal        : {res.original}\n"
            f"Bez stopwords   : {res.text_without_stopwords}\n"
            f"Usuniete slowa  : {res.removed}"
        )
