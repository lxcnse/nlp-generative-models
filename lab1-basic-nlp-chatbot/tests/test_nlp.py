import pytest

from src.errors import InvalidInputError
from src.nlp.cleaning import clean_text
from src.nlp.ngrams import generate_ngrams
from src.nlp.normalization import normalize_text
from src.nlp.service import NlpService
from src.nlp.stopwords import remove_stopwords
from src.nlp.tokenization import compare_tokenizers, tokenize_nltk, tokenize_spacy
from src.nlp.vectorizers import bag_of_words, tfidf

TEXT = "Przygotować prezentację z przetwarzania języka naturalnego"


def test_clean_text():
    assert clean_text("  Kup 3 MLEKA!!! (do 12.05) http://x.pl a  ") == "kup mleka do"


def test_clean_keeps_polish_letters_and_single_chars_optional():
    assert clean_text("Łódź i ćma") == "łódź ćma"
    assert clean_text("z domu", remove_single_chars=False) == "z domu"


def test_normalize():
    assert normalize_text("  “Łódź”\u00a0—  TEST ") == '"łódź" - test'
    assert normalize_text("Łódź żółć", remove_diacritics=True) == "lodz zolc"


def test_tokenizers_and_comparison():
    assert tokenize_nltk(TEXT) == tokenize_spacy(TEXT)
    cmp = compare_tokenizers("Termin: 12.05!")
    assert not cmp.identical            # NLTK i spaCy tnie '12.05' inaczej
    assert cmp.only_nltk and cmp.only_spacy


def test_ngrams():
    words = ["a", "b", "c"]
    assert generate_ngrams(words, 1) == [("a",), ("b",), ("c",)]
    assert generate_ngrams(words, 2) == [("a", "b"), ("b", "c")]
    assert generate_ngrams(words, 4) == []
    with pytest.raises(ValueError):
        generate_ngrams(words, 0)


def test_stopwords():
    res = remove_stopwords(TEXT)
    assert "z" in res.removed
    assert res.text_without_stopwords == "Przygotować prezentację przetwarzania języka naturalnego"


def test_vectorizers_use_corpus():
    corpus = ["kupić mleko", "kupić chleb"]
    bow = dict(bag_of_words(corpus, "kupić mleko").terms)
    assert bow == {"kupić": 1.0, "mleko": 1.0}
    w = dict(tfidf(corpus, "kupić mleko").terms)
    assert w["mleko"] > w["kupić"]       # slowo rzadsze w korpusie ma wieksza wage


def test_service_all_operations():
    out = NlpService(6).run("all", TEXT, [TEXT, "Kupić mleko"])
    for header in ["TOKENS", "CLEAN", "NORMALIZE", "UNIGRAM", "BIGRAM", "TRIGRAM", "XGRAM",
                   "BOW", "TFIDF", "STEM", "LEMMA", "STOPWORDS"]:
        assert f"=== {header} " in out
    assert "przetwarzania -> przetwarzać" in out
    assert "języka -> język" in out


def test_service_aliases_and_errors():
    s = NlpService(6)
    assert "BIGRAM" in s.run("bigrams", TEXT)
    assert "XGRAM" in s.run("6gram", TEXT)
    with pytest.raises(InvalidInputError):
        s.run("nie_ma_takiej", TEXT)
