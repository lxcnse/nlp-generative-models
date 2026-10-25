# Telegram TODO Bot + NLP

Bot do zarządzania listą zadań (TODO) z operacjami NLP na treści zadań
(tokenizacja, czyszczenie, normalizacja, n-gramy, BoW, TF-IDF, stemming, lematyzacja, stopwords).

## Uruchomienie krok po kroku

```bash
# 1. Środowisko wirtualne (izoluje biblioteki projektu od systemu)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Zależności
pip install -r requirements.txt

# 3. Zasoby NLP (NLTK punkt_tab + model spaCy pl_core_news_sm) - jednorazowo
python -m scripts.setup_nlp

# 4. Konfiguracja
cp .env.example .env               # Windows: copy .env.example .env
# w .env wpisz TELEGRAM_BOT_TOKEN (od @BotFather) oraz ALBUM_NUMBER

# 5. Start
python -m src
```

Token: w Telegramie otwórz **@BotFather** → `/newbot` → podaj nazwę i login (kończy się na `bot`) → skopiuj token.

## Komendy

| Komenda | Opis |
|---|---|
| `/tasks` | wszystkie zadania |
| `/add opis [--parent id] [--prio low\|medium\|high]` | dodaj zadanie (opcjonalnie relacja do nadrzędnego) |
| `/update id nowy opis` | zmień opis |
| `/delete id` | usuń zadanie |
| `/status id todo\|in_progress\|done` | zmień status |
| `/task id operacja` | operacja NLP na opisie |
| `/ops` | lista operacji |

`id` może być pełnym UUID albo jego początkiem (min. 4 znaki).

Operacje NLP: `tokens`, `clean`, `normalize`, `unigram`, `bigram`, `trigram`, `xgram`
(lub np. `6gram`), `bow`, `tfidf`, `stem`, `lemma`, `stopwords`, `all`.

## Testy i demo bez Telegrama

```bash
pip install -r requirements-dev.txt
pytest
python -m scripts.demo_nlp "Przygotować prezentację z NLP" tfidf
```

## Struktura

```
src/
  config.py            konfiguracja (.env) + wyliczanie x-gramu z numeru albumu
  errors.py            własne wyjątki
  main.py              składanie aplikacji i start
  storage/             model Pydantic + repozytorium SQLite
  nlp/                 po jednym module na operację + service.py (fasada)
  bot/                 handlery Telegrama, parsowanie argumentów, teksty
scripts/               setup_nlp.py, demo_nlp.py
tests/                 pytest
```

## Decyzje projektowe

* **Polski język** – NLTK nie ma polskiego stemmera ani listy stopwords, więc: stemming = `pystempel`,
  stopwords = lista spaCy, lematyzacja = `pl_core_news_sm`, tokenizacja = NLTK (`punkt`, `polish`) i spaCy.
* **BoW/TF-IDF** liczone na korpusie wszystkich zadań użytkownika (inaczej IDF byłby bezsensowny).
* **Zadania są per użytkownik** (`owner_id`) – nikt nie widzi cudzych zadań.
