# Telegram TODO Bot + NLP + klasyfikacja tekstu (zadanie 2)

Rozszerzenie bota z [zadania 1](../lab1-basic-nlp-chatbot) o moduł klasyfikacji tekstu:
etykietowanie danych, trening klasyfikatorów na trzech reprezentacjach (BoW, TF-IDF, embedding)
i ocena jakości macierzą pomyłek oraz metrykami z niej wyliczonymi.
Wszystkie komendy z zadania 1 działają bez zmian.

* **Metody klasyfikacji:** AdaBoost i Decision Tree. Nazwisko na literę **C** wypada w alfabecie
  między *AdaBoost* a *Decision Tree*. Litera jest ustawiana w `.env` (`SURNAME_INITIAL`), a para metod
  wylicza się sama (np. `M` daje Logistic Regression + MLP). Pozostałe 9 metod z listy też działa.
* **Reprezentacje:** `bow` (CountVectorizer), `tfidf` (TfidfVectorizer), `embedding`
  (sentence-transformers, wielojęzyczny `paraphrase-multilingual-MiniLM-L12-v2`).
* **Zbiory:** `todo` (zadania z zadania 1, klasy praca / dom / nauka), `agnews` (AG News, 4 klasy),
  `20newsgroups` (4 grupy).
* **Sprawozdanie:** [SPRAWOZDANIE.md](SPRAWOZDANIE.md), a pełne wyniki w [reports/](reports).

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt    # sentence-transformers pobiera PyTorch (~kilkaset MB)
python -m scripts.setup_nlp        # NLTK punkt_tab + spaCy pl_core_news_sm
copy .env.example .env             # wpisz TELEGRAM_BOT_TOKEN, ALBUM_NUMBER, SURNAME_INITIAL
python -m src
```

Jeśli `spacy download` zgłasza błąd certyfikatu SSL, zainstaluj model bezpośrednio przez pip:
`pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_sm-3.8.0/pl_core_news_sm-3.8.0-py3-none-any.whl`

## Nowe komendy

| Komenda | Opis |
|---|---|
| `/prepare_data_for_classification [todo\|agnews\|20newsgroups] [--seed]` | nadaje etykiety i ustawia aktywny zbiór (domyślnie `todo`) |
| `/label {id} {klasa}` | ręczne nadanie lub nadpisanie etykiety zadania (`none` usuwa etykietę) |
| `/classifier cleaning "tekst" {repr} {metoda}` | klasyfikacja tekstu |
| `/evaluate {repr} {metoda}` | macierz pomyłek, metryki per klasa (one-vs-rest) oraz macro/weighted |
| `/validate {repr} {metoda}` | test na własnych zdaniach spoza zbioru |
| `/methods` | wybrane metody, ich parametry i parametry reprezentacji |

`{repr}`: `bow`, `tfidf` lub `embedding`. `{metoda}`: `adaboost` lub `tree` (wybrane), a także
`gp`, `linear_svm`, `logreg`, `mlp`, `nb`, `knn`, `qda`, `rf` i `rbf_svm`.

### Jak działa etykietowanie (`todo`)

1. `--seed` dodaje do listy zadań użytkownika 150 przykładowych zadań z [data/todo_seed.csv](data/todo_seed.csv)
   (po 50 na klasę; klasy wymyślone przez autora: **praca / dom / nauka**).
2. Każde zadanie dostaje etykietę w polu `label` rekordu w SQLite. Zadania z przykładowego zbioru
   dostają etykietę z pliku, a pozostałe etykietę z reguł słownikowych (`KEYWORDS` w
   [datasets.py](src/classification/datasets.py)). Ponowne wywołanie nadpisuje etykiety.
3. Zadanie, którego reguły nie rozpoznały, zachowuje dotychczasową etykietę; można ją nadać przez `/label`.

Dla `agnews` i `20newsgroups` zbiór jest pobierany (zbalansowany, `SAMPLES_PER_CLASS` na klasę)
i zapisywany w tabeli `samples`.

### Przykład

```
/prepare_data_for_classification todo --seed
Labels assigned to 150 tasks (dataset: todo). Classes: dom, nauka, praca

/classifier cleaning "Napisać raport z laboratorium" embedding adaboost
Predicted class: nauka (p=0.39) | repr: embedding | method: adaboost | dataset: todo
cleaning: "napisać raport laboratorium"
Wszystkie klasy: nauka=0.39, praca=0.34, dom=0.26

/evaluate embedding adaboost
            pred_dom  pred_nauka  pred_praca
true_dom          10           0           0
true_nauka         0           8           2
true_praca         0           0          10

ACC = 0.933 | ERR = 0.067 | F1 (macro) = 0.933 | n = 30
...
```

Model danej konfiguracji trenuje się przy pierwszym użyciu i trafia do pamięci oraz `models/*.joblib`.
Klucz modelu zawiera hash danych, więc zmiana etykiet albo nowe zadania wymuszają ponowny trening.

## Porównanie 6 konfiguracji (do sprawozdania)

```bash
python -m scripts.compare todo
python -m scripts.compare agnews
python -m scripts.compare 20newsgroups --per-class 500
```

Każde wywołanie zapisuje do `reports/<zbiór>/` plik `results.md` (macierze, tabela, walidacja
na własnych zdaniach), `results.csv` oraz wykresy `cm_*.png`.

## Testy

```bash
pip install -r requirements-dev.txt
pytest
```

## Struktura (nowe elementy)

```
src/classification/
  representations.py  BoW / TF-IDF / embedding + opis parametrów, krok cleaning (clean_text)
  methods.py          rejestr 11 metod sklearn, wybór 2 metod wg litery nazwiska, SVD dla GP/QDA
  pipeline.py         Pipeline cleaning -> repr -> (svd) -> clf, podział 80/20, trening, predykcja
  metrics.py          macierz pomyłek i ACC/ERR/Precision/Recall/F1 liczone z niej (OvR, macro/weighted)
  datasets.py         zbiory, reguły etykietowania TODO, własne zdania walidacyjne
  service.py          fasada dla bota (cache modeli, aktywny zbiór użytkownika)
scripts/compare.py    porównanie konfiguracji -> reports/
data/todo_seed.csv    150 zadań TODO z etykietami
```
