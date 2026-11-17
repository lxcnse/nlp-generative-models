# Zadanie 2 – klasyfikacja tekstu w ChatBocie (sprawozdanie)

Pełne wyniki z macierzami, metrykami per klasa i wykresami są w `reports/<zbiór>/results.md`.
Można je odtworzyć poleceniem `python -m scripts.compare <zbiór>`; `random_state=42`
ustala podział danych i trening.

## 1. Wybór metod

Nazwisko na literę **C**. W alfabetycznej liście metod z dokumentacji scikit-learn litera C leży
między **AdaBoost** a **Decision Tree**, więc to te dwie metody są porównywane.

| Metoda | Konfiguracja | Uzasadnienie parametrów |
|---|---|---|
| Decision Tree (`tree`) | `criterion="gini"`, `max_depth=None`, `min_samples_leaf=2` | drzewo rośnie do czystych liści; `min_samples_leaf=2` nie pozwala tworzyć liści dla pojedynczych, rzadkich słów |
| AdaBoost (`adaboost`) | 300 × `DecisionTree(max_depth=2)`, `learning_rate=0.5` | przy 3–4 klasach „pieniek” (głębokość 1) oddziela tylko jedną klasę, a drzewo o głębokości 2 już kilka; mniejszy `learning_rate` ogranicza przeuczenie |

Obie metody przyjmują macierze rzadkie, więc BoW i TF-IDF nie wymagają `toarray()`.
Ograniczenia z polecenia są obsłużone dla pozostałych metod, które bot też udostępnia:
* GP i QDA wymagają gęstej macierzy, więc dla BoW i TF-IDF pipeline dodaje `TruncatedSVD`
  (100 składowych, LSA).
* QDA dostaje dodatkowo `shrinkage=0.5`, bo przy większej liczbie cech niż przykładów
  macierz kowariancji jest osobliwa.
* Naive Bayes dla embeddingów używa `GaussianNB`, bo `MultinomialNB` nie przyjmuje wartości ujemnych.

## 2. Reprezentacje tekstu

Każdy tekst przechodzi najpierw przez `clean_text` z zadania 1 (krok `cleaning` w `Pipeline`):
małe litery, usunięcie URL-i, cyfr, interpunkcji i pojedynczych znaków.

| Repr. | Parametry | Wyjaśnienie |
|---|---|---|
| **BoW** `CountVectorizer` | `ngram_range=(1,2)`, `min_df=2`, `max_df=0.9`, `max_features=20000`, `binary=False` | słowa i pary słów; term musi wystąpić w ≥ 2 dokumentach (odrzuca literówki i unikaty), a termy obecne w > 90% dokumentów działają jak stopwords i też odpadają; wartością cechy jest liczba wystąpień |
| **TF-IDF** `TfidfVectorizer` | jak BoW oraz `sublinear_tf=True`, `smooth_idf=True`, `norm="l2"` | tf zamienione na 1 + log(tf), IDF podnosi wagę słów rzadkich w korpusie, a normalizacja L2 zrównuje teksty krótkie i długie |
| **Embedding** `sentence-transformers` | `paraphrase-multilingual-MiniLM-L12-v2`, `normalize_embeddings=True` | gęsty wektor 384 liczb opisujący znaczenie zdania; model jest wielojęzyczny, więc obsługuje polskie zadania TODO i angielskie newsy; wektory mają długość 1 |

## 3. Dane

* **todo** (część 1, polski): 150 zadań TODO w 3 klasach wymyślonych przez autora:
  **praca / dom / nauka** (po 50 na klasę). Etykieta jest zapisana w polu `label` rekordu zadania.
  Nowe zadania użytkownika etykietują reguły słownikowe, a `/label` pozwala nadpisać etykietę ręcznie.
* **AG News** (część 2, główny zbiór, angielski): `sh0416/ag_news`, 4 klasy
  (World, Sports, Business, SciTech), zbalansowana próbka 1000 newsów na klasę (tytuł i opis).
* **20 Newsgroups** (dodatkowo): grupy `comp.graphics`, `rec.sport.hockey`, `sci.med` i `sci.space`,
  500 postów na klasę, bez nagłówków, stopek i cytatów (inaczej klasę zdradzałaby nazwa grupy w nagłówku).

Podział: 80% trening, 20% test, `train_test_split(stratify=y, random_state=42)`.
Dla AG News daje to 3200 przykładów treningowych i 800 testowych (200 na klasę),
a dla TODO 120 treningowych i 30 testowych (10 na klasę).

## 4. Metryki

Macierz pomyłek `M[i, j]` liczy przykłady z prawdziwą klasą i i przewidzianą klasą j.
Metryki są liczone wprost z niej ([metrics.py](src/classification/metrics.py)).
Klasyfikacja jest wieloklasowa, więc dla każdej klasy k stosujemy podejście one-vs-rest:
TP = M[k,k], FN = suma wiersza k − TP, FP = suma kolumny k − TP, TN = reszta, P = TP + FN, N = TN + FP.
Na tej podstawie liczone są ACC = (TP+TN)/(P+N), ERR = (FP+FN)/(P+N), Precision, Recall i F1.
Średnie podajemy jako **macro** (każda klasa z tą samą wagą) i **weighted** (ważone liczebnością klasy).
ACC całego klasyfikatora to suma przekątnej podzielona przez liczbę przykładów.
Test sprawdza zgodność z przykładem z polecenia (TP=250, FN=50, TN=600, FP=100 → ACC=85%, ERR=15%)
oraz z `sklearn.metrics`.

## 5. Wyniki – AG News (6 konfiguracji)

| repr | metoda | ACC | ERR | Precision (macro) | Recall (macro) | **F1 (macro)** | własne zdania | trening [s] |
|---|---|---|---|---|---|---|---|---|
| embedding | adaboost | **0.795** | **0.205** | 0.799 | 0.795 | **0.796** | **11/12** | 67.8 |
| tfidf | adaboost | 0.681 | 0.319 | 0.702 | 0.681 | 0.686 | 9/12 | 6.0 |
| bow | adaboost | 0.672 | 0.328 | 0.704 | 0.672 | 0.679 | 8/12 | 4.2 |
| bow | tree | 0.626 | 0.374 | 0.628 | 0.626 | 0.627 | 5/12 | 0.7 |
| tfidf | tree | 0.576 | 0.424 | 0.580 | 0.576 | 0.577 | 7/12 | 0.6 |
| embedding | tree | 0.569 | 0.431 | 0.567 | 0.569 | 0.568 | 7/12 | 0.7 |

Macierze pomyłek (wiersze to klasa prawdziwa, kolumny przewidziana; kolejność Business, SciTech, Sports, World):

```
bow + adaboost          tfidf + adaboost        embedding + adaboost
122  56   8  14         126  50  12  12         151  30   4  15
 36 145   5  14          37 141   8  14          34 157   1   8
  2  45 145   8           3  38 150   9           7   2 170  21
 12  45  17 126          13  41  18 128          17  16   9 158

bow + tree              tfidf + tree            embedding + tree
113  49  14  24         108  43  23  26          98  43  30  29
 56 105  20  19          47 113  26  14          49 103  18  30
 20  19 154   7          29  21 128  22          22  18 140  20
 21  27  23 129          34  28  26 112          31  28  27 114
```

Wykresy: `reports/agnews/cm_<repr>_<metoda>.png`.

## 6. Wyniki – TODO (praca / dom / nauka, 6 konfiguracji)

| repr | metoda | ACC | ERR | F1 (macro) | własne zdania |
|---|---|---|---|---|---|
| embedding | adaboost | **0.933** | **0.067** | **0.933** | **10/12** |
| embedding | tree | 0.767 | 0.233 | 0.768 | 8/12 |
| bow | tree | 0.500 | 0.500 | 0.500 | 7/12 |
| tfidf | adaboost | 0.467 | 0.533 | 0.467 | 8/12 |
| tfidf | tree | 0.467 | 0.533 | 0.463 | 8/12 |
| bow | adaboost | 0.433 | 0.567 | 0.438 | 9/12 |

```
(kolejność: dom, nauka, praca)
bow + adaboost     tfidf + adaboost    embedding + adaboost
 3  0  7            5  2  3            10  0  0
 4  4  2            2  4  4             0  8  2
 3  1  6            4  1  5             0  0 10

bow + tree         tfidf + tree        embedding + tree
 4  2  4            4  4  2             8  1  1
 1  5  4            2  4  4             0  8  2
 3  1  6            2  2  6             1  2  7
```

Zbiór testowy TODO ma tylko 30 przykładów, więc jedna pomyłka zmienia ACC o 3,3 p.p.
Różnice między konfiguracjami BoW i TF-IDF (0.43–0.50) mieszczą się w tym szumie.
Różnica względem embeddingu (0.77–0.93) jest za to wyraźna.

## 7. Dodatkowo – 20 Newsgroups

| repr | metoda | ACC | F1 (macro) |
|---|---|---|---|
| embedding | adaboost | **0.905** | **0.905** |
| bow | adaboost | 0.750 | 0.759 |
| tfidf | adaboost | 0.740 | 0.748 |
| embedding | tree | 0.723 | 0.722 |
| bow | tree | 0.680 | 0.683 |
| tfidf | tree | 0.677 | 0.677 |

Kolejność konfiguracji jest taka sama jak na AG News, co potwierdza wnioski z sekcji 9.

## 8. Walidacja na własnych zdaniach (spoza zbioru)

Zdania napisane przez autora pasują tematycznie do zbiorów, ale nie występują w nich
(lista w `CUSTOM_SENTENCES`, [datasets.py](src/classification/datasets.py); w bocie: `/validate repr metoda`).

**TODO.** Najlepiej wypadł *embedding + adaboost*: 10/12 poprawnych.
* Zdania ze słowami spoza słownika treningowego rozwiązuje tylko embedding. „Zabrać kota do groomera”
  (dom) wszystkie modele BoW i TF-IDF zaklasyfikowały jako *praca*, a oba modele embeddingowe trafiły.
  Model rozumie, że „kot” należy do tej samej domeny co „pies” i „weterynarz”.
  „Przygotować slajdy na obronę inżynierki” (nauka) trafił tylko *embedding + tree*.
  Słowa „slajdy” i „obrona” nie występują w zbiorze treningowym, a ich znaczenie jest bliskie
  słowu „prezentacja”, które w zbiorze pojawia się głównie w klasie *praca*.
* „Zrobić przegląd kodu kolegi z zespołu” (praca) wszystkie 6 modeli zaklasyfikowało jako *nauka*.
  Zbiór treningowy zawiera „code review” po angielsku, a polski „przegląd kodu” przypomina
  zadania programistyczne z klasy *nauka* („napisać program na laboratorium”). To typowy błąd
  związany z pokryciem danych: brakuje przykładów, a nie lepszego modelu.
* „Napisać raport z laboratorium” (przykład z polecenia) wszystkie modele zaklasyfikowały poprawnie
  jako *nauka*, choć słowo „raport” jest typowe dla klasy *praca*.

**AG News.** *Embedding + adaboost* trafił 11/12 zdań, modele BoW i TF-IDF 5–9/12.
* Zdania bez charakterystycznych słów, np. „Floods displace thousands across southern Asia”
  czy „Peace talks resume between the two countries…”, modele BoW + AdaBoost i TF-IDF + AdaBoost
  wysyłały do *SciTech*. Ta sama tendencja jest widoczna w macierzach (45 newsów sportowych
  i 45 ze świata zaklasyfikowanych jako SciTech): AdaBoost na rzadkich cechach uczy się reguł typu
  „jest słowo X → klasa”, a tekst, który nie zawiera żadnego z takich słów, ląduje w klasie domyślnej.
  Embedding nie ma tego problemu, bo „floods” i „displace” są semantycznie blisko innych newsów ze świata.
* Jedyny błąd embeddingu: „Tech giant buys gaming startup…” (Business) zaklasyfikowany jako SciTech.
  To zdanie naprawdę leży na granicy klas (przejęcie firmy technologicznej), więc błąd jest zrozumiały.

**Prawdopodobieństwa.** AdaBoost zwraca bardzo „płaskie” prawdopodobieństwa
(np. 0.26 / 0.25 / 0.25 / 0.24). `predict_proba` w AdaBoost (SAMME) to softmax z uśrednionej decyzji
300 słabych drzew podzielonej przez (K−1), więc wynik jest ściśnięty wokół 1/K. Klasa z maksimum
jest poprawna, ale samej wartości p nie należy czytać jako pewności. Pojedyncze drzewo przeciwnie,
zwraca prawie zawsze p = 1.00 (czysty liść) i jest przesadnie pewne siebie, także przy błędach.

## 9. Wnioski – która konfiguracja jest najlepsza i dlaczego

1. **Najlepsza konfiguracja to embedding + AdaBoost** na wszystkich trzech zbiorach:
   AG News F1 = 0.80, 20 Newsgroups F1 = 0.91, TODO F1 = 0.93. Ta konfiguracja najlepiej
   uogólnia też na własne zdania (11/12 i 10/12). Wektor zdania z modelu wytrenowanego na milionach
   zdań niesie wiedzę o synonimach i odmianie, której nie ma w 120 czy 3200 przykładach treningowych.
   Ma to szczególne znaczenie dla polskiego: przy silnej fleksji („raport”, „raportu”, „raporty”)
   słownik BoW dla zbioru TODO po `min_df=2` ma tylko **73 termy**. Przy `min_df=1` ma ich 892,
   ale wyniki są jeszcze gorsze (ACC 0.30–0.40), bo prawie każdy term występuje raz.
   Koszt: najdłuższy trening (68 s na AG News, głównie liczenie embeddingów i 300 drzew na 384 gęstych
   cechach) oraz potrzeba modelu językowego (~470 MB).
2. **AdaBoost jest lepszy od pojedynczego drzewa** dla każdej reprezentacji na AG News i 20 Newsgroups,
   o 5–23 p.p. ACC. Na TODO wygrywa wyraźnie tylko z embeddingiem. Dla BoW i TF-IDF przy 30 przykładach
   testowych różnice (0.43 vs 0.50 i remis 0.47) są w granicach szumu. Ensemble 300 drzew uśrednia błędy, a pojedyncze drzewo
   z `max_depth=None` przeucza się, czego objawem są liście z p = 1.00.
3. **Pojedyncze drzewo nie wykorzystuje embeddingów** (AG News: 0.569, najgorszy wynik ze wszystkich).
   Każdy podział sprawdza próg na jednej z 384 współrzędnych, a pojedyncza współrzędna embeddingu
   nic nie znaczy; informację niesie dopiero kierunek całego wektora. AdaBoost składa wiele takich
   progów i odzyskuje tę informację.
4. **BoW i TF-IDF dają podobne wyniki** (różnice ≤ 1 p.p. dla AdaBoost na AG News i 20 Newsgroups). Drzewa nie zależą od skali
   cech, a decyzja „waga > próg” wybiera prawie te same słowa niezależnie od tego, czy wagą jest
   liczba wystąpień, czy tf-idf. Przewaga TF-IDF ujawniłaby się przy metodach opartych na odległości
   lub iloczynie skalarnym (kNN, SVM, regresja logistyczna).
5. Najczęściej mylone klasy na AG News to **Business ↔ SciTech** (30–56 pomyłek w każdą stronę),
   bo newsy o firmach technologicznych należą tematycznie do obu klas. Najłatwiejszy jest **Sports**
   (recall do 0.85) dzięki wyrazistemu słownictwu: nazwy drużyn, „scored”, „cup”.

**Rekomendacja:** w bocie domyślnie `embedding adaboost`. Gdy liczy się czas (np. trening na słabym
sprzęcie), sensownym kompromisem jest `tfidf adaboost`: ok. 11 razy szybszy trening na AG News
kosztem ok. 11 p.p. F1.
