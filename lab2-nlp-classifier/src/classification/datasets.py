"""Zbiory danych z etykietami.

* todo          - zadania TODO uzytkownika z zadania 1, klasy: praca / dom / nauka
                  (etykiety nadaje /prepare_data_for_classification, mozna je nadpisac /label)
* agnews        - AG News (sh0416/ag_news), 4 klasy: World / Sports / Business / SciTech
* 20newsgroups  - 20 Newsgroups (sklearn), wybrane 4 grupy tematyczne
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.errors import BotError, InvalidInputError
from src.nlp.cleaning import clean_text

SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "todo_seed.csv"
TODO_CLASSES: tuple[str, ...] = ("praca", "dom", "nauka")

AGNEWS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "SciTech"}
NEWSGROUPS_CATEGORIES = {
    "comp.graphics": "graphics",
    "rec.sport.hockey": "hockey",
    "sci.med": "medicine",
    "sci.space": "space",
}

DATASET_ALIASES = {
    "todo": "todo", "tasks": "todo", "zadania": "todo",
    "agnews": "agnews", "ag_news": "agnews", "ag-news": "agnews",
    "20newsgroups": "20newsgroups", "20newsgroup": "20newsgroups", "newsgroups": "20newsgroups",
    "20ng": "20newsgroups",
}
DATASETS: tuple[str, ...] = ("todo", "agnews", "20newsgroups")


def resolve_dataset(name: Optional[str]) -> str:
    key = (name or "todo").strip().lower()
    if key not in DATASET_ALIASES:
        raise InvalidInputError(
            f"Nieznany zbior '{name}'. Dostepne: {', '.join(DATASETS)}"
        )
    return DATASET_ALIASES[key]


# ------------------------------------------------------------------ zadania TODO
def load_seed() -> list[tuple[str, str]]:
    """150 przykladowych zadan TODO z etykietami (po 50 na klase) - data/todo_seed.csv."""
    with SEED_PATH.open(encoding="utf-8", newline="") as f:
        return [(row["text"], row["label"]) for row in csv.DictReader(f)]


# Slownik regul do automatycznego etykietowania zadan uzytkownika.
# Klucze to poczatki slow (po clean_text), wiec "raport" lapie tez "raportu", "raporty".
KEYWORDS: dict[str, tuple[str, ...]] = {
    "praca": (
        "raport", "klient", "spotkani", "faktur", "szef", "kierownik", "zesp", "projekt",
        "deadline", "mail", "umow", "firm", "biur", "sprint", "wdroż", "wdroz", "ofert",
        "budżet", "budzet", "rekrut", "delegac", "kontrakt", "prezentacj", "serwer",
        "code", "review", "jira", "backlog", "release", "pracowni", "zarząd", "zarzad",
        "kontrahent", "dostawc", "sprzedaż", "sprzedaz", "timesheet", "urlop", "crm", "kadr",
    ),
    "dom": (
        "kupi", "zakup", "mleko", "chleb", "sprząt", "sprzat", "odkurz", "pranie", "prasow",
        "wypra", "obiad", "kolacj", "gotow", "ugotow", "naczy", "zmywar", "czynsz", "rachun",
        "mieszkani", "ogród", "ogrod", "kwiat", "pies", "psa", "kot", "śmieci", "smieci",
        "łazienk", "lazienk", "kuchn", "pralk", "hydraulik", "mechanik", "samoch", "lodówk",
        "lodowk", "dzieci", "dziecka", "mamy", "mama", "tata", "taty", "babci", "urodzin",
        "prezent", "weterynarz", "paczk", "pościel", "posciel", "okna", "trawnik",
    ),
    "nauka": (
        "egzamin", "kolokw", "kolos", "laborator", "sprawozdan", "wykład", "wyklad",
        "ćwicze", "cwicze", "studi", "uczelni", "notatk", "powtórz", "powtorz", "nauczy",
        "nauk", "książk", "ksiazk", "kurs", "promotor", "semestr", "sesj", "matematy",
        "fizyk", "chemi", "angielsk", "słówk", "slowk", "fiszk", "artykuł", "artykul",
        "magistersk", "licencjack", "dyplom", "seminari", "referat", "esej", "zaliczeni",
        "kartkówk", "kartkowk", "sprawdzian", "lektur", "zadania domowe", "nlp",
    ),
}


def keyword_label(text: str) -> Optional[str]:
    """Etykieta z regul slownikowych: klasa z najwieksza liczba trafien (remis/brak -> None)."""
    words = clean_text(text).split()
    joined = " ".join(words)
    scores = {
        label: sum(1 for w in words for k in keys if " " not in k and w.startswith(k))
        + sum(1 for k in keys if " " in k and k in joined)
        for label, keys in KEYWORDS.items()
    }
    best = max(scores.values())
    winners = [label for label, s in scores.items() if s == best]
    return winners[0] if best > 0 and len(winners) == 1 else None


# ---------------------------------------------------------- zbiory zewnetrzne
def _balanced_sample(rows: list[tuple[str, str]], per_class: int,
                     seed: int = 42) -> list[tuple[str, str]]:
    """Po `per_class` losowych przykladow z kazdej klasy (zbior zbalansowany, powtarzalny)."""
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for label in sorted(by_label):
        items = by_label[label]
        out.extend(rng.sample(items, min(per_class, len(items))))
    rng.shuffle(out)
    return out


def load_agnews(per_class: int = 1000) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise BotError("Zbior agnews wymaga pakietu 'datasets' (pip install datasets).") from exc
    ds = load_dataset("sh0416/ag_news", split="train")
    rows = []
    for item in ds:
        text = f"{item['title']}. {item['description']}".replace("\\", " ")
        rows.append((re.sub(r"\s+", " ", text).strip(), AGNEWS_LABELS[int(item["label"])]))
    return _balanced_sample(rows, per_class)


def load_20newsgroups(per_class: int = 1000) -> list[tuple[str, str]]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="all", categories=list(NEWSGROUPS_CATEGORIES),
        remove=("headers", "footers", "quotes"),   # bez naglowkow - inaczej klasa "wycieka"
    )
    rows = []
    for text, target in zip(data.data, data.target):
        text = re.sub(r"\s+", " ", text).strip()[:2000]    # posty bywaja bardzo dlugie
        if len(text.split()) >= 5:                          # pomijamy puste/szczatkowe posty
            rows.append((text, NEWSGROUPS_CATEGORIES[data.target_names[target]]))
    return _balanced_sample(rows, per_class)


EXTERNAL_LOADERS: dict[str, Callable[[int], list[tuple[str, str]]]] = {
    "agnews": load_agnews,
    "20newsgroups": load_20newsgroups,
}


# ------------------------------------------------ walidacja na wlasnych zdaniach
@dataclass(frozen=True)
class ValidationSentence:
    text: str
    expected: str
    note: str = ""          # komentarz: dlaczego zdanie jest latwe/trudne


CUSTOM_SENTENCES: dict[str, list[ValidationSentence]] = {
    "todo": [
        ValidationSentence("Napisać raport z laboratorium", "nauka",
                           "'raport' kojarzy sie z praca, 'laboratorium' z nauka"),
        ValidationSentence("Wysłać prezentację klientowi przed piątkiem", "praca"),
        ValidationSentence("Przygotować slajdy na obronę inżynierki", "nauka",
                           "brak slow ze zbioru treningowego (slajdy, obrona, inzynierka)"),
        ValidationSentence("Kupić pomidory i ser na pizzę", "dom"),
        ValidationSentence("Odkurzyć samochód i umyć dywaniki", "dom"),
        ValidationSentence("Zrobić przegląd kodu kolegi z zespołu", "praca"),
        ValidationSentence("Zapłacić za wodę i śmieci", "dom"),
        ValidationSentence("Przeczytać dwa rozdziały do zajęć z psychologii", "nauka"),
        ValidationSentence("Napisać do HR w sprawie umowy o pracę", "praca"),
        ValidationSentence("Pouczyć się do poprawki z mikroekonomii", "nauka",
                           "odmiana 'pouczyc' i nowy przedmiot"),
        ValidationSentence("Zabrać kota do groomera", "dom", "rzadkie slowo 'groomer'"),
        ValidationSentence("Ogarnąć wyniki A/B testów dla product ownera", "praca",
                           "slang i angielskie wtracenia"),
    ],
    "agnews": [
        ValidationSentence("The striker scored twice as the home team won the cup final", "Sports"),
        ValidationSentence("Central bank raises interest rates to fight inflation", "Business"),
        ValidationSentence("Researchers unveil a new chip that speeds up AI training", "SciTech"),
        ValidationSentence("Peace talks resume between the two countries after border clashes",
                           "World"),
        ValidationSentence("Shares of the carmaker fell after weak quarterly earnings", "Business"),
        ValidationSentence("NASA probe sends back first images from the far side of the moon",
                           "SciTech"),
        ValidationSentence("Tennis champion withdraws from the open with a knee injury", "Sports"),
        ValidationSentence("Prime minister resigns amid corruption scandal", "World"),
        ValidationSentence("Tech giant buys gaming startup for two billion dollars", "Business",
                           "mieszanka slow z Business i SciTech"),
        ValidationSentence("Apple releases a security update for the iPhone", "SciTech"),
        ValidationSentence("Olympic committee bans athletes over doping", "Sports"),
        ValidationSentence("Floods displace thousands across southern Asia", "World"),
    ],
    "20newsgroups": [
        ValidationSentence("How do I render a 3D mesh with smooth shading in OpenGL?", "graphics"),
        ValidationSentence("Which JPEG compression format gives the best image quality?",
                           "graphics"),
        ValidationSentence("The goalie made 40 saves and the Penguins won in overtime", "hockey"),
        ValidationSentence("Who will win the Stanley Cup playoffs this year?", "hockey"),
        ValidationSentence("My doctor prescribed antibiotics for a chronic sinus infection",
                           "medicine"),
        ValidationSentence("Is there any evidence that this diet lowers cholesterol?", "medicine"),
        ValidationSentence("The shuttle launch was delayed because of a booster problem", "space"),
        ValidationSentence("NASA plans to send a probe to orbit Jupiter", "space"),
    ],
}
