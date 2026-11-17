"""Fasada klasyfikacji dla bota: przygotowanie danych, trening (z cache), predykcja, ocena.

Modele sa trenowane leniwie - przy pierwszym /classifier lub /evaluate dla danej
konfiguracji - i trzymane w pamieci oraz na dysku (models/*.joblib). Klucz modelu
zawiera "odcisk" danych (hash tekstow i etykiet), wiec zmiana etykiet albo nowych
zadan automatycznie wymusza ponowny trening.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from src.classification import datasets as ds
from src.classification.metrics import Report
from src.classification.methods import chosen_methods, get_method
from src.classification.pipeline import (
    Prediction, Split, TrainedModel, evaluate, predict, split_data, train,
)
from src.classification.representations import (
    DEFAULT_EMBEDDING_MODEL, REPRESENTATIONS, repr_params,
)
from src.errors import InvalidInputError
from src.nlp.cleaning import clean_text
from src.storage.repository import TaskRepository

log = logging.getLogger(__name__)

CLEANING_MODES = ("cleaning",)


@dataclass(frozen=True)
class PrepareResult:
    dataset: str
    labeled: int
    unlabeled: int
    imported: int
    classes: list[str]


@dataclass(frozen=True)
class ClassifyResult:
    prediction: Prediction
    dataset: str
    repr_kind: str
    method: str
    cleaned_text: str


@dataclass(frozen=True)
class EvaluateResult:
    report: Report
    model: TrainedModel
    dataset: str


@dataclass(frozen=True)
class ValidationRow:
    sentence: ds.ValidationSentence
    predicted: str
    probability: Optional[float]

    @property
    def correct(self) -> bool:
        return self.predicted == self.sentence.expected


class ClassificationService:
    def __init__(self, repo: TaskRepository, models_dir: str | Path = "models",
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL, surname_initial: str = "C",
                 samples_per_class: int = 1000) -> None:
        self.repo = repo
        self.models_dir = Path(models_dir)
        self.embedding_model = embedding_model
        self.surname_initial = surname_initial
        self.samples_per_class = samples_per_class
        self._cache: dict[str, tuple[TrainedModel, Split]] = {}
        self._lock = threading.Lock()   # dwa rownolegle /evaluate nie trenuja tego samego 2x

    # ------------------------------------------------------------- informacje
    @property
    def chosen(self) -> list[str]:
        return [m.key for m in chosen_methods(self.surname_initial)]

    def active_dataset(self, owner_id: int) -> str:
        return self.repo.get_active_dataset(owner_id) or "todo"

    # ------------------------------------------------------ przygotowanie danych
    def prepare(self, owner_id: int, dataset: Optional[str] = None,
                seed: bool = False) -> PrepareResult:
        name = ds.resolve_dataset(dataset)
        if name == "todo":
            result = self._prepare_todo(owner_id, seed)
        else:
            rows = ds.EXTERNAL_LOADERS[name](self.samples_per_class)
            self.repo.replace_samples(name, rows)          # nadpisanie poprzednich etykiet
            classes = sorted({label for _, label in rows})
            result = PrepareResult(name, len(rows), 0, 0, classes)
        self.repo.set_active_dataset(owner_id, name)
        return result

    def _prepare_todo(self, owner_id: int, seed: bool) -> PrepareResult:
        imported = 0
        if seed:
            existing = {t.description for t in self.repo.list_tasks(owner_id)}
            new = [(text, None) for text, _ in ds.load_seed() if text not in existing]
            imported = self.repo.bulk_create(owner_id, new)

        gold = dict(ds.load_seed())     # dla zadan z przykladowego zbioru znamy etykiete
        labels: dict[str, Optional[str]] = {}
        for task in self.repo.list_tasks(owner_id):
            label = gold.get(task.description) or ds.keyword_label(task.description)
            labels[task.id] = label or task.label   # brak reguly -> zostaw recznie nadana
        self.repo.set_labels(owner_id, labels)
        labeled = sum(1 for v in labels.values() if v)
        classes = sorted({v for v in labels.values() if v}) or list(ds.TODO_CLASSES)
        return PrepareResult("todo", labeled, len(labels) - labeled, imported, classes)

    def set_label(self, owner_id: int, task_ref: str, label: str):
        label = label.strip().lower()
        if label in ("none", "-", "brak"):
            return self.repo.set_label(owner_id, task_ref, None)
        if not label or len(label) > 40:
            raise InvalidInputError("Etykieta musi miec od 1 do 40 znakow.")
        return self.repo.set_label(owner_id, task_ref, label)

    # ------------------------------------------------------------------- dane
    def _data(self, owner_id: int) -> tuple[str, list[str], list[str]]:
        name = self.active_dataset(owner_id)
        if name == "todo":
            rows = [(t.description, t.label) for t in self.repo.list_tasks(owner_id) if t.label]
            if not rows:
                raise InvalidInputError(
                    "Brak zadan z etykietami. Uzyj /prepare_data_for_classification todo "
                    "(albo z --seed, by dodac 150 przykladowych zadan)."
                )
        else:
            rows = self.repo.list_samples(name)
            if not rows:
                raise InvalidInputError(
                    f"Zbior '{name}' nie jest wczytany. Uzyj /prepare_data_for_classification {name}"
                )
        texts = [t for t, _ in rows]
        labels = [label for _, label in rows]
        return name, texts, labels

    def _model(self, owner_id: int, repr_kind: str, method: str
               ) -> tuple[str, TrainedModel, Split]:
        repr_kind = repr_kind.strip().lower()
        if repr_kind not in REPRESENTATIONS:
            repr_params(repr_kind)                     # rzuca czytelny blad
        spec = get_method(method)
        name, texts, labels = self._data(owner_id)
        digest = hashlib.sha1(
            "\x1f".join(f"{t}\x1e{label}" for t, label in zip(texts, labels)).encode("utf-8")
        ).hexdigest()[:12]
        owner_part = f"u{owner_id}_" if name == "todo" else ""
        key = f"{name}_{owner_part}{repr_kind}_{spec.key}_{digest}"

        with self._lock:
            if key in self._cache:
                model, split = self._cache[key]
                return name, model, split
            split = split_data(texts, labels)
            path = self.models_dir / f"{key}.joblib"
            model: Optional[TrainedModel] = None
            if path.exists():
                try:
                    model = joblib.load(path)
                except Exception:                      # uszkodzony / niekompatybilny plik
                    log.warning("Nie udalo sie wczytac %s - trenuje od nowa", path)
            if model is None:
                log.info("Trening %s (%d przykladow)", key, len(split.x_train))
                model = train(split, repr_kind, spec.key, self.embedding_model)
                self.models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
            self._cache[key] = (model, split)
            return name, model, split

    # ------------------------------------------------------------- komendy
    def classify(self, owner_id: int, text: str, repr_kind: str, method: str,
                 cleaning: str = "cleaning") -> ClassifyResult:
        if cleaning.lower() not in CLEANING_MODES:
            raise InvalidInputError(
                f"Pierwszy parametr musi byc trybem czyszczenia: {', '.join(CLEANING_MODES)}"
            )
        if not clean_text(text):
            raise InvalidInputError("Po czyszczeniu tekst jest pusty - podaj zdanie ze slowami.")
        name, model, _ = self._model(owner_id, repr_kind, method)
        # Pipeline sam wywoluje clean_text (krok 'cleaning'), tu tylko do wyswietlenia.
        return ClassifyResult(predict(model, text), name, model.repr_kind, model.method,
                              clean_text(text))

    def evaluate(self, owner_id: int, repr_kind: str, method: str) -> EvaluateResult:
        name, model, split = self._model(owner_id, repr_kind, method)
        return EvaluateResult(evaluate(model, split), model, name)

    def validate(self, owner_id: int, repr_kind: str, method: str
                 ) -> tuple[str, TrainedModel, list[ValidationRow]]:
        name, model, _ = self._model(owner_id, repr_kind, method)
        sentences = ds.CUSTOM_SENTENCES.get(name, [])
        rows = []
        for sentence in sentences:
            pred = predict(model, sentence.text)
            rows.append(ValidationRow(sentence, pred.label, pred.probability))
        return name, model, rows
