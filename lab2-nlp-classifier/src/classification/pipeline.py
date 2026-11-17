"""Budowa, trening i ocena Pipeline: cleaning -> reprezentacja -> (SVD) -> klasyfikator.

Czyste funkcje bez Telegrama i bazy - uzywa ich zarowno bot (ClassificationService),
jak i skrypt porownawczy (scripts/compare.py).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classification.metrics import Report, build_report
from src.classification.methods import RANDOM_STATE, DenseSVD, get_method
from src.classification.representations import (
    DEFAULT_EMBEDDING_MODEL, TextCleaner, build_representation, is_sparse_repr, repr_params,
)
from src.errors import InvalidInputError

TEST_SIZE = 0.2   # 80% trening / 20% test


def build_pipeline(repr_kind: str, method: str,
                   embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Pipeline:
    repr_params(repr_kind, embedding_model)            # walidacja nazwy reprezentacji
    spec = get_method(method)
    steps = [
        ("cleaning", TextCleaner()),                   # clean_text z zadania 1
        ("repr", build_representation(repr_kind, embedding_model)),
    ]
    if spec.needs_dense and is_sparse_repr(repr_kind):
        steps.append(("svd", DenseSVD(n_components=100)))   # rzadka -> gesta (LSA)
    steps.append(("clf", spec.factory(repr_kind)))
    return Pipeline(steps)


@dataclass(frozen=True)
class Split:
    x_train: list[str]
    x_test: list[str]
    y_train: list[str]
    y_test: list[str]


def split_data(texts: Sequence[str], labels: Sequence[str]) -> Split:
    """Stratyfikowany podzial 80/20 ze stalym random_state."""
    counts = Counter(labels)
    if len(counts) < 2:
        raise InvalidInputError("Do klasyfikacji potrzebne sa co najmniej 2 klasy z etykietami.")
    too_small = [c for c, n in counts.items() if n < 5]
    if too_small:
        raise InvalidInputError(
            f"Za malo przykladow w klasach: {', '.join(too_small)} (min. 5 na klase). "
            "Dodaj zadania lub uzyj /prepare_data_for_classification todo --seed."
        )
    x_tr, x_te, y_tr, y_te = train_test_split(
        list(texts), list(labels), test_size=TEST_SIZE, stratify=list(labels),
        random_state=RANDOM_STATE,
    )
    return Split(x_tr, x_te, y_tr, y_te)


@dataclass
class TrainedModel:
    pipeline: Pipeline
    classes: list[str]
    repr_kind: str
    method: str
    n_train: int
    n_test: int
    train_seconds: float


def train(split: Split, repr_kind: str, method: str,
          embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> TrainedModel:
    pipe = build_pipeline(repr_kind, method, embedding_model)
    start = time.perf_counter()
    pipe.fit(split.x_train, split.y_train)
    elapsed = time.perf_counter() - start
    return TrainedModel(pipe, [str(c) for c in pipe.classes_], repr_kind,
                        get_method(method).key, len(split.x_train), len(split.x_test), elapsed)


def evaluate(model: TrainedModel, split: Split) -> Report:
    y_pred = model.pipeline.predict(split.x_test)
    return build_report(split.y_test, [str(p) for p in y_pred], model.classes)


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float | None
    ranking: list[tuple[str, float]]     # wszystkie klasy z prawdopodobienstwem (malejaco)


def predict(model: TrainedModel, text: str) -> Prediction:
    pipe = model.pipeline
    if hasattr(pipe, "predict_proba"):
        try:
            proba = np.asarray(pipe.predict_proba([text])[0], dtype=float)
        except (AttributeError, NotImplementedError):
            proba = None
        if proba is not None:
            order = np.argsort(-proba)
            ranking = [(model.classes[i], float(proba[i])) for i in order]
            return Prediction(ranking[0][0], ranking[0][1], ranking)
    label = str(pipe.predict([text])[0])
    return Prediction(label, None, [])
