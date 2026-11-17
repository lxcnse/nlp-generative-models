"""Testy modulu klasyfikacji (bez sieci: tylko BoW/TF-IDF i zbior TODO)."""
import asyncio

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.bot import handlers
from src.bot.parsing import parse_classifier_args, parse_prepare_args, parse_repr_method
from src.classification import datasets as ds
from src.classification.methods import METHODS, chosen_methods, get_method
from src.classification.metrics import build_report
from src.classification.pipeline import build_pipeline, split_data
from src.classification.service import ClassificationService
from src.errors import InvalidInputError
from src.nlp.service import NlpService
from src.storage.repository import TaskRepository
from tests.test_handlers import make_update, replies


# ------------------------------------------------------------------- metryki
def test_metrics_match_task_example():
    # przyklad z polecenia: TP=250, FN=50, TN=600, FP=100
    y_true = ["pos"] * 300 + ["neg"] * 700
    y_pred = ["pos"] * 250 + ["neg"] * 50 + ["neg"] * 600 + ["pos"] * 100
    s = build_report(y_true, y_pred, ["pos", "neg"]).per_class[0]
    assert (s.tp, s.fn, s.tn, s.fp, s.p, s.n) == (250, 50, 600, 100, 300, 700)
    assert s.acc == pytest.approx(0.85) and s.err == pytest.approx(0.15)
    assert s.precision == pytest.approx(250 / 350) and s.recall == pytest.approx(250 / 300)


def test_metrics_agree_with_sklearn():
    rng = np.random.default_rng(0)
    classes = ["a", "b", "c"]
    y_true = list(rng.choice(classes, 200))
    y_pred = list(rng.choice(classes, 200))
    rep = build_report(y_true, y_pred, classes)
    assert (rep.matrix == confusion_matrix(y_true, y_pred, labels=classes)).all()
    assert rep.macro("f1") == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert rep.weighted("f1") == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert rep.macro("precision") == pytest.approx(
        precision_score(y_true, y_pred, average="macro"))
    assert rep.macro("recall") == pytest.approx(recall_score(y_true, y_pred, average="macro"))


# ------------------------------------------------------------------- metody
@pytest.mark.parametrize("letter, expected", [
    ("C", {"AdaBoost", "Decision Tree"}),
    ("M", {"Logistic Regression", "MLP (Neural Net)"}),     # przyklad z polecenia
    ("R", {"Random Forest", "RBF SVM"}),
])
def test_chosen_methods(letter, expected):
    assert {m.name for m in chosen_methods(letter)} == expected


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("kind", ["bow", "tfidf"])
def test_every_method_trains_on_sparse_repr(method, kind):
    rows = ds.load_seed()
    texts, labels = [t for t, _ in rows], [lbl for _, lbl in rows]
    pipe = build_pipeline(kind, method)
    pipe.fit(texts, labels)
    assert pipe.predict(["Kupić mleko"])[0] in ds.TODO_CLASSES
    if get_method(method).needs_dense:
        assert "svd" in pipe.named_steps     # GP/QDA dostaja gesta macierz po SVD


def test_naive_bayes_uses_gaussian_for_embeddings():
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    assert isinstance(get_method("nb").factory("embedding"), GaussianNB)
    assert isinstance(get_method("nb").factory("tfidf"), MultinomialNB)


def test_unknown_names():
    with pytest.raises(InvalidInputError):
        get_method("magic")
    with pytest.raises(InvalidInputError):
        build_pipeline("word2vec", "tree")


# ------------------------------------------------------------------- dane
def test_seed_is_balanced_and_split_is_stratified():
    rows = ds.load_seed()
    labels = [lbl for _, lbl in rows]
    assert {lbl: labels.count(lbl) for lbl in ds.TODO_CLASSES} == {"praca": 50, "dom": 50,
                                                                   "nauka": 50}
    split = split_data([t for t, _ in rows], labels)
    assert len(split.x_train) == 120 and len(split.x_test) == 30
    assert all(split.y_test.count(c) == 10 for c in ds.TODO_CLASSES)


@pytest.mark.parametrize("text, label", [
    ("Kupić mleko i chleb", "dom"),
    ("Powtórzyć do egzaminu z fizyki", "nauka"),
    ("Wysłać raport do szefa", "praca"),
    ("Coś zupełnie innego", None),
])
def test_keyword_label(text, label):
    assert ds.keyword_label(text) == label


# ------------------------------------------------------------------- parsowanie
def test_parse_classifier_args():
    a = parse_classifier_args('cleaning "Przygotować prezentację z NLP" tfidf mlp')
    assert (a.cleaning, a.text, a.repr_kind, a.method) == (
        "cleaning", "Przygotować prezentację z NLP", "tfidf", "mlp")
    b = parse_classifier_args("cleaning „Kupić mleko” BOW Tree")      # cudzyslowy z telefonu
    assert (b.text, b.repr_kind, b.method) == ("Kupić mleko", "bow", "tree")
    c = parse_classifier_args("cleaning Kupić mleko bow tree")         # bez cudzyslowu
    assert c.text == "Kupić mleko"
    with pytest.raises(InvalidInputError):
        parse_classifier_args('cleaning "x" tfidf')


def test_parse_other_args():
    assert parse_repr_method("tfidf adaboost", "evaluate") == ("tfidf", "adaboost")
    assert parse_prepare_args("todo --seed") == ("todo", True)
    assert parse_prepare_args("") == (None, False)
    with pytest.raises(InvalidInputError):
        parse_repr_method("tfidf", "evaluate")


# ------------------------------------------------------------------- serwis + bot
@pytest.fixture
def ctx(tmp_path):
    repo = TaskRepository(tmp_path / "c.db")
    repo.init_schema()
    clf = ClassificationService(repo, tmp_path / "models", surname_initial="C")
    from types import SimpleNamespace
    app = SimpleNamespace(bot_data={"repo": repo, "nlp": NlpService(6), "clf": clf})
    return SimpleNamespace(application=app)


def run(cmd, text, ctx):
    u = make_update(text)
    asyncio.run(cmd(u, ctx))
    return replies(u)


def test_bot_classification_flow(ctx):
    run(handlers.add_cmd, "/add Zrobić pranie i odkurzyć", ctx)
    out = run(handlers.prepare_cmd, "/prepare_data_for_classification todo --seed", ctx)
    assert out[0].startswith("Labels assigned to 151 tasks (dataset: todo). Classes: dom, nauka, praca")
    repo = ctx.application.bot_data["repo"]
    own = next(t for t in repo.list_tasks(7) if t.description.startswith("Zrobić pranie"))
    assert own.label == "dom"

    out = run(handlers.classifier_cmd, '/classifier cleaning "Kupić chleb i mleko" tfidf tree', ctx)
    assert out[0].startswith("Predicted class: ") and "| repr: tfidf | method: tree" in out[0]

    out = run(handlers.evaluate_cmd, "/evaluate bow adaboost", ctx)
    assert "pred_dom" in out[0] and "ACC =" in out[0] and "F1 (macro)" in out[0]

    out = run(handlers.label_cmd, f"/label {own.id[:8]} praca", ctx)
    assert "praca" in out[0]
    assert repo.get(7, own.id).label == "praca"          # nadpisanie etykiety

    out = run(handlers.validate_cmd, "/validate tfidf tree", ctx)
    assert "Poprawnie:" in out[0]


def test_bot_classification_errors(ctx):
    out = run(handlers.evaluate_cmd, "/evaluate tfidf tree", ctx)
    assert out[0].startswith("⚠️ Brak zadan z etykietami")
    out = run(handlers.classifier_cmd, '/classifier raw "x y" tfidf tree', ctx)
    assert out[0].startswith("⚠️")


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:      # schemat z zadania 1 (bez kolumny label)
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL,"
                     " description TEXT NOT NULL, relation TEXT, status TEXT NOT NULL DEFAULT"
                     " 'todo', priority TEXT NOT NULL DEFAULT 'medium', created_at TEXT NOT NULL,"
                     " updated_at TEXT NOT NULL)")
    repo = TaskRepository(path)
    repo.init_schema()
    task = repo.create(1, "Kupić mleko")
    assert repo.set_label(1, task.id, "dom").label == "dom"
