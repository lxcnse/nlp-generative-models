"""Metody klasyfikacji z porownania scikit-learn (plot_classifier_comparison).

Zaimplementowane sa wszystkie metody z listy, ale do sprawozdania kazda osoba
bierze 2 metody, ktorych angielskie nazwy sa alfabetycznie najblizsze pierwszej
literze nazwiska - wylicza to `chosen_methods()` (SURNAME_INITIAL w .env).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.errors import InvalidInputError

RANDOM_STATE = 42


@dataclass(frozen=True)
class MethodSpec:
    key: str                                   # nazwa w komendach bota, np. "adaboost"
    name: str                                  # nazwa angielska z dokumentacji sklearn
    factory: Callable[[str], BaseEstimator]    # reprezentacja -> nowy estymator
    needs_dense: bool = False                  # wymaga gestej macierzy (np. QDA, GP)
    params: dict[str, str] = field(default_factory=dict)  # parametr -> wyjasnienie
    aliases: tuple[str, ...] = ()


def _naive_bayes(kind: str) -> BaseEstimator:
    # MultinomialNB modeluje liczby wystapien i NIE przyjmuje wartosci ujemnych,
    # a embeddingi maja ujemne wspolrzedne -> dla nich GaussianNB.
    return GaussianNB() if kind == "embedding" else MultinomialNB(alpha=0.5)


METHODS: dict[str, MethodSpec] = {m.key: m for m in [
    MethodSpec(
        "adaboost", "AdaBoost",
        lambda kind: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
            n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE,
        ),
        params={
            "estimator": "DecisionTree(max_depth=2) - slaby klasyfikator; plytkie drzewo "
                         "(2 pytania) zamiast 'pienka', bo przy 3-4 klasach pienek rozroznia "
                         "naraz tylko jedna klase",
            "n_estimators": "300 kolejnych drzew; kazde skupia sie na przykladach, "
                            "na ktorych poprzednie sie mylily",
            "learning_rate": "0.5 - zmniejsza wage kazdego drzewa; wolniejsza nauka, "
                             "ale mniejsze przeuczenie",
            "random_state": "42 - powtarzalnosc wynikow",
        },
        aliases=("ada", "ada_boost"),
    ),
    MethodSpec(
        "tree", "Decision Tree",
        lambda kind: DecisionTreeClassifier(
            criterion="gini", max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE,
        ),
        params={
            "criterion": "gini - miara nieczystosci wezla przy wyborze podzialu",
            "max_depth": "None - drzewo rosnie, az liscie beda czyste (ryzyko przeuczenia)",
            "min_samples_leaf": "2 - lisc musi miec >= 2 przyklady; lagodzi przeuczenie "
                                "na pojedynczych, rzadkich slowach",
            "random_state": "42 - powtarzalnosc (losowa kolejnosc cech przy remisach)",
        },
        aliases=("decision_tree", "dt", "decisiontree"),
    ),
    MethodSpec(
        "gp", "Gaussian Process",
        lambda kind: GaussianProcessClassifier(random_state=RANDOM_STATE),
        needs_dense=True,
        params={"kernel": "domyslnie 1.0 * RBF(1.0); wymaga gestej macierzy -> TruncatedSVD"},
        aliases=("gaussian_process", "gaussianprocess"),
    ),
    MethodSpec(
        "linear_svm", "Linear SVM",
        lambda kind: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=3),
        params={"C": "1.0 - sila regularyzacji", "calibration": "CalibratedClassifierCV "
                "daje prawdopodobienstwa (LinearSVC sam ich nie zwraca)"},
        aliases=("svm", "linearsvm", "linear_svc"),
    ),
    MethodSpec(
        "logreg", "Logistic Regression",
        lambda kind: LogisticRegression(C=1.0, max_iter=2000),
        params={"C": "1.0 - odwrotnosc sily regularyzacji L2", "max_iter": "2000"},
        aliases=("logistic_regression", "lr", "logistic"),
    ),
    MethodSpec(
        "mlp", "MLP (Neural Net)",
        lambda kind: MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
                                   early_stopping=False, random_state=RANDOM_STATE),
        params={"hidden_layer_sizes": "(128,) - jedna warstwa ukryta", "alpha": "1e-3 - L2",
                "max_iter": "300 epok"},
        aliases=("neural_net", "nn"),
    ),
    MethodSpec(
        "nb", "Naive Bayes",
        _naive_bayes,
        params={"bow/tfidf": "MultinomialNB(alpha=0.5) - wygladzanie Laplace'a",
                "embedding": "GaussianNB - bo MultinomialNB nie przyjmuje liczb ujemnych"},
        aliases=("naive_bayes", "bayes"),
    ),
    MethodSpec(
        "knn", "Nearest Neighbors",
        lambda kind: KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
        params={"n_neighbors": "5", "metric": "cosine - podobienstwo kierunku wektorow",
                "weights": "distance - blizsi sasiedzi licza sie bardziej"},
        aliases=("nearest_neighbors", "kneighbors"),
    ),
    MethodSpec(
        "qda", "QDA",
        lambda kind: QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5),
        needs_dense=True,
        params={"solver": "eigen", "shrinkage": "0.5 - sciaga kowariancje klasy w strone diagonalnej; przy wiekszej liczbie "
                "cech niz przykladow w klasie macierz kowariancji jest osobliwa"},
    ),
    MethodSpec(
        "rf", "Random Forest",
        lambda kind: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                            n_jobs=-1),
        params={"n_estimators": "300 drzew glosujacych"},
        aliases=("random_forest", "forest"),
    ),
    MethodSpec(
        "rbf_svm", "RBF SVM",
        lambda kind: CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
            cv=3, ensemble=False),
        params={"kernel": "rbf", "C": "1.0", "gamma": "scale",
                "calibration": "CalibratedClassifierCV - prawdopodobienstwa"},
        aliases=("rbfsvm", "svc"),
    ),
]}

_ALIASES: dict[str, str] = {
    alias: spec.key for spec in METHODS.values() for alias in (spec.key, *spec.aliases)
}


def get_method(name: str) -> MethodSpec:
    key = _ALIASES.get(name.strip().lower().replace("-", "_"))
    if key is None:
        raise InvalidInputError(
            f"Nieznana metoda '{name}'. Dostepne: {', '.join(METHODS)}"
        )
    return METHODS[key]


def chosen_methods(surname_initial: str) -> list[MethodSpec]:
    """2 metody, ktorych nazwy sa alfabetycznie najblizsze pierwszej literze nazwiska.

    Litere wstawiamy w alfabetyczna liste nazw i bierzemy sasiadow:
      * nazwy zaczynajace sie na te litere maja pierwszenstwo,
      * brakujace miejsca uzupelnia najblizsza nazwa PRZED litera, potem PO niej.
    Np. 'M' -> Logistic Regression + MLP (jak w poleceniu); 'C' -> AdaBoost + Decision Tree.
    """
    letter = (surname_initial or "A").strip()[:1].upper() or "A"
    ordered = sorted(METHODS.values(), key=lambda m: m.name.upper())
    starting = [m for m in ordered if m.name.upper().startswith(letter)]
    if len(starting) >= 2:
        return starting[:2]
    before = [m for m in ordered if m.name.upper() < letter][::-1]   # najblizsze najpierw
    after = [m for m in ordered if m.name.upper() > letter and m not in starting]
    if starting:
        neighbours = before + after
    else:
        neighbours = before[:1] + after[:1] + before[1:] + after[1:]
    picked = starting + neighbours[: 2 - len(starting)]
    return sorted(picked, key=lambda m: m.name)


class DenseSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD (LSA) z bezpiecznym n_components dla malych slownikow.

    Metody wymagajace gestej macierzy (GP, QDA) dostaja zamiast 20 000 rzadkich
    kolumn ~100 gestych skladowych - toarray() na pelnym slowniku zjadlby pamiec.
    """

    def __init__(self, n_components: int = 100, random_state: int = RANDOM_STATE) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)
