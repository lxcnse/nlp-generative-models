"""Macierz pomylek i metryki liczone z niej "recznie" (zgodnie z definicjami z polecenia).

Dla klasyfikacji wieloklasowej kazda klasa jest traktowana jak problem binarny
(one-vs-rest): klasa = "pozytywna", wszystkie pozostale = "negatywna".
    TP = M[k, k]                 (prawdziwa k, przewidziana k)
    FN = suma wiersza k - TP     (prawdziwa k, przewidziana inna)
    FP = suma kolumny k - TP     (prawdziwa inna, przewidziana k)
    TN = reszta
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """M[i, j] = liczba przykladow klasy classes[i] przewidzianych jako classes[j]."""
    index = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[index[t], index[p]] += 1
    return m


@dataclass(frozen=True)
class ClassStats:
    label: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def p(self) -> int:              # P = TP + FN (wszystkie prawdziwie pozytywne)
        return self.tp + self.fn

    @property
    def n(self) -> int:              # N = TN + FP
        return self.tn + self.fp

    @property
    def acc(self) -> float:          # (TP + TN) / (P + N)
        return _div(self.tp + self.tn, self.p + self.n)

    @property
    def err(self) -> float:          # (FP + FN) / (P + N)
        return _div(self.fp + self.fn, self.p + self.n)

    @property
    def precision(self) -> float:    # TP / (TP + FP)
        return _div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:       # TP / (TP + FN)
        return _div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:           # 2PR / (P + R)
        p, r = self.precision, self.recall
        return _div(2 * p * r, p + r)


@dataclass(frozen=True)
class Report:
    classes: list[str]
    matrix: np.ndarray
    per_class: list[ClassStats]

    @property
    def total(self) -> int:
        return int(self.matrix.sum())

    @property
    def acc(self) -> float:
        """Jakosc calego klasyfikatora: poprawne (przekatna) / wszystkie."""
        return _div(np.trace(self.matrix), self.total)

    @property
    def err(self) -> float:
        return 1.0 - self.acc if self.total else 0.0

    def _avg(self, attr: str, weighted: bool) -> float:
        values = np.array([getattr(s, attr) for s in self.per_class])
        if not weighted:
            return float(values.mean()) if len(values) else 0.0   # macro: kazda klasa po rowno
        support = np.array([s.p for s in self.per_class], dtype=float)
        return _div((values * support).sum(), support.sum())     # weighted: wg licznosci

    def macro(self, attr: str) -> float:
        return self._avg(attr, weighted=False)

    def weighted(self, attr: str) -> float:
        return self._avg(attr, weighted=True)


def build_report(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> Report:
    classes = list(classes)
    m = confusion(y_true, y_pred, classes)
    total = int(m.sum())
    stats = []
    for k, label in enumerate(classes):
        tp = int(m[k, k])
        fn = int(m[k, :].sum()) - tp
        fp = int(m[:, k].sum()) - tp
        stats.append(ClassStats(label, tp, fp, fn, total - tp - fn - fp))
    return Report(classes, m, stats)


# ------------------------------------------------------------------- formatowanie
def format_matrix(report: Report) -> str:
    header = [""] + [f"pred_{c}" for c in report.classes]
    rows = [[f"true_{c}"] + [str(v) for v in report.matrix[i]]
            for i, c in enumerate(report.classes)]
    widths = [max(len(r[j]) for r in [header, *rows]) for j in range(len(header))]
    fmt = lambda r: "  ".join(  # noqa: E731
        cell.ljust(widths[j]) if j == 0 else cell.rjust(widths[j]) for j, cell in enumerate(r))
    return "\n".join([fmt(header), *map(fmt, rows)])


def format_report(report: Report) -> str:
    lines = [format_matrix(report), ""]
    lines.append(f"ACC = {report.acc:.3f} | ERR = {report.err:.3f} | "
                 f"F1 (macro) = {report.macro('f1'):.3f} | n = {report.total}")
    lines.append("")
    lines.append("Metryki per klasa (one-vs-rest):")
    width = max(len(c) for c in report.classes)
    lines.append(f"{'klasa'.ljust(width)}  TP  FP  FN   TN   ACC   ERR  Prec   Rec    F1")
    for s in report.per_class:
        lines.append(
            f"{s.label.ljust(width)} {s.tp:>3} {s.fp:>3} {s.fn:>3} {s.tn:>4} "
            f"{s.acc:5.3f} {s.err:5.3f} {s.precision:5.3f} {s.recall:5.3f} {s.f1:5.3f}"
        )
    lines.append("")
    for name, fn in (("macro", report.macro), ("weighted", report.weighted)):
        lines.append(f"{name:<8}: Precision = {fn('precision'):.3f} | "
                     f"Recall = {fn('recall'):.3f} | F1 = {fn('f1'):.3f}")
    return "\n".join(lines)
