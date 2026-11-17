"""Porownanie 6 konfiguracji (2 metody x 3 reprezentacje) do sprawozdania.

    python -m scripts.compare todo            # 150 zadan TODO z data/todo_seed.csv
    python -m scripts.compare agnews          # AG News (SAMPLES_PER_CLASS na klase)
    python -m scripts.compare 20newsgroups --per-class 500
    python -m scripts.compare agnews --methods logreg,mlp   # inne metody

Wyniki trafiaja do reports/<zbior>/: results.md (macierze + tabela), results.csv,
cm_<repr>_<metoda>.png (wykresy macierzy pomylek).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from src.classification import datasets as ds
from src.classification.methods import chosen_methods, get_method
from src.classification.metrics import format_report
from src.classification.pipeline import evaluate, predict, split_data, train
from src.classification.representations import REPRESENTATIONS, describe_params
from src.config import load_settings


def _plot(report, title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return
    disp = ConfusionMatrixDisplay(report.matrix, display_labels=report.classes)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    settings = load_settings(require_token=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="todo")
    parser.add_argument("--per-class", type=int, default=settings.samples_per_class)
    parser.add_argument("--methods", default="",
                        help="np. adaboost,tree (domyslnie wg SURNAME_INITIAL)")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    name = ds.resolve_dataset(args.dataset)
    rows = ds.load_seed() if name == "todo" else ds.EXTERNAL_LOADERS[name](args.per_class)
    texts, labels = [t for t, _ in rows], [label for _, label in rows]
    split = split_data(texts, labels)
    methods = ([get_method(m) for m in args.methods.split(",")] if args.methods
               else chosen_methods(settings.surname_initial))

    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Wyniki: {name}", "",
          f"Liczba przykladow: {len(rows)} | trening: {len(split.x_train)} | "
          f"test: {len(split.x_test)} (stratyfikacja, random_state=42)", "",
          "Metody: " + ", ".join(f"{m.name} (`{m.key}`)" for m in methods), "",
          "## Parametry reprezentacji", "", "```"]
    md += [describe_params(k, settings.embedding_model) for k in REPRESENTATIONS]
    md += ["```", ""]
    for m in methods:
        md += [f"### {m.name}", ""] + [f"* `{k}`: {v}" for k, v in m.params.items()] + [""]

    summary, validation = [], []
    sentences = ds.CUSTOM_SENTENCES.get(name, [])
    for m in methods:
        for kind in REPRESENTATIONS:
            print(f"-> {kind} + {m.key} ...", flush=True)
            model = train(split, kind, m.key, settings.embedding_model)
            report = evaluate(model, split)
            preds = [predict(model, s.text) for s in sentences]
            custom_ok = sum(p.label == s.expected for p, s in zip(preds, sentences))
            summary.append({
                "repr": kind, "method": m.key, "ACC": round(report.acc, 4),
                "ERR": round(report.err, 4), "Precision_macro": round(report.macro("precision"), 4),
                "Recall_macro": round(report.macro("recall"), 4),
                "F1_macro": round(report.macro("f1"), 4),
                "F1_weighted": round(report.weighted("f1"), 4),
                "custom_acc": round(custom_ok / len(sentences), 4) if sentences else "",
                "train_s": round(model.train_seconds, 1),
            })
            validation.append((kind, m.key, preds))
            md += [f"## {kind} + {m.name}", "", "```", format_report(report), "```", "",
                   f"![cm](cm_{kind}_{m.key}.png)", ""]
            _plot(report, f"{name}: {kind} + {m.name}", out / f"cm_{kind}_{m.key}.png")
            print(f"   ACC={report.acc:.3f} F1={report.macro('f1'):.3f} "
                  f"custom={custom_ok}/{len(sentences)} ({model.train_seconds:.1f}s)")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    md += ["## Podsumowanie", "",
           "| repr | metoda | ACC | ERR | Precision (macro) | Recall (macro) | F1 (macro) "
           "| F1 (weighted) | wlasne zdania | czas treningu [s] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(summary, key=lambda r: -r["F1_macro"]):
        md.append(f"| {r['repr']} | {r['method']} | {r['ACC']:.3f} | {r['ERR']:.3f} | "
                  f"{r['Precision_macro']:.3f} | {r['Recall_macro']:.3f} | {r['F1_macro']:.3f} | "
                  f"{r['F1_weighted']:.3f} | {r['custom_acc']} | {r['train_s']} |")

    if sentences:
        cols = [f"{k}+{m}" for k, m, _ in validation]
        md += ["", "## Walidacja na wlasnych zdaniach", "",
               "| zdanie | oczekiwana | " + " | ".join(cols) + " |",
               "|---|---|" + "---|" * len(cols)]
        for i, s in enumerate(sentences):
            cells = []
            for _, _, preds in validation:
                p = preds[i]
                mark = "✅" if p.label == s.expected else "❌"
                prob = f" {p.probability:.2f}" if p.probability is not None else ""
                cells.append(f"{mark} {p.label}{prob}")
            md.append(f"| {s.text} | {s.expected} | " + " | ".join(cells) + " |")

    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Zapisano: {out / 'results.md'}")


if __name__ == "__main__":
    main()
