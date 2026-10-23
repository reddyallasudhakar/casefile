"""Score document-type predictions on the frozen test set.

    uv run python -m bench.score preds/gemini.jsonl
    uv run python -m bench.score preds/qwen_ft.jsonl --compare preds/gemini.jsonl

A predictions file has one JSON line per test page:
    {"seed": 42, "doc_type": "hospital_bill", "predicted": "hospital_bill"}
(seed + doc_type identify the page; doc_type is the true label.)

Reports accuracy with a 95% interval, macro-F1, the invalid-output rate, per-type
precision/recall/F1 and the confusion matrix. With --compare, it also runs McNemar's exact
test on the two models' right/wrong answers for the same pages. Every model is scored by
this one script, so none is graded more leniently than another.
"""

import argparse
import json
import math
from pathlib import Path

INDEX = Path("data/docs/index.jsonl")
SPLITS = Path("bench/splits.json")


def test_pages(split: str = "test") -> dict[tuple[int, str], dict]:
    """The pages of a split, keyed by (seed, doc_type)."""
    seeds = set(json.loads(SPLITS.read_text())[split])
    rows = (json.loads(line) for line in INDEX.open())
    return {(r["seed"], r["doc_type"]): r for r in rows if r["seed"] in seeds}


def load_predictions(path: Path) -> dict[tuple[int, str], str]:
    preds = {}
    for line in path.open():
        p = json.loads(line)
        preds[(p["seed"], p["doc_type"])] = p["predicted"]
    return preds


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; better behaved than ±1.96·SE near 0 or 1."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def score(preds: dict, pages: dict) -> dict:
    labels = sorted({doc_type for _, doc_type in pages})
    # A page with no prediction, or a prediction outside the label set, counts as invalid and wrong.
    predicted = {key: preds.get(key) for key in pages}
    invalid = sum(1 for v in predicted.values() if v not in labels)
    correct = sum(1 for (seed, truth), v in predicted.items() if v == truth)

    confusion = {t: {p: 0 for p in labels + ["(invalid)"]} for t in labels}
    for (_, truth), v in predicted.items():
        confusion[truth][v if v in labels else "(invalid)"] += 1

    per_type = {}
    for t in labels:
        tp = confusion[t][t]
        fp = sum(confusion[other][t] for other in labels if other != t)
        fn = sum(n for p, n in confusion[t].items() if p != t)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_type[t] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    n = len(pages)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_ci": wilson_interval(correct, n),
        "macro_f1": sum(m["f1"] for m in per_type.values()) / len(labels),
        "invalid_rate": invalid / n,
        "per_type": per_type,
        "confusion": confusion,
        "labels": labels,
        "right": {key: predicted[key] == key[1] for key in pages},
    }


def mcnemar(right_a: dict, right_b: dict) -> dict:
    """Exact McNemar test: of the pages where exactly one model is right, is the split
    between 'only A' and 'only B' too lopsided to be chance (a fair coin)?"""
    only_a = sum(1 for k in right_a if right_a[k] and not right_b[k])
    only_b = sum(1 for k in right_a if right_b[k] and not right_a[k])
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def report(name: str, s: dict) -> None:
    lo, hi = s["accuracy_ci"]
    print(f"\n== {name} ==")
    print(f"accuracy   {s['accuracy']:.1%}  ({s['correct']}/{s['n']}, 95% interval {lo:.1%}–{hi:.1%})")
    print(f"macro-F1   {s['macro_f1']:.3f}")
    print(f"invalid    {s['invalid_rate']:.1%}")

    print(f"\n{'type':<20}{'prec':>6}{'rec':>6}{'f1':>6}{'n':>4}")
    for t, m in s["per_type"].items():
        print(f"{t:<20}{m['precision']:>6.2f}{m['recall']:>6.2f}{m['f1']:>6.2f}{m['support']:>4}")

    labels = s["labels"]
    short = [t[:4] for t in labels] + ["inv"]
    print("\nconfusion (rows = truth, columns = predicted)")
    print(f"{'':<20}" + "".join(f"{c:>5}" for c in short))
    for t in labels:
        row = s["confusion"][t]
        print(f"{t:<20}" + "".join(f"{(row[p] or '.'):>5}" for p in labels + ["(invalid)"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path, help="a second predictions file for McNemar's test")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    pages = test_pages(args.split)
    a = score(load_predictions(args.predictions), pages)
    report(args.predictions.stem, a)

    if args.compare:
        b = score(load_predictions(args.compare), pages)
        report(args.compare.stem, b)
        m = mcnemar(a["right"], b["right"])
        verdict = "a real difference" if m["p_value"] < 0.05 else "not distinguishable from chance"
        print(f"\n== McNemar: {args.predictions.stem} vs {args.compare.stem} ==")
        print(f"only {args.predictions.stem} right: {m['only_a']}   only {args.compare.stem} right: {m['only_b']}")
        print(f"p = {m['p_value']:.3f}  → {verdict} (at 5%)")


if __name__ == "__main__":
    main()
