"""Split the document set into train / val / test by case.

    uv run python -m bench.split

All pages of a case go to the same split, so a test page never shares a patient, hospital
or amounts with anything trained on. Cases are shuffled with a fixed seed; a draw is
rejected and re-drawn if any document type is too thin in val or test, since a per-type
score on a handful of pages is noise. The result is reproducible from the attempt number.
"""

import json
import random
from collections import Counter
from pathlib import Path

INDEX = Path("data/docs/index.jsonl")
OUT = Path("bench/splits.json")  # small, committed, so the test set is fixed for everyone
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_PER_TYPE = 8  # pages of every document type required in val and in test


def draw(seeds, attempt):
    shuffled = sorted(seeds)
    random.Random(f"casefile-split-{attempt}").shuffle(shuffled)
    n_train = round(len(shuffled) * RATIOS["train"])
    n_val = round(len(shuffled) * RATIOS["val"])
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def type_counts(rows, split):
    where = {seed: name for name, seeds in split.items() for seed in seeds}
    counts = {name: Counter() for name in split}
    for row in rows:
        counts[where[row["seed"]]][row["doc_type"]] += 1
    return counts


def main():
    rows = [json.loads(line) for line in INDEX.open()]
    seeds = {r["seed"] for r in rows}
    doc_types = sorted({r["doc_type"] for r in rows})

    for attempt in range(1000):
        split = draw(seeds, attempt)
        counts = type_counts(rows, split)
        if all(counts[s][t] >= MIN_PER_TYPE for s in ("val", "test") for t in doc_types):
            break
    else:
        raise SystemExit("no balanced split found; lower MIN_PER_TYPE")

    OUT.write_text(json.dumps({"attempt": attempt, "min_per_type": MIN_PER_TYPE, **split}, indent=1))

    print(f"accepted draw #{attempt}  ({len(seeds)} cases, {len(rows)} pages)\n")
    print(f"{'doc_type':<20}" + "".join(f"{s:>8}" for s in split))
    for t in doc_types:
        print(f"{t:<20}" + "".join(f"{counts[s][t]:>8}" for s in split))
    print(f"{'pages':<20}" + "".join(f"{sum(counts[s].values()):>8}" for s in split))
    print(f"{'cases':<20}" + "".join(f"{len(split[s]):>8}" for s in split))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
