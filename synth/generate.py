"""Generate the synthetic document set: cases → documents → clean and degraded page images.

    uv run python -m synth.generate --per-type 5 --out data/docs_pilot     # small pilot
    uv run python -m synth.generate --per-type 80                            # full set

For each document it writes, under <out>/<seed>/:
    <doc_type>.html          what Gemini wrote
    <doc_type>.txt           the page text (what a digital PDF's text layer would give)
    <doc_type>.png           clean render
    <doc_type>_<kind>.jpg    degraded copy, kind = phone | scan
and appends one line per document to <out>/index.jsonl with the labels.
Re-running skips documents that already exist, so an interrupted run can resume.
"""

import argparse
import html as html_lib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from synth.render import Renderer, degrade
from synth.sample import sample_case
from synth.write_docs import DOC_SPECS, doc_types_for, write_doc


def page_text(html: str) -> str:
    html = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|h\d|li|table)>", "\n", html)
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", html))
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def plan(per_type: int, max_cases: int, types: list[str] | None = None, start_seed: int = 0) -> list[tuple[int, str]]:
    """Pick (seed, doc_type) jobs so every wanted document type reaches per_type, using realistic claim files."""
    wanted = types or list(DOC_SPECS)
    counts, jobs = Counter(), []
    for seed in range(start_seed, start_seed + max_cases):
        if all(counts[t] >= per_type for t in wanted):
            break
        case = sample_case(seed)
        for doc_type in doc_types_for(case, random.Random(seed)):
            if doc_type in wanted and counts[doc_type] < per_type:
                counts[doc_type] += 1
                jobs.append((seed, doc_type))
    short = {t: per_type - counts[t] for t in wanted if counts[t] < per_type}
    if short:
        print(f"warning: ran out of cases before filling {short}")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("data/docs"))
    parser.add_argument("--workers", type=int, default=8, help="parallel Gemini calls")
    parser.add_argument("--max-cases", type=int, default=5000)
    parser.add_argument("--types", nargs="+", choices=list(DOC_SPECS), help="only these document types")
    parser.add_argument("--start-seed", type=int, default=0, help="first case seed; use a new range for a new set")
    args = parser.parse_args()

    jobs = plan(args.per_type, args.max_cases, args.types, args.start_seed)
    todo = [(s, t) for s, t in jobs if not (args.out / str(s) / f"{t}.html").exists()]
    print(f"{len(jobs)} documents planned, {len(todo)} to write")

    def write(job):
        seed, doc_type = job
        html, missing = write_doc(sample_case(seed), doc_type, seed)
        folder = args.out / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{doc_type}.html").write_text(html)
        return seed, doc_type, missing

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(write, job) for job in todo]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                seed, doc_type, missing = future.result()
                if missing:
                    failed.append((seed, doc_type, missing))
            except Exception as e:  # keep going; the job is retried on the next run
                print(f"  error: {str(e)[:160]}")
                if "Reauthentication" in str(e):
                    print("Google login expired: run `gcloud auth application-default login`, then re-run to resume.")
                    for f in futures:
                        f.cancel()
                    break
            if i % 10 == 0 or i == len(todo):
                print(f"  wrote {i}/{len(todo)}")

    # Render and degrade everything written, then rebuild the index from disk.
    index = []
    with Renderer() as renderer:
        for seed, doc_type in jobs:
            folder = args.out / str(seed)
            html_path = folder / f"{doc_type}.html"
            if not html_path.exists():
                continue
            html = html_path.read_text()
            if not (folder / f"{doc_type}.png").exists():
                (folder / f"{doc_type}.txt").write_text(page_text(html))
                clean = renderer.render(html)
                clean.save(folder / f"{doc_type}.png")
                noisy, kind = degrade(clean, random.Random(f"{seed}-{doc_type}-degrade"))
                noisy.save(folder / f"{doc_type}_{kind}.jpg", quality=95)
            kind = next(p.stem.split("_")[-1] for p in folder.glob(f"{doc_type}_*.jpg"))
            case = sample_case(seed)
            index.append({
                "seed": seed,
                "doc_type": doc_type,
                "outcome": case.outcome,
                "rejection_reason": case.rejection_reason,
                "html": str(html_path),
                "text": str(folder / f"{doc_type}.txt"),
                "clean_image": str(folder / f"{doc_type}.png"),
                "degraded_image": str(folder / f"{doc_type}_{kind}.jpg"),
                "degradation": kind,
            })
    with (args.out / "index.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in index)

    print(f"{len(index)} documents indexed in {args.out / 'index.jsonl'}")
    print("per type:", dict(Counter(r["doc_type"] for r in index)))
    if failed:
        print(f"{len(failed)} documents kept despite missing key facts:")
        for seed, doc_type, missing in failed:
            print(f"  seed {seed} {doc_type}: missing {missing}")


if __name__ == "__main__":
    main()
