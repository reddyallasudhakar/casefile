"""Hide the heading lines that give a page's document type away.

    uv run python -m bench.mask_titles

Many generated pages announce their type in a heading ("LETTER OF CLAIM REPUDIATION",
"FINAL INPATIENT BILL"). That makes classification a lookup, not understanding. For each
page this writes <doc_type>.masked.txt (from the clean text) and <doc_type>.ocr.masked.txt
(from the OCR text), with heading-like lines that name a type replaced by "[heading removed]".

Only headings are masked: short lines near the top of the page. A sentence in the body
saying "your claim has been repudiated" is content the model should understand, so it stays.
"""

import json
import re
from pathlib import Path

INDEX = Path("data/docs/index.jsonl")
HEADING_ZONE = 25  # only the first N non-empty lines can be headings
MAX_HEADING_CHARS = 90

TYPE_WORDS = [
    r"settlement (advice|letter|intimation|statement)", r"payment (advice|intimation)",
    r"repudiation", r"rejection", r"denial", r"pre[- ]?auth", r"cashless (facility|request|authori)",
    r"query", r"deficiency", r"requisition", r"additional documents",
    r"policy schedule", r"schedule of (insurance|benefits)", r"certificate of insurance",
    r"\bbill\b", r"invoice", r"bill of supply", r"\breceipt\b",
    r"discharge (summary|card)", r"lab(oratory)? report", r"radiology", r"investigation report",
    r"pathology", r"prescription", r"consultation (note|record)",
    r"health (card|insurance card)", r"e-?card", r"\bid card\b", r"member card",
    r"claim form", r"policy wording", r"terms (and|&) conditions", r"admission (form|record)",
    # Hindi headings that the generator sometimes added
    r"दावा", r"अस्वीकृति", r"निपटान", r"बिल", r"डिस्चार्ज", r"सारांश", r"रिपोर्ट", r"पॉलिसी", r"कार्ड",
]
TYPE_RE = re.compile("|".join(TYPE_WORDS), re.IGNORECASE)


def mask(text: str) -> tuple[str, int]:
    """Return (masked text, number of heading lines masked)."""
    out, seen, masked = [], 0, 0
    for line in text.splitlines():
        if line.strip():
            seen += 1
        is_heading = seen <= HEADING_ZONE and len(line.strip()) <= MAX_HEADING_CHARS
        if is_heading and TYPE_RE.search(line):
            out.append("[heading removed]")
            masked += 1
        else:
            out.append(line)
    return "\n".join(out), masked


def main():
    rows = [json.loads(line) for line in INDEX.open()]
    total = {"clean": 0, "ocr": 0}
    pages_with_mask = {"clean": 0, "ocr": 0}
    for r in rows:
        clean = Path(r["text"])
        for name, src, dst in [
            ("clean", clean, clean.with_suffix(".masked.txt")),
            ("ocr", clean.with_suffix(".ocr.txt"), clean.with_suffix(".ocr.masked.txt")),
        ]:
            text, n = mask(src.read_text())
            dst.write_text(text)
            total[name] += n
            pages_with_mask[name] += n > 0
    for name in total:
        print(f"{name}: masked {total[name]} heading lines; {pages_with_mask[name]}/{len(rows)} pages had at least one")


if __name__ == "__main__":
    main()
