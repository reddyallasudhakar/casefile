"""OCR every degraded page image with Google Cloud Vision.

    uv run python -m bench.ocr

Writes <doc_type>.ocr.txt next to each page's clean <doc_type>.txt, so every model can be
tested on the text a real upload would actually give: misread characters, broken tables,
lines out of order. Skips pages already done. Cloud Vision's first 1,000 pages a month are free.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.cloud import vision

import synth.net  # noqa: F401  (IPv4-first, see synth/net.py)

INDEX = Path("data/docs/index.jsonl")


def ocr_path(row: dict) -> Path:
    return Path(row["text"]).with_suffix(".ocr.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.open()]
    todo = [r for r in rows if not ocr_path(r).exists()]
    print(f"{len(rows)} pages, {len(todo)} to OCR")

    # REST rather than gRPC: it goes through Python's own sockets, so synth.net's IPv4 fix applies.
    client = vision.ImageAnnotatorClient(transport="rest")

    def read(row):
        image = vision.Image(content=Path(row["degraded_image"]).read_bytes())
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(response.error.message)
        ocr_path(row).write_text(response.full_text_annotation.text)

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(read, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
