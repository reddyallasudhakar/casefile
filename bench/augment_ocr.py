"""F5b: extra OCR readings of training and RL pages, from freshly degraded images.

    uv run python -m bench.augment_ocr --copies 3

For every extraction page in the train and rl splits, degrade its clean render K more times
(new random phone/scan effects), OCR each with Cloud Vision, and save <doc_type>.ocr<k>.txt.
The model then sees each page's facts through several different kinds of OCR noise. Test and
validation pages are never touched, so they keep their single original degraded image.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.cloud import vision
from PIL import Image

import synth.net  # noqa: F401  (IPv4-first, see synth/net.py)
from bench.extract import DOC_TYPES, SOURCES, split_of
from synth.render import degrade


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, default=3)
    args = parser.parse_args()

    rows = [r for src in SOURCES if src.exists() for r in map(json.loads, src.open())
            if r["doc_type"] in DOC_TYPES and split_of(r["seed"]) in ("train", "rl")]
    jobs = [(r, k) for r in rows for k in range(1, args.copies + 1)
            if not Path(r["text"]).with_suffix(f".ocr{k}.txt").exists()]
    print(f"{len(rows)} train/rl pages, {len(jobs)} new OCR readings to make")

    client = vision.ImageAnnotatorClient(transport="rest")

    def make(job):
        row, k = job
        clean = Image.open(row["clean_image"]).convert("RGB")
        noisy, kind = degrade(clean, random.Random(f"{row['seed']}-{row['doc_type']}-aug{k}"))
        image_path = Path(row["clean_image"]).with_name(f"{row['doc_type']}_aug{k}_{kind}.jpg")
        noisy.save(image_path, quality=95)
        response = client.document_text_detection(image=vision.Image(content=image_path.read_bytes()))
        if response.error.message:
            raise RuntimeError(response.error.message)
        Path(row["text"]).with_suffix(f".ocr{k}.txt").write_text(response.full_text_annotation.text)

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(make, jobs), 1):
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
