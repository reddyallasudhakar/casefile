"""F5: extract claim facts from a page as JSON, and score the extraction.

    uv run python -m bench.extract stats                         # gold coverage per type and split
    uv run python -m bench.extract predict gemini --variant ocr  # writes preds/f5/<name>.jsonl
    uv run python -m bench.extract predict qwen --size 0.5B --variant ocr [--adapter adapters/f5-qwen-0.5B]
    uv run python -m bench.extract score preds/f5/a.jsonl [--compare preds/f5/b.jsonl]

The model is told the document type (F2 solved that) and returns one JSON object with fixed
keys. Gold answers come from each page's claim case, filtered to the facts that actually
appear in the page's clean text, so a model is never marked wrong for a fact the page doesn't
contain. A page counts as correct only if every gold field on it is right.
"""

import argparse
import json
import random
import re
import time
from datetime import date
from pathlib import Path

from synth.sample import sample_case

# --- the pages -------------------------------------------------------------------------

DOC_TYPES = ["settlement_letter", "hospital_bill", "policy_schedule", "discharge_summary"]
SOURCES = [Path("data/docs/index.jsonl"), Path("data/docs_f5/index.jsonl"), Path("data/docs_rl/index.jsonl")]
F2_SPLITS = json.loads(Path("bench/splits.json").read_text())


def split_of(seed: int) -> str:
    """F2's frozen split for the original cases; a fixed hash split (70/15/15) for the F5 additions;
    "rl" for cases from seed 20,000 on, which only GRPO sees (never SFT, validation or test)."""
    if seed >= 20_000:
        return "rl"
    for name in ("train", "val", "test"):
        if seed in F2_SPLITS[name]:
            return name
    r = random.Random(f"casefile-f5-split-{seed}").random()
    return "train" if r < 0.70 else "val" if r < 0.85 else "test"


def pages(split: str, variant: str = "clean") -> list[dict]:
    """Extraction pages in a split: {seed, doc_type, text, gold}.
    variant: clean | ocr (the original degraded image) | ocr1, ocr2… (F5b re-degraded copies)."""
    out = []
    for source in SOURCES:
        if not source.exists():
            continue
        for row in map(json.loads, source.open()):
            if row["doc_type"] not in DOC_TYPES or split_of(row["seed"]) != split:
                continue
            clean_path = Path(row["text"])
            text_path = clean_path if variant == "clean" else clean_path.with_suffix(f".{variant}.txt")
            if not text_path.exists():
                continue
            gold = gold_fields(row["seed"], row["doc_type"], clean_path.read_text())
            if gold:
                out.append({"seed": row["seed"], "doc_type": row["doc_type"], "text": text_path.read_text(), "gold": gold})
    return sorted(out, key=lambda p: (p["seed"], p["doc_type"]))


# --- schemas ---------------------------------------------------------------------------

FIELDS = {  # field → (type, description for the prompt)
    "settlement_letter": {
        "patient_name": ("name", "the patient's full name"),
        "policy_number": ("id", "the policy number exactly as printed"),
        "claimed_amount": ("amount", "total amount claimed, in rupees"),
        "approved_amount": ("amount", "amount approved or paid, in rupees"),
        "deduction_amounts": ("amounts", "list of each separate deduction amount in rupees (empty list if none)"),
    },
    "hospital_bill": {
        "patient_name": ("name", "the patient's full name"),
        "admission_date": ("date", "date of admission"),
        "discharge_date": ("date", "date of discharge"),
        "total_amount": ("amount", "grand total of the bill, in rupees"),
    },
    "policy_schedule": {
        "policy_number": ("id", "the policy number exactly as printed"),
        "patient_name": ("name", "full name of the insured member who was hospitalised"),
        "sum_insured": ("amount", "sum insured, in rupees"),
        "copay_percent": ("amount", "co-payment percentage as a number (0 if none)"),
        "room_rent_limit": ("amount", "room rent limit in rupees per day (null if no limit)"),
    },
    "discharge_summary": {
        "patient_name": ("name", "the patient's full name"),
        "admission_date": ("date", "date of admission"),
        "discharge_date": ("date", "date of discharge"),
    },
}


def gold_fields(seed: int, doc_type: str, clean_text: str) -> dict:
    """True values from the case, keeping only those visible in the page's clean text."""
    case = sample_case(seed)
    p, h = case.policy, case.hospitalisation
    truth = {
        "patient_name": case.patient.name,
        "policy_number": p.policy_number,
        "claimed_amount": round(case.claimed),
        "approved_amount": round(case.approved),
        "deduction_amounts": [round(d.amount) for d in case.deductions],
        "admission_date": h.admission_date,
        "discharge_date": h.discharge_date,
        "total_amount": round(case.claimed),
        "sum_insured": round(p.sum_insured),
        "copay_percent": round(p.copay_pct),
        "room_rent_limit": round(p.room_rent_limit) if p.room_rent_limit else None,
    }
    numbers, dates, low = numbers_in(clean_text), dates_in(clean_text), norm_text(clean_text)
    gold = {}
    for field, (kind, _) in FIELDS[doc_type].items():
        value = truth[field]
        if value is None:
            continue
        if doc_type == "policy_schedule" and field == "patient_name":
            continue  # unanswerable: a schedule lists members but never says who was later hospitalised
        visible = {
            "name": lambda v: norm_text(v) in low,
            "id": lambda v: re.sub(r"\s", "", v).lower() in re.sub(r"\s", "", low),
            "amount": lambda v: v in numbers or (field == "copay_percent" and v == 0),
            "amounts": lambda v: all(a in numbers for a in v),
            "date": lambda v: v in dates,
        }[kind](value)
        if visible:
            gold[field] = value
    return gold


# --- parsing helpers -------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def norm_text(s: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", str(s))  # "Krish Karan (Parent)" → "Krish Karan"
    s = re.sub(r"\b(mr|mrs|ms|smt|shri|sri|dr|master|baby|kumari)\.?\s+", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def numbers_in(text: str) -> set[int]:
    return {round(float(n.replace(",", ""))) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def to_amount(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(value or ""))
    return float(m.group().replace(",", "")) if m else None


def dates_in(text: str) -> set[date]:
    found = set()

    def add(y, m, d):
        y = int(y)
        y = y + 2000 if y < 100 else y
        try:
            found.add(date(y, int(m), int(d)))
        except ValueError:
            pass

    for y, m, d in re.findall(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text):
        add(y, m, d)
    for d, m, y in re.findall(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", text):
        add(y, m, d)  # Indian documents: day first
    mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
    for d, m, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]+{mon},?[\s\-]+(\d{{2,4}})\b", text, re.I):
        add(y, MONTHS[m.lower()[:3]], d)
    for m, d, y in re.findall(rf"\b{mon}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text, re.I):
        add(y, MONTHS[m.lower()[:3]], d)
    return found


def parse_json(raw: str) -> dict | None:
    """The first JSON object in a model's reply, or None."""
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M)
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                obj = json.loads(raw[start : i + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


# --- prompt and target -----------------------------------------------------------------


def prompt(doc_type: str, text: str) -> str:
    fields = "\n".join(f'- "{k}": {desc}' for k, (_, desc) in FIELDS[doc_type].items())
    return (
        f"Extract fields from this {doc_type.replace('_', ' ')}, a page from an Indian health-insurance claim file.\n\n"
        f"Return only one JSON object with exactly these keys:\n{fields}\n\n"
        "Write amounts as plain numbers (no ₹ symbol, no commas), dates as YYYY-MM-DD, and null for anything "
        "not on the page.\n\n"
        f"Page text:\n<<<\n{text}\n>>>"
    )


def target(doc_type: str, gold: dict) -> str:
    """The training answer: every key, gold value or null, dates as ISO strings."""
    out = {k: gold.get(k) for k in FIELDS[doc_type]}
    out = {k: v.isoformat() if isinstance(v, date) else v for k, v in out.items()}
    return json.dumps(out, ensure_ascii=False)


# --- scoring ---------------------------------------------------------------------------


def field_correct(kind: str, gold, pred) -> bool:
    if pred is None:
        return False
    if kind == "name":
        return norm_text(pred) == norm_text(gold)
    if kind == "id":
        return re.sub(r"\s", "", str(pred)).lower() == re.sub(r"\s", "", gold).lower()
    if kind == "amount":
        a = to_amount(pred)
        return a is not None and abs(a - gold) <= 1
    if kind == "date":
        return gold in dates_in(str(pred))
    if kind == "amounts":
        if not isinstance(pred, list):
            return False
        # ₹0 rows aren't deductions: letters often show a deduction column for every bill line
        got = sorted(a for a in (to_amount(x) for x in pred) if a is not None and abs(a) > 0.5)
        return len(got) == len(gold) and all(abs(a - b) <= 1 for a, b in zip(got, sorted(gold)))
    raise ValueError(kind)


def score_page(page: dict, raw: str) -> dict:
    pred = parse_json(raw)
    fields = {}
    for field, value in page["gold"].items():
        kind = FIELDS[page["doc_type"]][field][0]
        fields[field] = pred is not None and field_correct(kind, value, pred.get(field))
    return {"valid_json": pred is not None, "fields": fields, "page_correct": bool(fields) and all(fields.values())}


def reward(page: dict, raw: str) -> float:
    """The GRPO reward: share of gold fields correct, plus 0.1 for valid JSON. Range 0–1.1."""
    s = score_page(page, raw)
    return (sum(s["fields"].values()) / len(s["fields"]) if s["fields"] else 0.0) + 0.1 * s["valid_json"]


def summarise(scored: list[dict], pages_: list[dict]) -> dict:
    n = len(scored)
    by_field, by_type = {}, {}
    for s, p in zip(scored, pages_):
        for f, ok in s["fields"].items():
            by_field.setdefault(f, []).append(ok)
        by_type.setdefault(p["doc_type"], []).append(s["page_correct"])
    amount_fields = [ok for s, p in zip(scored, pages_) for f, ok in s["fields"].items()
                     if FIELDS[p["doc_type"]][f][0] in ("amount", "amounts")]
    all_fields = [ok for s in scored for ok in s["fields"].values()]
    return {
        "pages": n,
        "page_accuracy": sum(s["page_correct"] for s in scored) / n,
        "field_accuracy": sum(all_fields) / len(all_fields),
        "amount_accuracy": sum(amount_fields) / len(amount_fields) if amount_fields else None,
        "valid_json": sum(s["valid_json"] for s in scored) / n,
        "by_field": {f: sum(v) / len(v) for f, v in sorted(by_field.items())},
        "by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }


# --- predictions -----------------------------------------------------------------------


def predict_gemini(page_list, model: str) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    from google.genai import types

    import synth.net  # noqa: F401
    from synth.write_docs import client

    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))

    def ask(page):
        for attempt in range(4):
            try:
                return client().models.generate_content(model=model, contents=prompt(page["doc_type"], page["text"]), config=config).text or ""
            except Exception as e:  # rate limits: back off and retry
                if attempt == 3:
                    return f"ERROR {str(e)[:100]}"
                time.sleep(5 * (attempt + 1))

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(ask, page_list))


def load_qwen(size: str, adapter: list[Path] | Path | None, dtype: str = "float32"):
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.baselines import QWEN_SIZES

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    name = QWEN_SIZES[size]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device)
    # Adapters are merged in order, e.g. [SFT, GRPO]: GRPO's LoRA was trained on the merged SFT model.
    for path in [adapter] if isinstance(adapter, Path) else adapter or []:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return tok, model.eval()


def generate_batch(tok, model, prompts: list[str], max_new_tokens: int = 220, sample: bool = False, n: int = 1) -> list[str]:
    import torch

    chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=sample, num_return_sequences=n,
            temperature=1.0 if sample else None, top_p=1.0 if sample else None, top_k=0 if sample else None,
            pad_token_id=tok.pad_token_id,
        )
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def predict_qwen(page_list, size: str, adapter: list[Path] | None, batch: int = 8) -> list[str]:
    tok, model = load_qwen(size, adapter)
    replies, start = [], time.time()
    for i in range(0, len(page_list), batch):
        chunk = page_list[i : i + batch]
        replies += generate_batch(tok, model, [prompt(p["doc_type"], p["text"]) for p in chunk])
    print(f"  {(time.time() - start) / len(page_list):.2f}s/page on {model.device}")
    return replies


# --- CLI -------------------------------------------------------------------------------


def load_preds(path: Path) -> dict:
    return {(r["seed"], r["doc_type"]): r["raw"] for r in map(json.loads, path.open())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    pr = sub.add_parser("predict")
    pr.add_argument("model", choices=["gemini", "gemini-pro", "qwen"])
    pr.add_argument("--size", default="0.5B")
    pr.add_argument("--adapter", type=Path, nargs="+", help="one or more adapters, merged in order")
    pr.add_argument("--variant", default="clean", choices=["clean", "ocr"])
    pr.add_argument("--split", default="test")
    pr.add_argument("--name", help="output name (default derived from the options)")
    sc = sub.add_parser("score")
    sc.add_argument("preds", type=Path)
    sc.add_argument("--compare", type=Path)
    args = parser.parse_args()

    if args.cmd == "stats":
        for split in ("train", "val", "test"):
            ps = pages(split)
            counts = {t: sum(p["doc_type"] == t for p in ps) for t in DOC_TYPES}
            fields = sum(len(p["gold"]) for p in ps)
            print(f"{split:<6} {len(ps):>4} pages {fields:>5} gold fields  {counts}")
        return

    if args.cmd == "predict":
        page_list = pages(args.split, args.variant)
        if args.model == "qwen":
            replies = predict_qwen(page_list, args.size, args.adapter)
            default = f"qwen-{args.size}" + {0: "", 1: "-sft", 2: "-grpo"}[len(args.adapter or [])]
        else:
            replies = predict_gemini(page_list, "gemini-3.5-flash" if args.model == "gemini" else "gemini-3.1-pro-preview")
            default = args.model
        name = args.name or default
        out = Path("preds/f5") / f"{name}__{args.variant}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for p, raw in zip(page_list, replies):
                f.write(json.dumps({"seed": p["seed"], "doc_type": p["doc_type"], "raw": raw}, ensure_ascii=False) + "\n")
        print(f"wrote {out}")
        return

    if args.cmd == "score":
        variant = args.preds.stem.split("__")[-1]
        page_list = pages("test", variant if variant in ("clean", "ocr") else "clean")

        def run(path):
            preds = load_preds(path)
            scored = [score_page(p, preds.get((p["seed"], p["doc_type"]), "")) for p in page_list]
            return scored, summarise(scored, page_list)

        a_scored, a = run(args.preds)
        print(json.dumps(a, indent=1))
        if args.compare:
            from bench.score import mcnemar

            b_scored, b = run(args.compare)
            m = mcnemar({i: s["page_correct"] for i, s in enumerate(a_scored)}, {i: s["page_correct"] for i, s in enumerate(b_scored)})
            print(f"{args.compare.stem}: page accuracy {b['page_accuracy']:.1%}")
            print(f"McNemar on whole pages: only A right {m['only_a']}, only B right {m['only_b']}, p = {m['p_value']:.3f}")


if __name__ == "__main__":
    main()
