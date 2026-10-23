"""F2 baselines: predict the document type of every page in a split.

    uv run python -m bench.baselines majority
    uv run python -m bench.baselines keywords
    uv run python -m bench.baselines qwen-zeroshot        # label scoring, never invalid
    uv run python -m bench.baselines qwen-zeroshot-gen    # free generation, can ramble
    uv run python -m bench.baselines qwen-fewshot
    uv run python -m bench.baselines gemini

Each writes preds/<name>.jsonl for bench.score. Every model sees the same input: the full
page text (median ~880 tokens) and the same label definitions.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from bench.score import test_pages

SPLITS = json.loads(Path("bench/splits.json").read_text())

# Short definitions, from sprints/f2-doc-type/labels.md in the learning repo.
LABELS = {
    "settlement_letter": "insurer/TPA letter saying a claim was paid, fully or partly (amounts approved, deductions, UTR/NEFT reference)",
    "rejection_letter": "insurer/TPA letter repudiating a claim outright, ₹0 paid, citing a clause or exclusion",
    "preauth_denial": "denial of CASHLESS pre-authorisation, usually to the hospital; the patient can still claim reimbursement",
    "query_letter": "insurer/TPA letter asking for more documents or clarification before deciding; no decision yet",
    "policy_schedule": "the policy's schedule: policy number, period, insured members, sum insured, premium, limits",
    "hospital_bill": "a hospital bill, invoice or receipt with line items and amounts",
    "discharge_summary": "the hospital's clinical summary of the whole stay: diagnosis, course in hospital, discharge advice",
    "medical_report": "a single clinical report that is not a discharge summary: lab results, radiology, prescription",
    "health_card": "an insurer/TPA health e-card: member name, card ID, policy number, validity; no coverage details",
    "other": "anything else in a claim file: claim form, policy wording page, covering email, admission form, blank page",
}
LABEL_NAMES = sorted(LABELS)

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
GEMINI = "gemini-3.5-flash"


def pages_of(split: str) -> list[tuple[int, str, str]]:
    """(seed, true doc_type, page text) for every page in the split."""
    return [(seed, t, Path(r["text"]).read_text()) for (seed, t), r in sorted(test_pages(split).items())]


def prompt(text: str, examples: str = "") -> str:
    guide = "\n".join(f"- {name}: {desc}" for name, desc in LABELS.items())
    return (
        "Classify this page from an Indian health-insurance claim file into exactly one document type.\n\n"
        f"Document types:\n{guide}\n\n{examples}"
        f"Page text:\n<<<\n{text}\n>>>\n\n"
        "Answer with only the document type label."
    )


def normalise(answer: str) -> str:
    """Map a free-text answer to a label if it clearly is one; otherwise return it unchanged (invalid)."""
    first = (answer or "").strip().splitlines()[0] if (answer or "").strip() else ""
    cleaned = re.sub(r"[`*'\".:\s]+", " ", first).strip().lower().replace(" ", "_")
    return cleaned if cleaned in LABELS else first


# --- baselines that need no model ---------------------------------------------------


def majority(pages):
    train = test_pages("train")
    most_common = Counter(t for _, t in train).most_common(1)[0][0]
    return {(s, t): most_common for s, t, _ in pages}


KEYWORD_RULES = [  # checked in order; first match wins
    ("preauth_denial", ["pre-auth", "preauth", "pre auth", "cashless facility", "cashless request", "cashless authori"]),
    ("query_letter", ["deficiency", "documents required", "additional documents", "query letter", "requisition"]),
    ("rejection_letter", ["repudiat", "not payable under", "claim is not admissible", "regret to inform"]),
    ("settlement_letter", ["utr", "neft", "settlement", "amount approved", "approved amount"]),
    ("health_card", ["e-card", "health card", "ecard", "member id", "card no"]),
    ("discharge_summary", ["discharge summary", "course in hospital", "condition at discharge", "advice on discharge"]),
    ("medical_report", ["reference range", "impression", "laboratory", "radiology", "investigation report", "rx"]),
    ("hospital_bill", ["final bill", "bill no", "grand total", "invoice", "gst"]),
    ("policy_schedule", ["policy schedule", "period of insurance", "sum insured", "premium"]),
]


def keywords(pages):
    preds = {}
    for seed, t, text in pages:
        low = text.lower()
        preds[(seed, t)] = next(
            (label for label, words in KEYWORD_RULES if any(re.search(rf"\b{re.escape(w)}", low) for w in words)),
            "other",
        )
    return preds


# --- Qwen 0.5B on the Mac -----------------------------------------------------------


def load_qwen():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is cached; skip the network check
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).to(device).eval()
    return tok, model


def qwen_label_scores(tok, model, user_message: str) -> dict[str, float]:
    """log P(label | prompt) for every label.

    This is completion_logprob from M1 Exercise 5, applied to 10 candidate answers. The model
    can't produce an invalid label, because it only ever chooses among the 10. The page is
    read once and its KV cache reused for all 10 label endings (M1 Exercise 6): re-reading
    the page per label made each page ~10× slower on the Mac's GPU.
    """
    import torch

    chat = tok.apply_chat_template(
        [{"role": "user", "content": user_message}], add_generation_prompt=True, tokenize=False
    )
    prefix = torch.tensor([tok(chat, add_special_tokens=False)["input_ids"]], device=model.device)
    label_ids = [tok(name, add_special_tokens=False)["input_ids"] for name in LABEL_NAMES]
    longest = max(len(ids) for ids in label_ids)
    pad = tok.pad_token_id or 0

    with torch.no_grad():
        out = model(prefix, use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)  # predicts each label's first token
        cache = out.past_key_values
        cache.batch_repeat_interleave(len(LABEL_NAMES))  # one copy of the page's memory per label
        endings = torch.tensor([ids + [pad] * (longest - len(ids)) for ids in label_ids], device=model.device)
        rest = torch.log_softmax(model(endings, past_key_values=cache).logits.float(), -1)

    return {
        name: first[ids[0]].item() + sum(rest[i, j - 1, ids[j]].item() for j in range(1, len(ids)))
        for i, (name, ids) in enumerate(zip(LABEL_NAMES, label_ids))
    }


def few_shot_block() -> str:
    """One short example per type, taken from TRAINING pages only."""
    train = test_pages("train")
    lines = ["Examples (start of a page → its type):"]
    for name in LABEL_NAMES:
        seed, _ = next(k for k in sorted(train) if k[1] == name)
        snippet = re.sub(r"\s+", " ", Path(train[(seed, name)]["text"]).read_text())[:300]
        lines.append(f"[{name}] {snippet}…")
    return "\n".join(lines) + "\n\n"


def qwen(pages, mode: str):
    import torch

    tok, model = load_qwen()
    examples = few_shot_block() if mode == "fewshot" else ""
    preds, start = {}, time.time()
    for i, (seed, t, text) in enumerate(pages, 1):
        message = prompt(text, examples)
        if mode == "zeroshot-gen":
            chat = tok.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**chat, max_new_tokens=12, do_sample=False)
            preds[(seed, t)] = normalise(tok.decode(out[0, chat["input_ids"].shape[1]:], skip_special_tokens=True))
        else:
            scores = qwen_label_scores(tok, model, message)
            preds[(seed, t)] = max(scores, key=scores.get)
        if i % 20 == 0:
            print(f"  {i}/{len(pages)}  {(time.time() - start) / i:.2f}s/page")
    print(f"  {(time.time() - start) / len(pages):.2f}s/page on {model.device}")
    return preds


# --- Gemini Flash ---------------------------------------------------------------------


def gemini(pages):
    from concurrent.futures import ThreadPoolExecutor

    from google.genai import types

    import synth.net  # noqa: F401
    from synth.write_docs import client

    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))
    tokens = Counter()

    def ask(page):
        seed, t, text = page
        reply = client().models.generate_content(model=GEMINI, contents=prompt(text), config=config)
        u = reply.usage_metadata
        tokens["in"] += u.prompt_token_count or 0
        tokens["out"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        return (seed, t), normalise(reply.text)

    start = time.time()
    with ThreadPoolExecutor(8) as pool:
        preds = dict(pool.map(ask, pages))
    n = len(pages)
    print(f"  {time.time() - start:.0f}s total; per page ~{tokens['in'] // n} input + {tokens['out'] // n} output/thinking tokens")
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", choices=["majority", "keywords", "qwen-zeroshot", "qwen-zeroshot-gen", "qwen-fewshot", "gemini"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    pages = pages_of(args.split)
    run = {
        "majority": lambda: majority(pages),
        "keywords": lambda: keywords(pages),
        "qwen-zeroshot": lambda: qwen(pages, "zeroshot"),
        "qwen-zeroshot-gen": lambda: qwen(pages, "zeroshot-gen"),
        "qwen-fewshot": lambda: qwen(pages, "fewshot"),
        "gemini": lambda: gemini(pages),
    }[args.baseline]
    preds = run()

    out = Path("preds") / (f"{args.baseline}.jsonl" if args.split == "test" else f"{args.baseline}.{args.split}.jsonl")
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for (seed, t), p in sorted(preds.items()):
            f.write(json.dumps({"seed": seed, "doc_type": t, "predicted": p}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
