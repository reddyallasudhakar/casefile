"""Fine-tune Qwen with LoRA to classify claim pages into the 10 document types.

    uv run python -m bench.finetune --size 0.5B
    uv run python -m bench.finetune --size 3B

Steps (mirroring docs/LOG.md):
1. Each training page becomes a chat example: the same prompt the baselines use (label
   definitions + page text) → the correct label + end-of-turn. Every page appears in all four
   text variants (clean, masked, OCR, OCR + masked), so the model can't lean on headings.
2. Loss only on the answer tokens: prompt tokens get label -100 (M1 Exercise 5).
3. LoRA: the base weights stay frozen; small rank-r matrices on every attention and MLP
   projection are trained (~0.5–1% of parameters).
4. One epoch, AdamW, lr 2e-4 with warmup + cosine decay, gradient accumulation to an
   effective batch of 16, bf16 on the GPU.
5. Validation accuracy (label scoring, hardest variant) after training. Test pages are never touched.
6. Saves the adapter and a training log to adapters/qwen-<size>/.
Run on the GCP GPU VM, not a laptop.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from bench.baselines import LABEL_NAMES, QWEN_SIZES, VARIANTS, pages_of, prompt, qwen_label_scores


def build_examples(tok, split: str, variants: list[str], max_len: int):
    """Tokenised (input_ids, labels) pairs; labels are -100 except on the answer."""
    examples, skipped = [], 0
    for variant in variants:
        for _, label, text in pages_of(split, variant):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt(text)}], add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tok(chat, add_special_tokens=False)["input_ids"]
            answer_ids = tok(label, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
            if len(prompt_ids) + len(answer_ids) > max_len:
                skipped += 1
                continue
            examples.append((prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids))
    return examples, skipped


def val_accuracy(tok, model, variant: str = "ocr-masked") -> float:
    pages = pages_of("val", variant)
    right = 0
    for _, label, text in pages:
        scores = qwen_label_scores(tok, model, prompt(text))
        right += max(scores, key=scores.get) == label
    return right / len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="0.5B", choices=list(QWEN_SIZES))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = QWEN_SIZES[args.size]
    out = Path("adapters") / f"qwen-{args.size}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(name)
    examples, skipped = build_examples(tok, "train", args.variants, args.max_len)
    print(f"{len(examples)} training examples ({skipped} too long, skipped)")

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda")
    if args.size != "0.5B":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for checkpointing with frozen base weights
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    steps = math.ceil(len(examples) * args.epochs / args.accum)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.05 * steps)), steps)

    model.train()
    log, running, start, step = [], 0.0, time.time(), 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i, (ids, labels) in enumerate(examples, 1):
            batch = torch.tensor([ids], device="cuda")
            loss = model(input_ids=batch, labels=torch.tensor([labels], device="cuda")).loss
            (loss / args.accum).backward()
            running += loss.item()
            if i % args.accum == 0 or i == len(examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0 or step == steps:
                    avg = running / args.accum / 10 if step % 10 == 0 else running / args.accum
                    log.append({"step": step, "loss": avg, "lr": schedule.get_last_lr()[0]})
                    print(f"  step {step}/{steps}  loss {avg:.4f}  ({time.time() - start:.0f}s)")
                    running = 0.0
    train_seconds = time.time() - start

    model.eval()
    val = val_accuracy(tok, model)
    print(f"validation accuracy (OCR + masked, 107 pages): {val:.1%}")

    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({
        "base_model": name, "examples": len(examples), "variants": args.variants, "epochs": args.epochs,
        "lr": args.lr, "rank": args.rank, "accum": args.accum, "trainable_params": trainable,
        "train_seconds": round(train_seconds), "val_accuracy_ocr_masked": val, "loss_curve": log,
    }, indent=1))
    print(f"saved adapter to {out}  (training took {train_seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
