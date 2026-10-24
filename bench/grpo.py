"""GRPO on top of the F5 SFT model: reward = share of gold fields extracted correctly.

    uv run python -m bench.grpo --size 0.5B --steps 120

A minimal, readable GRPO (see docs/RL_PRIMER.md) rather than a library trainer:
1. Sample a batch of training pages; for each, generate a *group* of G answers at temperature 1.
2. Score every answer with the verifiable reward from bench/extract.py (fields correct + JSON).
3. Advantage = (reward − group mean) / group std. Answers better than their siblings get
   pushed up, worse ones pushed down. Groups where every answer scored the same carry no
   signal and are skipped.
4. One gradient step on −advantage × mean log-probability of each answer's tokens. Each batch
   is used once (on-policy), so PPO's ratio clipping is a no-op and is omitted; no KL penalty
   (as in Dr. GRPO / DAPO), with a small learning rate instead.
The SFT adapter is merged into the base model first, and GRPO trains a fresh LoRA on top.
Every `--eval-every` steps it scores the validation pages (OCR, greedy) and saves the adapter.
Run on the GCP GPU VM.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from bench import extract
from bench.baselines import QWEN_SIZES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="0.5B", choices=list(QWEN_SIZES))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--prompts", type=int, default=8, help="pages per step")
    parser.add_argument("--group", type=int, default=8, help="answers sampled per page")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--micro", type=int, default=2, help="sequences per backward pass")
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = QWEN_SIZES[args.size]
    sft = Path("adapters") / f"f5-qwen-{args.size}"
    out = Path("adapters") / f"f5-grpo-qwen-{args.size}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda")
    base = PeftModel.from_pretrained(base, sft).merge_and_unload()  # start from the SFT model
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.enable_input_require_grads()  # lets gradient checkpointing work with a frozen base
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    # Pages the SFT model never saw: on its own training pages it's already near-perfect, so its
    # sampled answers agree and GRPO gets no signal (seen in the smoke test: 0 of 2 groups informative).
    train = extract.pages("rl", "ocr") + extract.pages("rl", "clean")
    val = extract.pages("val", "ocr")
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def evaluate() -> float:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        right = 0
        for i in range(0, len(val), 8):
            chunk = val[i : i + 8]
            replies = extract.generate_batch(tok, model, [extract.prompt(p["doc_type"], p["text"]) for p in chunk], args.max_new)
            right += sum(extract.score_page(p, r)["page_correct"] for p, r in zip(chunk, replies))
        return right / len(val)

    log = [{"step": 0, "val_page_accuracy": evaluate()}]
    print(f"step 0: val page accuracy (OCR) {log[0]['val_page_accuracy']:.1%}", flush=True)
    start = time.time()

    for step in range(1, args.steps + 1):
        batch = random.sample(train, args.prompts)
        prompts = [extract.prompt(p["doc_type"], p["text"]) for p in batch]
        chats = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) for q in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

        # 1. sample G answers per page (no checkpointing: generation needs the KV cache)
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        with torch.no_grad():
            seqs = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new,
                num_return_sequences=args.group, pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        completions = seqs[:, prompt_len:]
        texts = tok.batch_decode(completions, skip_special_tokens=True)

        # 2–3. reward and group-relative advantage
        rewards = [extract.reward(batch[i // args.group], t) for i, t in enumerate(texts)]
        advantages, informative = [], 0
        for g in range(args.prompts):
            r = rewards[g * args.group : (g + 1) * args.group]
            mean, std = statistics.mean(r), statistics.pstdev(r)
            informative += std > 1e-6
            advantages += [(x - mean) / (std + 1e-4) if std > 1e-6 else 0.0 for x in r]

        # 4. policy-gradient step on informative samples (checkpointing keeps activations small)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        optimizer.zero_grad()
        idx = [i for i, a in enumerate(advantages) if a != 0.0]
        for j in range(0, len(idx), args.micro):
            part = idx[j : j + args.micro]
            ids = seqs[part]
            attn = (ids != tok.pad_token_id).long()
            attn[:, :prompt_len] = enc["attention_mask"].repeat_interleave(args.group, 0)[part]
            comp = ids[:, prompt_len:]
            # completion tokens up to and including the first end-of-turn
            mask = torch.ones_like(comp, dtype=torch.float)
            ended = (comp == eos).float().cumsum(1)
            mask = ((ended - (comp == eos).float()) == 0).float() * (comp != tok.pad_token_id).float()
            # score the vocabulary only at the answer positions: full-length logits don't fit in memory
            n_comp = comp.shape[1]
            logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=n_comp + 1).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            per_seq = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
            adv = torch.tensor([advantages[i] for i in part], device="cuda")
            loss = -(adv * per_seq).sum() / max(1, len(idx))
            loss.backward()
        if idx:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        page_ok = sum(extract.score_page(batch[i // args.group], t)["page_correct"] for i, t in enumerate(texts)) / len(texts)
        lengths = [len(t) for t in texts]
        entry = {"step": step, "reward": statistics.mean(rewards), "sample_page_accuracy": page_ok,
                 "informative_groups": informative, "mean_chars": statistics.mean(lengths)}
        if step % args.eval_every == 0 or step == args.steps:
            entry["val_page_accuracy"] = evaluate()
            model.save_pretrained(out)
        log.append(entry)
        extra = f"  val {entry['val_page_accuracy']:.1%}" if "val_page_accuracy" in entry else ""
        print(f"step {step}/{args.steps}  reward {entry['reward']:.3f}  sample-acc {page_ok:.1%}  "
              f"informative {informative}/{args.prompts}  chars {entry['mean_chars']:.0f}  ({time.time() - start:.0f}s){extra}", flush=True)
        (out / "grpo_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=1))

    # keep a sample of answers for the reward-hacking check
    (out / "samples.json").write_text(json.dumps([{"doc_type": batch[i // args.group]["doc_type"], "reward": rewards[i], "text": t}
                                                   for i, t in enumerate(texts[:16])], indent=1, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
