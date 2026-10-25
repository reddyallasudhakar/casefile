# Research log

A running record of what I'm trying to do, the questions I asked along the way, the decisions
I made and why, and what surprised me. Newest entries at the bottom. Written up from my
working sessions; the decisions and the reasoning are mine.

---

## 2026-10-22: Why this project

I ran Raksha at Gaido AI, where AI agents fought rejected and underpaid health-insurance
claims. The hard parts were never "reading the letter". They were knowing which clause
applies, whether a deduction is legitimate, and what should actually have been paid.

I want to go one level below the API: fine-tuning, post-training, distillation and RL, on a
problem I know deeply, in public. The central question I keep coming back to:

> **Can a 0.5B model running on a laptop match a frontier model on a narrow claims task,
> after minutes of training?**

If yes, that matters for products like FairClaims: claim files are sensitive health data,
so a small model that runs privately and costs almost nothing per page is a real product
argument, not just an exercise.

## 2026-10-22: Decisions on scope and openness

- **Name: Casefile.** A claim *is* a case file, and the data design is literally "one case,
  many documents". English name; it works for the repo, the models and the benchmark.
- **Public, on my personal GitHub.** For recruiters and for the companies I pitch B2B work
  to. It's credited to Policygaido/FairClaims in the README.
- **Public artefacts use synthetic data only.** Real customers' claim documents are their
  health data. Even with registration consent, they never go into a public dataset or
  model. At most, they become a private evaluation set, and only if the consent wording
  covers it.
- **Fictional insurer and TPA names** (Sahyog, Kavach, Medilink…). Generated rejection
  letters on real insurers' letterheads would be impersonation.
- **Train on GCP** (Policygaido credits), **evaluate and run on my M3 MacBook**. That keeps
  the "runs on a laptop" claim honest.

## 2026-10-22: Synthetic data design: cases first, documents second

Instead of generating loose pages, I generate a structured **claim case** first (patient,
policy, hospitalisation, bill, deductions, outcome), then have Gemini write every document
*from that case*. Why:

- The bill, discharge summary and settlement letter **agree with each other**, like a real file.
- **Labels are free and exact**: I know every true amount, date and policy number, so
  there's no hand-labelling.
- Nothing about real people can leak.

The schema (`synth/case.py`) enforces consistency rules: discharge after admission,
admission after policy inception, approved = claimed − deductions, a rejection pays ₹0 and
must have a reason.

## 2026-10-22: Does the 0.5B model even read images?

My question: the plan uses Qwen 0.5B, so can it read a photographed document?

**No.** Qwen2.5-0.5B is text-only. So the system has two separate jobs:

```
page image ──[ READING (OCR) ]──▶ text ──[ UNDERSTANDING (Qwen) ]──▶ label / fields
```

This led to a worry I wanted to face directly: *if ChatGPT reads documents better than
our pipeline, are we doomed?* My conclusion: reading is becoming a commodity that I can
measure and buy; the product is what happens after reading (rules, deductions, appeals,
follow-through). The design that follows is a **cascade**: a cheap private reader for most
pages, and a frontier model as the fallback for low-confidence ones.

## 2026-10-23: Apples to apples: separating reading from understanding

I realised that if Qwen's results depend on whichever OCR is in front of it, then comparing
"Qwen pipeline vs Gemini" mixes up two things. **Total error ≈ reading error +
understanding error.**

So I'm measuring them separately:

| Comparison | Input | What it isolates |
|---|---|---|
| Model vs model | the same **perfect text** for both | pure understanding |
| Model vs model, realistic | the same **OCR output** for both | robustness to OCR noise |
| System vs system | OCR + Qwen vs Gemini reading the **image** | what a user experiences |

**Reordered plan:** F2 (document type) on clean text first → F0 (pick the OCR) → rerun F2
on OCR text. The drop between the first and last is exactly what reading costs.

## 2026-10-23: Baselines before fine-tuning

A fine-tuned score means nothing without a "before". 92% is impressive from 30% and
trivial from 90%. So the order is: freeze the test set → measure every baseline → fine-tune
→ measure on the same test set. The baseline ladder:

1. Always guess the most common type (the floor)
2. **Keyword rules.** Many pages name their type in the title, so this may score high, and
   I need to know that before celebrating a fine-tune
3. Qwen 0.5B zero-shot (score each label's likelihood rather than free generation)
4. Qwen 0.5B few-shot
5. Gemini Flash (the target)

## 2026-10-23: Splitting the data

**Split by case, not by page.** Pages from one case share the patient, hospital and amounts.
A random page split would let the model "recognise the case" instead of learning the
document type. Cases are shuffled with a fixed seed (reproducible), 70/15/15, and a draw is
rejected if any document type has fewer than 8 pages in validation or test. The first draw
passed: 569 / 107 / 124 pages.

## 2026-10-23: Success bar, set before any results

> **The fine-tuned Qwen 0.5B matches or beats Gemini Flash on the test set.**

Written down and committed *before* running anything, so I can't move the goalposts. With
124 test pages, one page is 0.8 points, so "matches" is checked with a **McNemar test**: it
looks only at pages where the two models disagree, and asks whether the split is too
lopsided to be chance. A lower-but-not-significant score is reported as "indistinguishable",
not as a pass. Details in [results/f2.md](../results/f2.md).

## 2026-10-23: One judge for every model

`bench/score.py` is the single definition of "correct" for F2. Every model writes its
answers to a predictions file, and this one script grades all of them, so none is scored
more leniently than another. Choices I made:

- **A missing or malformed answer counts as wrong**, and is reported separately as the
  *invalid-output rate*. Small models often answer "This appears to be a letter…" instead
  of a label; hiding that would flatter them.
- **Accuracy comes with a 95% interval** (Wilson). At 124 pages, 88% really means "somewhere
  around 81–93%", and I want that uncertainty visible in every row.
- **Macro-F1 alongside accuracy**, so a model can't look good by mastering the easy types.
- **The confusion matrix** shows *which* types get mixed up, e.g. rejection letter vs
  pre-auth denial, the distinction the label guide cares most about.
- **McNemar's exact test** for comparing two models.

I was confused at first about how McNemar differs from F1, since both involve 2×2 tables.
The difference: a confusion matrix compares **one model with the truth** (and gives it a
grade); McNemar's table compares **two models with each other** on the same pages (right
vs wrong for each). McNemar isn't a grade, it's a test of whether a gap between grades is
real. Like two students who sat the same exam: only the questions where exactly one of them
was right say anything about who is better.

Checked on planted predictions: 15 planted errors gave exactly 109/124; a perfect file
scores 100%; the McNemar p-values match hand calculations (9 vs 2 → 0.065, 10 vs 1 → 0.012).

## 2026-10-23: First baselines, and a problem with the test set

The "before" numbers on the 124 test pages, given clean page text:

| Model | Accuracy |
|---|---|
| Always the most common type | 8.9% |
| Keyword rules (written once, from the label guide, not tuned) | 54.8% |
| Qwen 0.5B zero-shot (label scoring) | 27.4% |
| Gemini Flash zero-shot | **100%** |

Untrained, Qwen 0.5B **collapses onto two labels**: nearly everything is called
`policy_schedule` or `health_card`. It loses even to keyword rules.

Gemini's 100% worries me more than it impresses me. **Gemini wrote these documents.** It
generated each page from a description of its type, and many pages name their type in the
title. Asking Gemini to classify them is close to marking its own homework. It also turns
my success bar ("match or beat Gemini") into "reach 100%". I need a harder, less circular
test before trusting this comparison: OCR'd text from the degraded images, documents
written by a different model, or real pages.

## 2026-10-23: Model work moves off my laptop

The Qwen few-shot baseline slowed my Mac badly, and got slower as it ran (16 → 50 s per
page as the machine throttled). From now on, all model work runs on GCP. I compared the
options before spending anything ([COSTS.md](COSTS.md)): an **L4 Spot VM with a 4-hour
auto-stop**, at ~$0.25–0.42/hour. F2 should cost $1–2 in GPU time; the biggest real risk
to the budget is a forgotten VM, so the auto-stop matters more than the hourly rate.

## 2026-10-23: The full "before" table, and the first GPU VM

The last two Qwen baselines ran on the new L4 Spot VM in under a minute each: 0.4 s per page,
against up to 50 s per page on my Mac. The VM was stopped straight after.

- **Few-shot didn't help** (25.8% vs 27.4% zero-shot): one short example per type isn't
  enough for a 0.5B model.
- **The most telling detail:** when allowed to answer freely, Qwen's "invalid" answers were
  semantically right in the wrong format: `claim_repudiation_letter` for a rejection letter,
  `claim_query_letter` for a query letter. So the small model partly understands the pages;
  it just doesn't know my label set. That's precisely what fine-tuning teaches, which makes
  me optimistic about the jump from ~27%.

Full table: [results/f2.md](../results/f2.md).

## 2026-10-23: Why own a model at all, when APIs are this cheap?

I asked how API providers can sell inference so cheaply, and whether self-hosting ever pays.
What I learned:

- **Providers batch many customers onto each GPU.** Generating a token streams all of the
  model's weights from memory once, and that one pass can produce the next token for ~100
  conversations nearly as cheaply as for one. Our own L4 serving one request at a time
  works out to ~$6.50 per million tokens; the same GPU batched across ~50 conversations,
  ~$0.13. Add always-busy GPUs, prefix caching, 8-bit weights and bulk hardware deals.
- **Fine-tuning a small model is cheap** (F2 will cost under $2). The costly part of owning a
  model is **keeping a GPU running for serving**, plus the engineering time to maintain it.
- **On cost alone, an API wins at FairClaims' current volume** (~10k pages a month ≈ $6–30 on
  Gemini Flash). A always-on L4 (~$500/month) only breaks even at hundreds of thousands of
  pages a month.

**So why do this?** For FairClaims, the reason is **privacy and compliance, not cost**. Claim
files are health data under India's DPDP Act, and insurers, TPAs and brokers increasingly
want that data to stay inside their own cloud. A small model we own can run inside a client's
VPC, or even on-device; an API call can't. Casefile is the evidence that a small model is
good enough to make that offer.

How I'll serve it when the time comes: a scale-to-zero GPU service or a CPU for a tiny model
(near-zero fixed cost), several task-specific LoRA adapters on one base model so a single
server handles all tasks, and a cascade that sends uncertain pages to a bigger model.

## 2026-10-23: Hardening the test without making new pages

I decided not to generate new pages from a different writer model. Instead, the hard
versions of the test come from what we already have:

1. **OCR text:** each page's degraded phone or scan image, read by Google Cloud Vision.
2. **Hidden titles:** type-revealing headings masked.
3. **A different frontier reader:** Gemini 3.1 Pro joins as the frontier baseline, so the
   frontier score isn't only Gemini 3.5 Flash grading pages it wrote itself. Pro will likely
   hit 100% on clean text too; the hard variants are where models should separate.

**Known limitation I'm accepting for now:** training and test pages share one writer
(Gemini 3.5 Flash), so a fine-tuned model could partly learn that writer's style rather than
the document types. The real test of that is real pages, later.

**OCR results** (all 800 pages, ~6 minutes, free within Cloud Vision's monthly 1,000-page tier).
Share of key facts that survive OCR, counting only facts present in the clean text:

| Fact | Phone photos | Scans |
|---|---|---|
| Patient name | 99.7% | 100% |
| Policy number | 99.1% | 100% |
| Claimed amount | 98.8% | 98.7% |
| Approved amount | 97.4% | 97.6% |

Much better than I expected. But the misses are telling: one policy number came back with
**a 6 read as an 8** (`626455` → `826455`), a plausible-looking single-digit error, which is
exactly the kind that silently breaks a claim. Amounts are the weakest field (~2–3% lost),
probably split across lines. For FairClaims, that argues for **always double-checking amounts
and IDs** (cross-checking bill totals against line items, or asking the user to confirm) rather
than trusting any single read.

## 2026-10-23: Hiding the titles

`bench/mask_titles.py` replaces heading-like lines that name a type ("CLAIM REPUDIATION
LETTER", "FINAL INPATIENT BILL", Hindi headings too) with `[heading removed]`. Only short
lines near the top of the page count as headings; a body sentence like "your claim has been
repudiated" is content the model should understand, so it stays. That gives four test
variants per page: **clean, masked, OCR, OCR + masked**. Headings were masked on 755/800 clean
pages and 669/800 OCR pages (OCR breaks some headings so they no longer match).

First surprise: hiding titles barely moves the keyword rules (54.8% → 52.4% clean, 51.6% for
OCR + masked). They key on body words, not headings. Whether the *models* were leaning on
titles is what the next baseline round will show.

## 2026-10-23: The hard test, and what model size buys

Ran every baseline on all four variants (clean, masked, OCR, OCR + masked), plus Qwen at
0.5B, 1.5B and 3B, in one VM session to avoid paying for start-up twice.

| Zero-shot accuracy | Clean | OCR + masked |
|---|---|---|
| Qwen 0.5B | 27% | 19% |
| Qwen 1.5B | 72% | 51% |
| Qwen 3B | 93% | 80% |
| Gemini 3.5 Flash / 3.1 Pro | 100% | 100% |

What I take from it:

- **Gemini stays at 100% even with OCR noise and no titles.** So my circularity worry was
  only part of the story: this task really is easy for frontier models. My success bar
  becomes "reach 100%, or be statistically indistinguishable from it".
- **The small models were leaning on the titles.** Hiding headings costs Qwen 3B 13 points.
  The hard variant was worth building: it shows *how* small models are cheating.
- **Size is a big lever before any training** (27% → 72% → 93%). The interesting question for
  fine-tuning is now: **how much of that gap does training close, at each size?** If
  fine-tuning lifts 0.5B or 1.5B to near 100%, the smaller model wins on speed and cost.
- **The catch-all `other` class is the hardest.** Qwen 3B never predicted it once. Worth
  watching after fine-tuning.

## 2026-10-23: How I'm fine-tuning

`bench/finetune.py`, written before training anything, so the design is fixed in advance:

1. **Examples:** each training page becomes a chat pair: the *same prompt the baselines use*
   (label definitions + page) → the correct label + Qwen's end-of-turn token. Every page appears
   in all four variants (clean, masked, OCR, OCR + masked): 2,276 examples, ~2.5M tokens.
   Showing each page with and without its heading is aimed squarely at the title shortcut
   the zero-shot runs exposed.
2. **Loss only on the answer:** prompt tokens are masked with -100, so the model is trained
   on `discharge_summary<|im_end|>`, not on reproducing page text. It's M1's
   `completion_logprob`, turned into a training signal.
3. **LoRA, rank 16,** on every attention and MLP projection (`q/k/v/o_proj`,
   `gate/up/down_proj`, the layers I looked at in Module 1). The base weights stay frozen;
   ~0.5–1% of parameters train.
4. **One epoch** (each page is already seen 4 times), lr 2e-4 with warmup and cosine decay,
   effective batch 16 via gradient accumulation, bf16 on the L4, gradient checkpointing for
   the bigger models.
5. **Validation** on the 107 val pages, hardest variant; **test pages untouched** until the
   final run, which uses the exact same label-scoring code as the zero-shot baselines.

Same recipe for 0.5B, 1.5B and 3B, so the only thing that changes is model size.

## 2026-10-24: Fine-tuning results: the smallest model is enough

| OCR + titles hidden (the hardest variant) | Zero-shot | Fine-tuned |
|---|---|---|
| Qwen 0.5B | 18.5% | **98.4%** |
| Qwen 1.5B | 50.8% | 99.2% |
| Qwen 3B | 79.8% | 99.2% |
| Gemini 3.1 Pro | 100% | — |

On clean text, all three fine-tuned sizes score **124/124, matching Gemini exactly: my success
bar is met, by the 0.5B.** On the noisy variants, every size is within 0–2 pages of Gemini, and
McNemar says none of the gaps is real (all p ≥ 0.5).

What I take from it:

- **Fine-tuning closed the gap that size couldn't.** Zero-shot, going from 0.5B to 3B was
  worth +60 points. After fine-tuning, size is worth nothing measurable. So for a narrow task
  the question isn't "how big a model?" but "have I trained it?".
- **The sweet spot is 0.5B:** 8 minutes to train, 0.2 s per page on one GPU, a 34 MB adapter,
  and small enough to run on a laptop or in a client's own cloud. That's the DPDP story in numbers.
- **Training on all four variants worked.** Zero-shot, the small models leaned on titles;
  fine-tuned, they score the same with titles hidden.
- **The remaining errors are exactly the near-neighbour pairs** from my label guide: medical
  report vs discharge summary for 0.5B, and **rejection letter vs pre-auth denial** for the
  bigger models. That second one matters most for FairClaims, because a cashless denial isn't
  a claim rejection and needs a different next step.
- **The loss went almost to zero** (≈0.0005), and validation stayed at 98–99%, so the task is
  very learnable from this data. Next, the honest test: real pages, which weren't written by
  the same model as the training data.

**Cost:** about $1 of GPU for all three fine-tunes plus evaluations.

**Operational lessons:** Google reclaimed the Spot VM partway through the 1.5B run
(preemption), and the first restart hit a "stockout" (no L4s free in the zone); a retry
worked. I switched to on-demand for that session (~$0.70/h) and back to Spot afterwards. Long
jobs now run detached on the VM (`nohup`), so a dropped SSH connection doesn't kill them.

## 2026-10-24: Was fine-tuning actually necessary? Reading the lift

My question after the results: if 3B already got 92.7% zero-shot on clean text, couldn't I
have skipped fine-tuning and just used a bigger model? The lift table answers it:

| Hardest variant (OCR + titles hidden) | Zero-shot | Fine-tuned | Lift |
|---|---|---|---|
| 0.5B | 18.5% | 98.4% | +79.8 |
| 1.5B | 50.8% | 99.2% | +48.4 |
| 3B | 79.8% | 99.2% | +19.4 |

My reasoning:

1. **Clean-text accuracy flatters untrained models.** 3B's 92.7% becomes 79.8% on realistic
   input, one page in five wrong. A misrouted page in FairClaims (a rejection treated as a
   cashless denial) sends the user down the wrong path, so that's not shippable.
2. **Fine-tuning beats size.** A fine-tuned 0.5B beats an untrained 3B by ~19 points on
   realistic input, at a sixth of the size. Given the choice between "bigger model" and
   "trained model", train.
3. **The lift is biggest where the input is hardest**, for every size. Fine-tuning mostly
   bought robustness (reading content without the heading, through OCR noise), which is
   exactly what real uploads need.
4. **The lift shrinks with size** (+80 → +48 → +19), because bigger models start further
   ahead, but all three end in the same place (98–99%). Starting point differs; finish doesn't.
5. **What I haven't ruled out:** prompting a bigger model *with examples* (few-shot) might
   close part of the gap without training. Few-shot didn't help 0.5B, but larger models use
   examples better. The rigorous version of "did I need fine-tuning?" compares fine-tuned 0.5B
   with the *best-prompted* 3B, not just zero-shot.

## 2026-10-24: Closing the question: could prompting have replaced fine-tuning?

I ran few-shot (one example per type in the prompt) for all three sizes on all four variants,
so the "before" numbers are the *best* prompting, not just zero-shot.

| OCR + titles hidden | Zero-shot | Few-shot | Fine-tuned |
|---|---|---|---|
| Qwen 0.5B | 18.5% | 18.5% | 98.4% |
| Qwen 1.5B | 50.8% | 57.3% | 99.2% |
| Qwen 3B | 79.8% | 79.8% | 99.2% |

**Answer: no.** Examples in the prompt add at most ~7 points (1.5B) and nothing for 3B on the
realistic variant. The best-prompted 3B is still at 79.8%, one page in five wrong. Fine-tuning
was necessary, and it made the *smallest* model the best choice.

Published the full grid, a chart and the method as a results page:
[results/f2.md](../results/f2.md).

## 2026-10-24: Planning the rules engine, and what my own backend actually computes

I planned to port the claim logic from my TypeScript backend. A read-only review changed the
plan: **most of the money rules aren't code at all.** Room-rent proportionate deduction, ICU
limits, deductibles, sub-limits and non-payable items exist only as instructions in LLM
prompts. A model decides what's fair; code just does the final arithmetic (co-pay, sum-insured
cap). The deterministic parts are the waiting-period and moratorium date maths, and the
settlement arithmetic.

So the engine is **new work, with the backend as a reference**: its conventions, its synthetic
test cases, and a list of inconsistencies not to copy (three different "continuous cover start"
rules, the 30-day wait measured from the renewal date, moratorium months computed two ways,
substring matching on "ped" that also matches "orthopedic").

It's also a product insight: today, FairClaims' most consequential numbers come from model
judgement. A tested, deterministic engine underneath would make them auditable, which
matters when you're telling someone their insurer owes them ₹67,500.

Plan: [RULES_ENGINE.md](RULES_ENGINE.md). Eligibility → payable (six fixed steps, every rupee
traceable to a rule) → auditing each insurer deduction as valid / excessive / invalid /
needs info. Golden cases in JSON so a TypeScript version can be checked against the same cases.

## 2026-10-24: F5 starts, and a scoring fix I'm disclosing

F5 (extraction) is running in auto mode at my request: Claude sets up and runs each step and
commits along the way. The success bar was committed before any model ran (`330706b`).

**A correction after the first Gemini run, before any Qwen run was scored.** Gemini 3.1 Pro
initially scored only ~90% of pages fully correct. Looking at its "errors", most were mine:

- **An unanswerable field.** I asked the policy schedule for "the insured member who was
  hospitalised", but a schedule lists members and never says who was later hospitalised.
  Gemini rightly answered null. That field is no longer scored.
- **₹0 deductions.** Settlement letters often show a deduction column for every bill line,
  mostly ₹0, so listing `[0, 0, 0, 17588]` is a fair reading of "each deduction". Zeros are
  now ignored when scoring.
- **Names with annotations**, e.g. "Krish Karan (Parent)". Parentheticals are now ignored.

The fixes apply identically to every model and were made before any Qwen result was looked at.
After them, Gemini: 98–99% of pages fully correct on clean text, 96.9% on OCR text. The
remaining misses are real reading errors (a date off by one day, amounts on noisy OCR).

Lesson: **an eval's first job is to be checked against a strong model.** If the strongest
model "fails", read the failures before believing them; several were my bugs.

## 2026-10-24: F5 results: reading precisely, and my first RL run

**The headline is mixed, and that's the honest result.**

- **Clean text: small models read as precisely as Gemini.** Fine-tuned Qwen 1.5B and 3B get
  98% of pages fully right against Gemini 3.1 Pro's 99%.
- **OCR text: they don't.** Best fine-tuned model 87.8% against Gemini's 96.9%, a real gap
  (p = 0.004). Success bar 1 not met. F2 (recognising a document) came easily; reading exact
  numbers through OCR noise is where a small model still pays for its size.
- **GRPO didn't help** (86.7% vs 85.7% on OCR, not significant). Success bar 2 not met.

What I learned from the RL run, which I think matters more than the number:

1. **GRPO needs disagreement to learn.** It compares a model's own answers to each other, so on
   pages where all 8 attempts agree, there's nothing to learn. My SFT model was already
   consistent; only ~2 of 8 pages per step carried signal. In the smoke test, on SFT's own
   training pages, it was 0 of 2, which is why I generated 200 RL-only pages.
2. **Reward up, validation down** is the signature of learning the training pages rather than
   the skill. With 200 pages seen ~3 times each, the model memorised.
3. **Some errors can't be learned:** when OCR has destroyed the digit, no policy recovers it.
4. **No reward hacking**, which I checked for explicitly: no invented room-rent limits, no
   length drift.

Engineering notes: the first GRPO attempt ran out of GPU memory computing vocabulary scores
for every position of long pages; scoring only the answer positions fixed it. The VM's 4-hour
auto-stop would have cut the run short, so I restarted it to reset the clock first.

**And an important process catch:** the F2 results commit had never actually reached GitHub.
The public repo was still showing pre-fine-tuning numbers. Fixed (`74b7d2f`); I now check the
remote after pushing instead of assuming.

Next for this line of work: more RL pages and harder-page selection, and the simpler baseline
of SFT on more OCR examples, before concluding anything about RL in general.

## 2026-10-25: F5b: closing the OCR gap; supervised variety beat RL

I asked for a follow-up aimed at F5's weak spot (OCR text, 86% vs Gemini's 97%): more OCR
data for SFT, and a better-designed GRPO run. Success bar pre-registered first (`f0d92cd`).

**OCR augmentation, the cheap version of "more data".** Instead of generating new pages with
Gemini (~$10+), each training and RL page's clean image was degraded 3 more ways and OCR'd
(~$3). The model then sees every page through 4 different kinds of OCR noise. Result on 1.5B:
**OCR 85.7% → 90.8%**, gap to Gemini halved (11 → 6 points). Not yet statistically conclusive
(6 vs 1 pages, p = 0.125), and clean text dipped 3 points, probably because the training mix
became 4 OCR copies per clean one.

**GRPO v2, redesigned around F5's failure:** 1,000 RL prompts instead of 400, hard-page mining
(keep only prompts where the model's 8 answers disagree: 444 kept), and keeping the best
validation checkpoint instead of the last. Mining did what it was meant to: ~5 of 8 groups
informative early on, against ~2 before. **But the outcome repeated F5:** validation peaked
at +1 page by step 10, then fell while training reward rose. On test: no difference.

My takeaway: **for reading through OCR noise, varied supervised examples were the lever; RL
wasn't.** A strong SFT model leaves few *consistent* mistakes for RL to fix, much of the
remaining error is unrecoverable from the OCR text, and a reward over 4–5 fields per page is
a blunt signal. RL with verifiable rewards shines where the model must *reason* to the answer;
copying digits off a noisy page isn't that.

One process slip: the first launch failed in seconds because `--tag -aug` reads as two flags;
`--tag=-aug` fixed it. Cheap to catch only because the pipeline fails fast.

## Surprises along the way

- **My network hands out IPv6 addresses but doesn't route IPv6.** Python tries IPv6 first
  and hangs for minutes; browsers race both and never notice. This one problem explained
  every mysterious hang (the Hugging Face downloads, 11-minute API errors). `synth/net.py`
  makes the project prefer IPv4: one document went from 399s to 24s.
- **The newest model isn't always usable.** `gemini-3.8-flash` was listed but queued for
  minutes per call; `gemini-3.5-flash` answers in seconds. Generation uses 3.5 with low
  thinking (writing from given facts needs little reasoning).
- **Reading generated data by eye catches real bugs.** The pilot had a 2-year-old getting a
  total knee replacement on a "Senior Citizen Care" policy. Conditions and products are now
  age-appropriate. Never train on data nobody has looked at.
- **Personal Google logins expire mid-run** (Workspace re-authentication). Long jobs belong
  on a VM with a service account.
