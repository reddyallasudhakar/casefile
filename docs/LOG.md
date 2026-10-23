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
