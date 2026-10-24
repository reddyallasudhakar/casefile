# Casefile

Open models and a benchmark for Indian health-insurance claims.

Built by Sudhakar Reddy, founder of [Policygaido](https://policygaido.com). Casefile powers
FairClaims, which helps people challenge rejected and underpaid health-insurance claims.

## The question

**Can a 0.5B model running on a laptop match a frontier model on a narrow claims task,
after minutes of training?** Claim files are sensitive health data, so a small model that
runs privately and costs almost nothing per page is a real product argument.

Casefile answers it one task at a time, each with a frozen test set, a ladder of baselines
and a success bar written down before any results. The reasoning behind every decision is
in the **[research log](docs/LOG.md)**.

## Latest result

**A fine-tuned 0.5B model matches Gemini at sorting claim documents:** 100% on clean text and
98.4% on OCR'd pages with headings hidden, against 79.8% for the best *prompted* 3B model.
8 minutes of training on one GPU. **[Full results →](results/f2.md)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/f2_chart_dark.png">
  <img alt="Zero-shot, few-shot and fine-tuned accuracy for Qwen 0.5B, 1.5B and 3B: fine-tuned models reach 98–100%, prompted models 19–93%, Gemini 3.1 Pro 100%." src="results/f2_chart_light.png">
</picture>

## Status

| Stage | What | State |
|---|---|---|
| Synthetic data | 800 claim pages (10 document types × 80) generated from 314 sampled claim cases, each with a clean render and a phone-photo or scanner copy | ✅ done |
| F2: document type | Classify a page into 10 types; baselines → LoRA fine-tune → compare with Gemini | ✅ **Fine-tuned Qwen 0.5B matches Gemini: 100% on clean text, 98.4% on OCR + hidden titles (from 18.5% zero-shot), statistically indistinguishable.** 8 min to train on one GPU, 34 MB adapter ([results/f2.md](results/f2.md)) |
| F0: reading | Which OCR reads amounts, dates and policy numbers correctly | ⏳ next |
| Later | Field extraction, deduction checks with a rules engine, RL with verifiable rewards | planned |

## How the data is made

```
sample a claim case ──▶ Gemini writes each document from the case ──▶ render to an A4 page ──▶ degrade (phone / scan)
   synth/sample.py           synth/write_docs.py                        synth/render.py
```

Every fact in a document comes from its case, so labels are exact and free. All names,
insurers and hospitals are fictional. No real customer data is used anywhere in this repo.

## Layout

```
synth/     case schema, case sampler, document writer, renderer, degrader, generate.py
bench/     split (splits.json is frozen), OCR and title masking, baselines, finetune.py, score.py, report.py
results/   one file per task: success bar, baselines, results
docs/      research log
```

## Reproduce

```bash
uv sync
uv run python -m synth.try_case                 # schema checks
uv run python -m synth.sample --n 1000          # sample cases
uv run python -m synth.generate --per-type 80   # generate documents (needs a GCP project with Vertex AI)
uv run python -m bench.split                    # case-level split
```

Generated data goes to `data/` and is not committed.
