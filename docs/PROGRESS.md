# Progress tracker

The single checklist for Casefile and the learning plan: what's done, what's next, what's open.
Updated at the end of every step. Reasoning behind decisions lives in [LOG.md](LOG.md); the
full plan in `llm-training-lab/PROJECT_PLAN.md`.

Legend: `[x]` done · `[~]` in progress / partly done · `[ ]` to do

_Last updated: 2026-10-24_

---

## ▶ Now / next

- [ ] **F5: extract claim facts** (settlement letter, bill, policy schedule, discharge summary → JSON). See *F5* below
- [ ] Set the F5 success bar **before** any results
- [x] Commit `docs/RULES_ENGINE.md` + this file (`8c65581`)

---

## Setup and infrastructure

- [x] Learning plan written, published as [llm-training-lab](https://github.com/reddyallasudhakar/llm-training-lab)
- [x] Casefile repo created, public on personal GitHub
- [x] Python environments (uv), M0 environment check
- [x] GCP project `casefile-ml`: billing, budget alerts (50/90/100%), Vertex AI, Compute, Storage, Vision APIs
- [x] GPU VM `casefile-gpu`: L4, Spot, 4-hour automatic stop, us-central1-a (currently **stopped**)
- [x] Gemini access via Vertex AI (`gemini-3.5-flash`, `gemini-3.1-pro-preview`)
- [x] IPv6 fix for Python (`synth/net.py`)
- [x] Research log (`docs/LOG.md`), cost analysis (`docs/COSTS.md`)
- [ ] Hugging Face login (needed to publish models)
- [ ] CI on GitHub Actions (run schema checks and golden tests on every push)

## Synthetic data

- [x] Claim case schema with consistency rules (`synth/case.py`, 14 checks)
- [x] Case sampler: 11 conditions, age-appropriate, room-priced bills (`synth/sample.py`)
- [x] Document writer (Gemini) → renderer → phone/scan degrader (`synth/write_docs.py`, `render.py`)
- [x] 800 pages: 10 types × 80, from 314 cases (`synth/generate.py`)
- [x] OCR of all 800 degraded pages with Cloud Vision (`bench/ocr.py`)
- [x] Title-hidden variants (`bench/mask_titles.py`)
- [x] Case-level split 70/15/15, frozen (`bench/splits.json`)
- [ ] Record each page's writing style in the index (needed for held-out-style tests)
- [ ] More pages for extraction, if 80 per type proves too few

## F0: reading (OCR)

- [x] Cloud Vision: key-fact survival measured (97–100%; one 6→8 misread in a policy number)
- [ ] Compare OCR engines (Cloud Vision vs Tesseract vs an open document model vs Gemini reading the image)

## F2: document-type classification ✅

- [x] Label guide (10 types)
- [x] Success bar set before results
- [x] Scoring script with Wilson intervals and McNemar (`bench/score.py`)
- [x] Baselines: majority, keywords, Qwen 0.5B/1.5B/3B zero-shot + few-shot, Gemini Flash + Pro, on 4 text variants
- [x] LoRA fine-tune at 0.5B, 1.5B, 3B (`bench/finetune.py`)
- [x] **Result: fine-tuned 0.5B = 100% clean, 98.4% OCR + titles hidden; indistinguishable from Gemini**
- [x] Results page + chart published ([results/f2.md](../results/f2.md))
- [ ] Reconcile the 10 labels with the FairClaims app's production document types
- [ ] Publish the 0.5B adapter on Hugging Face with a model card
- [ ] Laptop timing check for the 0.5B (ask first)
- [ ] Test on real pages (private, de-identified; consent permitting)
- [ ] Test on pages from a different writer model

## F5: extracting claim facts (next)

Question: F2 showed a small model can *recognise* a document. Can it *read it precisely*, i.e.
copy every amount, date and ID exactly, as well as Gemini? And does GRPO improve on SFT?

- [ ] Schemas for 4 types (settlement letter, hospital bill, policy schedule, discharge summary)
- [ ] Success bar, set before results
- [ ] Gold JSON per page from the cases, filtered to facts actually on the page
- [ ] Scoring: per-field, whole-document, valid-JSON rate, deduction-list precision/recall
- [ ] Baselines: Qwen 0.5B/1.5B/3B prompted, Gemini Flash + Pro (clean and OCR)
- [ ] SFT with LoRA (JSON output)
- [ ] RL lesson: policy gradients → PPO → GRPO
- [ ] GRPO on top of SFT, reward = fields correct + valid JSON; reward-hacking check
- [ ] Results page + log

## Rules engine (deferred; designed)

- [x] Backend review: most money rules are only in LLM prompts; inconsistencies listed
- [x] Plan written ([RULES_ENGINE.md](RULES_ENGINE.md))
- [ ] Decide: continuous-inception rule, order of deductions, products to model
- [ ] Fill clause citations from the IRDAI Master Circular 2024 (hands-on reading)
- [ ] R1 eligibility · R2 payable · R3 audit · R4 CI · R5 sampler injects insurer mistakes
- Trigger: build when the adjudication task (T3) starts; minimal version (payable only) may be enough

## Later phases (from the project plan)

- [ ] F3 bill line items · F4 rejection reason · F6/F7 other extraction
- [ ] T3 adjudication (needs the rules engine) · T4 appeal letters with DPO
- [ ] Post-training from a base model (SFT → distillation → DPO → GRPO)
- [ ] Distillation D1/D2/D3 · compress to 0.5B
- [ ] QLoRA at 7B
- [ ] DSPy prompt-optimisation comparison
- [ ] Demo (Gradio on Hugging Face Spaces)
- [ ] Portfolio posts: (1) the benchmark, (2) F2 fine-tuning, (3) RL, (4) distillation

## Learning modules (llm-training-lab)

- [x] M0 environment
- [~] M1 how LLMs generate: Part 1 taught; **exercises 2–6 not done**
- [x] M2 evals (via F2)
- [ ] M3 embeddings · M4 prompt vs RAG vs fine-tune vs RL
- [ ] M5 PyTorch · M6 transformer from scratch · M7 model lifecycle
- [ ] M8 encoder classifier
- [x] M9 SFT data (via F2) · [x] M10 LoRA (via F2)
- [ ] M11 QLoRA · M12 distillation · M13 shipping a model
- [ ] M14–M15 reward models, DPO
- [ ] M16–M20 RL foundations (bandits → PPO)
- [ ] M21–M24 RLHF, GRPO, reward design, RL for agents
- [ ] M25 search and prompt optimisation

## Open decisions

- [ ] F5 success bar
- [ ] Rules-engine defaults (continuous inception, deduction order)
- [ ] Whether registration consent covers a private real-page test set

## Known issues and risks

- [ ] **Private backend repo has what looks like real customer data in git history** (`src/tests/e2e/cases/`, some test fixtures). Review under DPDP, separately from Casefile
- [ ] Google Workspace logins expire mid-session (`gcloud auth login`, `application-default login`); long jobs should run on the VM with its service account
- [ ] IPv6 on the home network is broken; other Python tools on the Mac may still hang
- [ ] Spot VMs can be preempted, and L4s can be out of stock; run long jobs detached (`nohup`)
- [ ] Training and test pages share one generator (Gemini 3.5 Flash)
