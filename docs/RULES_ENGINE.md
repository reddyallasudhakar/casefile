# Rules engine: plan (v1)

**Purpose.** Given a policy, a hospitalisation and its bill, and the insurer's decision,
compute deterministically what the insurer *should* have paid, check every rejection and
deduction against the rules, and say which rule and clause applies. No model involved.

It serves three jobs:

1. **Ground truth for the hard tasks.** The adjudication task (T3) and deduction-strength task
   (F8) get exact labels from the engine, the same way F2 got labels from the case files.
2. **The RL reward.** GRPO training later rewards a model for matching the engine's verdict and
   amount, so the engine *is* the definition of correct.
3. **A deterministic core for FairClaims.** Today, the product's most consequential numbers
   come from an LLM's judgement (see below). A tested engine can sit underneath it.

## What the existing TypeScript backend does (reference, not a port)

A read-only review of `policygaido_backend` (Oct 2026) found:

| Area | In the backend | Engine v1 |
|---|---|---|
| Waiting periods (initial, specific disease, pre-existing), moratorium | **Deterministic date maths** | Re-implement, fixing the inconsistencies below |
| Co-pay, sum-insured cap, rounding | **Deterministic**, applied after the LLM decides what's recoverable | Re-implement |
| Cashless pre-auth eligibility | Deterministic rule scorer | Out of scope for v1 |
| Room-rent limit and proportionate deduction | **Only in LLM prompts** | **New** |
| ICU limits, deductible, sub-limits, non-payable items | **Only in LLM prompts** | **New** |
| Order of deductions (deductible → sub-limits → proportionate → co-pay) | Described in prompts, never computed | **New, fixed and documented** |
| Whether a deduction is fair | LLM judgement | **Deterministic where the rules decide it**; "needs info" otherwise |

**Inconsistencies in the backend that v1 deliberately does not copy:**

- Three different rules for "continuous cover start" (priority order vs earliest date vs a
  third ordering). v1 uses **one**.
- The initial 30-day wait is measured from the **current** policy term in two places, so a
  renewed policy can wrongly show "not cleared". v1 measures from continuous inception.
- Moratorium months are computed two ways (calendar months vs days ÷ 30.4375), which can
  disagree at the 60-month boundary. v1 uses **calendar months**.
- Substring matching on "ped" also matches "orthopedic", "stopped", "expedited". v1 matches
  structured fields, not text.
- Two wiring bugs mean the specific-disease fallback and declared-PED checks never run in
  production. (Worth fixing in the backend separately.)
- Sum-insured headroom ignores earlier claims in the same policy year. v1 takes
  "sum insured already used" as an input.

## Scope of v1

### A. Eligibility checks (date maths)

| Rule | Logic | Source to verify |
|---|---|---|
| Policy in force | admission date within the policy period (grace period: v2) | policy terms |
| Accident | an accident claim bypasses all waiting periods | IRDAI standard terms |
| Initial waiting period | 30 days from **continuous inception**, unless accident | IRDAI standard exclusion |
| Specific-disease waiting period | the condition's waiting period (from the policy, default 24 months) from continuous inception | policy schedule; IRDAI cap |
| Pre-existing disease waiting period | declared PED's waiting period (default 36 months, the IRDAI maximum since 2024) from continuous inception | Master Circular on Health Insurance, 2024 |
| Moratorium | after **60 continuous months**, no rejection for non-disclosure or PED except proven fraud | Master Circular 2024 |

**Continuous inception** = the earliest valid date among the original inception date, the
previous insurer's inception (if ported) and the current policy start. Months are calendar
months; a month completes on the same day-of-month.

### B. Payable amount, in a fixed order

1. **Non-payable items** removed (the IRDAI list of non-medical expenses); consumables are
   payable if the policy covers them.
2. **Room-rent limit:** if the room's daily rate exceeds the limit (absolute ₹/day or % of
   sum insured/day), the excess room rent is deducted, and a **proportionate deduction**
   applies to associated charges at `limit ÷ actual rate`. Not applied to ICU, pharmacy,
   consumables, implants or diagnostics.
3. **Sub-limits** (per category or condition, e.g. cataract per eye).
4. **Deductible.**
5. **Co-pay** (% of what remains).
6. **Sum-insured cap:** `min(amount, sum insured − already used)`.

Money is rounded to whole rupees at each step. **Each step records the rule ID, the clause
it encodes, the amount and a one-line reason**, so every rupee is traceable.

### C. Auditing the insurer's decision

- **Each deduction** the insurer made is compared with what the rules allow for that reason:
  - `valid`: matches the engine (within ₹1), or the engine would deduct more;
  - `excessive`: the rule applies but the insurer deducted more; **recoverable = the difference**;
  - `invalid`: the rule doesn't apply (e.g. a proportionate deduction on pharmacy, a room-rent
    deduction when there's no cap); recoverable = the full amount;
  - `needs_info`: the engine can't decide from the data (e.g. "not medically necessary").
- **A rejection** is checked the same way: a waiting-period or pre-existing-disease rejection
  after the moratorium is `invalid`; one within the waiting period is `valid`; exclusions and
  medical necessity are `needs_info` in v1.
- **Output:** recomputed payable, per-item verdicts with rule citations, and total recoverable.

### Out of scope for v1

Packages and unbundling, restoration and cumulative bonus, family-floater consumption,
pre/post-hospitalisation bill windows, AYUSH caps, day-care list lookups, grace periods,
the cashless pre-auth scorer.

## Design

```
rules/
  terms.py         input types: policy terms, hospitalisation, bill lines, insurer decision
  eligibility.py   A: dates, continuous inception, waiting periods, moratorium
  payable.py       B: the six-step settlement with an audit trail
  audit.py         C: verdicts on each deduction and rejection
  catalogue.py     every rule's ID, plain-English statement and source clause
tests/
  golden/*.json    input → expected output, one file per scenario (shared format)
  test_golden.py   runs every golden case; CI runs it on every push
```

**Golden cases are JSON, not Python,** so the same cases can later test a TypeScript version
in FairClaims, and any disagreement shows up in CI.

**Where golden cases come from:**
1. Worked examples written from the rules themselves (room-rent proportionate deduction,
   co-pay ordering, moratorium boundary days).
2. **The backend's synthetic test cases, adapted:** waiting-period date maths (porting,
   initial period, waived periods, moratorium at 60/61 months, missing dates), co-pay after
   deductions, sum-insured headroom, the "unlimited" sum-insured sentinel.
3. **Not** the backend tests flagged as containing real customer data. Those are excluded
   entirely.

## Connecting it to the data

Once the engine exists, the synthetic sampler stops inventing deductions. Instead:
1. sample a policy and a bill;
2. the engine computes the correct settlement;
3. the sampler plays an imperfect insurer, injecting realistic **mistakes** at a known rate
   (proportionate deduction on pharmacy, waiting period measured from the renewal date, PED
   rejection after the moratorium, co-pay applied before deductions…);
4. every case then carries exact labels: which deductions are wrong, by how much, and why.

That dataset is what the adjudication model is trained on, and the engine's verdict is the
RL reward.

## Milestones

| | Milestone | Done when |
|---|---|---|
| R1 ✅ | Eligibility (A) + golden cases | **Done 2026-10-25:** `rules/eligibility.py`, 20 golden cases in `tests/golden/eligibility/` (8 adapted from the backend's synthetic tests, 1 per backend inconsistency fixed), all passing; a mutation test (reintroducing the renewal-date bug) is caught |
| R2 ✅ | Payable (B) + golden cases | **Done 2026-10-29:** `rules/payable.py`, 18 hand-worked golden cases in `tests/golden/payable/`, every step traced to a rule; mutation-tested (proportionate deduction on pharmacy/diagnostics, banker's rounding) |
| R3 | Audit (C) | every verdict type covered by golden cases |
| R4 ✅ | CI on GitHub Actions | **Done 2026-10-25:** `.github/workflows/tests.yml` runs golden cases and schema checks on every push |
| R5 | Sampler uses the engine, with injected insurer mistakes | regenerated cases carry engine labels |

## Decisions

1. **Continuous-inception rule:** ✅ adopted for R1: the earliest valid date. (Overrule if
   your claims experience says insurers treat it differently.)
2. **Order of deductions** (B1–B6): ✅ approved 2026-10-29.
3. **Clause references:** the IRDAI Master Circular 2024 should be read in the original, and
   each rule's citation filled in from it. A good hands-on task.
4. **Which products to model first:** 3–5 real product structures (room-rent cap type,
   co-pay, sub-limits), encoded as parameters, never copied wording.
