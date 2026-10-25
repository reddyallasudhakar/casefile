"""Checks for the ClaimCase schema in synth/case.py.

Run from the repo root:  uv run python -m synth.try_case

This file is the spec: it builds cases as plain dicts, so case.py must use these field
names and values. Every check should print ✓ once case.py is complete.
"""

import copy

from pydantic import ValidationError

from synth.case import ClaimCase

VALID = {
    "patient": {"name": "Asha Verma", "age": 58, "gender": "female", "relationship": "self"},
    "policy": {
        "insurer": "Example General Insurance",
        "tpa": "Example TPA Services",  # may be None
        "product": "Family Health Optima",
        "policy_number": "P/211100/01/2025/004512",
        "inception_date": "2023-04-01",
        "sum_insured": 500000,
        "room_rent_limit": 5000,  # rupees per day; None means no cap
        "copay_pct": 10,
        "deductible": 0,
    },
    "hospitalisation": {
        "hospital": "City Care Hospital",
        "city": "Pune",
        "mode": "reimbursement",  # or "cashless"
        "admission_date": "2025-08-10",
        "discharge_date": "2025-08-14",
        "diagnosis": "Acute cholecystitis",
        "procedure": "Laparoscopic cholecystectomy",
        "room_category": "single_private",
    },
    "bill_items": [
        {"category": "room", "description": "Single room, 4 days", "amount": 28000},
        {"category": "professional_fees", "description": "Surgeon and anaesthetist", "amount": 60000},
        {"category": "pharmacy", "description": "Medicines", "amount": 32000},
        {"category": "investigations", "description": "USG abdomen, LFT, CBC", "amount": 15000},
        {"category": "consumables", "description": "Gloves, masks, kits", "amount": 15000},
    ],
    "claimed": 150000,
    "approved": 132500,
    "deductions": [
        {"amount": 12500, "reason": "room_rent_proportionate", "insurer_wording": "Proportionate deduction as room rent exceeds eligibility"},
        {"amount": 5000, "reason": "non_payable_item", "insurer_wording": "Non-medical consumables not payable"},
    ],
    "outcome": "partially_settled",  # settled | partially_settled | rejected | query_raised
}


def case(**changes):
    """A deep copy of VALID with dotted-path overrides, e.g. case(**{"patient.age": 200})."""
    data = copy.deepcopy(VALID)
    for path, value in changes.items():
        *parents, key = path.split(".")
        target = data
        for p in parents:
            target = target[p]
        target[key] = value
    return data


VALID_REJECTED = case(
    approved=0, deductions=[], outcome="rejected", rejection_reason="specific_disease_waiting_period"
)

SHOULD_PASS = {
    "valid partial settlement": VALID,
    "valid rejection (approved 0, no deductions)": VALID_REJECTED,
    "no TPA and no room-rent cap": case(**{"policy.tpa": None, "policy.room_rent_limit": None}),
}

SHOULD_FAIL = {
    # the four consistency rules
    "rule 1: discharged before admitted": case(**{"hospitalisation.discharge_date": "2025-08-09"}),
    "rule 2: admitted before policy started": case(**{"policy.inception_date": "2025-09-01"}),
    "rule 3: approved ≠ claimed − deductions": case(approved=140000),
    "rule 4: rejected but money approved": case(outcome="rejected", rejection_reason="other"),
    "rule 5: rejected without a reason": case(approved=0, deductions=[], outcome="rejected"),
    "rule 5: reason on a non-rejection": case(rejection_reason="non_disclosure"),
    # field-level checks
    "age out of range": case(**{"patient.age": 200}),
    "co-pay over 100%": case(**{"policy.copay_pct": 150}),
    "negative bill amount": case(**{"bill_items": [{"category": "room", "description": "x", "amount": -1}]}),
    "unknown outcome": case(outcome="maybe"),
    "unknown deduction reason": case(deductions=[{"amount": 17500, "reason": "because", "insurer_wording": "x"}]),
}

results = []
for name, data in SHOULD_PASS.items():
    try:
        ClaimCase.model_validate(data)
        results.append(("✓", name, ""))
    except ValidationError as e:
        results.append(("✗", name, f"rejected a valid case: {e.errors()[0]['msg']}"))

for name, data in SHOULD_FAIL.items():
    try:
        ClaimCase.model_validate(data)
        results.append(("✗", name, "accepted, should have raised an error"))
    except ValidationError:
        results.append(("✓", name, ""))

for mark, name, msg in results:
    print(f"{mark} {name}" + (f" — {msg}" if msg else ""))
passed = sum(r[0] == "✓" for r in results)
print(f"\n{passed}/{len(results)} checks pass")
raise SystemExit(0 if passed == len(results) else 1)  # non-zero so CI fails on any broken check
