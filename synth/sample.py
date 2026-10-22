"""Sample random but realistic ClaimCase objects.

    uv run python -m synth.sample --n 1000           # writes data/cases.jsonl
    uv run python -m synth.sample --show 42          # print the case with seed 42

Every case comes from its own integer seed, so any case can be regenerated exactly.
Deductions are plausible, not rule-correct: computing what *should* have been paid is the
rules engine's job, and comparing the two is the point of the project.
"""

import argparse
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from synth.case import ClaimCase

# Invented names only: this data is published, and fake letters must not carry real brands.
INSURERS = [
    "Sahyog General Insurance", "Kavach Health Insurance", "Suraksha Mutual General",
    "Arogya Shield Insurance", "Bharosa General Insurance", "Navjeevan Health Insurance",
]
TPAS = ["Medilink TPA", "CareBridge Health TPA", "Sanjeevani TPA Services", "HealthAxis TPA"]
PRODUCTS = [
    "Family Health Optima", "Arogya Plus", "Complete Care Floater", "Health Secure Gold",
    "Senior Citizen Care", "Individual Health Basic",
]
HOSPITAL_SUFFIXES = ["Hospital", "Multispeciality Hospital", "Medical Centre", "Healthcare", "Nursing Home"]
CITIES = [
    "Mumbai", "Pune", "Bengaluru", "Chennai", "Hyderabad", "Delhi", "Kolkata", "Ahmedabad",
    "Jaipur", "Lucknow", "Kochi", "Indore", "Nagpur", "Coimbatore", "Vijayawada",
]

# (diagnosis, procedure or None for medical management, stay in days, total bill range, has implant)
CONDITIONS = [
    ("Acute calculous cholecystitis", "Laparoscopic cholecystectomy", (2, 5), (110_000, 200_000), False),
    ("Senile cataract, right eye", "Phacoemulsification with IOL implantation", (0, 1), (30_000, 65_000), True),
    ("Dengue fever with thrombocytopenia", None, (4, 7), (40_000, 110_000), False),
    ("Acute appendicitis", "Laparoscopic appendicectomy", (2, 4), (70_000, 150_000), False),
    ("Primary osteoarthritis, right knee", "Total knee replacement", (4, 7), (250_000, 450_000), True),
    ("Coronary artery disease, single vessel", "PTCA with drug-eluting stent", (3, 5), (180_000, 350_000), True),
    ("Community-acquired pneumonia", None, (4, 8), (60_000, 160_000), False),
    ("Right inguinal hernia", "Laparoscopic hernioplasty with mesh", (2, 3), (80_000, 150_000), True),
    ("Acute gastroenteritis with dehydration", None, (2, 4), (25_000, 60_000), False),
    ("Left ureteric calculus", "Ureteroscopic lithotripsy with DJ stenting", (1, 3), (70_000, 140_000), False),
    ("Enteric fever", None, (4, 7), (35_000, 90_000), False),
]

ROOM_RATES = {  # ₹ per day
    "general_ward": (1_500, 3_000),
    "shared": (2_500, 4_500),
    "single_private": (4_000, 9_000),
    "deluxe": (8_000, 15_000),
    "suite": (15_000, 30_000),
}

# How the non-room part of a bill splits, before random jitter
SURGICAL_SPLIT = {"professional_fees": 0.30, "procedure_charges": 0.22, "pharmacy": 0.20, "investigations": 0.13, "consumables": 0.15}
MEDICAL_SPLIT = {"professional_fees": 0.20, "pharmacy": 0.40, "investigations": 0.28, "consumables": 0.12}
IMPLANT_SHARE = (0.15, 0.35)  # implants take this share of the non-room bill when present

OUTCOMES = {"partially_settled": 0.50, "settled": 0.25, "rejected": 0.15, "query_raised": 0.10}
REJECTION_REASONS = {
    "initial_waiting_period": 0.15,
    "specific_disease_waiting_period": 0.20,
    "pre_existing_disease_waiting_period": 0.15,
    "non_disclosure": 0.20,
    "excluded_treatment": 0.10,
    "not_requiring_hospitalisation": 0.15,
    "other": 0.05,
}


def pick(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def sample_patient(rng, fake):
    relationship = rng.choices(["self", "spouse", "child", "parent"], weights=[45, 25, 15, 15])[0]
    age = {"self": (25, 65), "spouse": (23, 65), "child": (1, 24), "parent": (50, 85)}[relationship]
    gender = rng.choice(["female", "male"])
    name = fake.name_female() if gender == "female" else fake.name_male()
    return {"name": name, "age": rng.randint(*age), "gender": gender, "relationship": relationship}


def sample_policy(rng, admission, total_bill, patient_age, rejection_reason):
    # Waiting-period rejections only make sense if the policy is young enough.
    days_before = {
        "initial_waiting_period": (5, 29),
        "specific_disease_waiting_period": (60, 700),
        "pre_existing_disease_waiting_period": (60, 1_000),
    }.get(rejection_reason, (180, 8 * 365))
    inception = admission - timedelta(days=rng.randint(*days_before))

    options = [300_000, 500_000, 700_000, 1_000_000, 1_500_000, 2_000_000, 5_000_000]
    sum_insured = rng.choice([s for s in options if s >= total_bill] or options[-1:])
    room_cap = rng.random()
    room_rent_limit = None if room_cap < 0.5 else sum_insured * 0.01 if room_cap < 0.8 else rng.choice([3_000, 4_000, 5_000])
    copay_pct = rng.choice([10, 20]) if patient_age >= 60 and rng.random() < 0.6 else 0

    year = inception.year
    return {
        "insurer": rng.choice(INSURERS),
        "tpa": rng.choice(TPAS) if rng.random() < 0.7 else None,
        "product": rng.choice(PRODUCTS),
        "policy_number": f"{rng.randint(1000, 9999)}/{rng.randint(100000, 999999)}/{year}/{rng.randint(1, 99999):05d}",
        "inception_date": inception,
        "sum_insured": sum_insured,
        "room_rent_limit": room_rent_limit,
        "copay_pct": copay_pct,
        "deductible": 0,
    }


def sample_bill(rng, condition, room_category, room_rate, days, total):
    diagnosis, procedure, _, _, has_implant = condition
    room_days = max(days, 1)
    items = [{"category": "room", "description": f"{room_category.replace('_', ' ').title()} room, {room_days} day(s) @ ₹{room_rate}/day", "amount": room_rate * room_days}]

    rest = total - items[0]["amount"]
    if has_implant:
        implant = round(rest * rng.uniform(*IMPLANT_SHARE))
        items.append({"category": "implants", "description": f"Implant for {procedure.lower()}", "amount": implant})
        rest -= implant

    split = SURGICAL_SPLIT if procedure else MEDICAL_SPLIT
    jittered = {k: v * rng.uniform(0.7, 1.3) for k, v in split.items()}
    norm = sum(jittered.values())
    descriptions = {
        "professional_fees": "Surgeon, anaesthetist and consultant fees" if procedure else "Consultant visit charges",
        "procedure_charges": f"OT charges for {procedure.lower()}" if procedure else "",
        "pharmacy": "Medicines and IV fluids",
        "investigations": "Laboratory and radiology investigations",
        "consumables": "Gloves, masks, syringes and disposable kits",
    }
    for category, share in jittered.items():
        items.append({"category": category, "description": descriptions[category], "amount": round(rest * share / norm)})
    return items


def sample_deductions(rng, bill_items, room_rate, days, policy):
    """1–3 plausible deductions, as an insurer might apply them (not necessarily correctly)."""
    by_cat = {i["category"]: i["amount"] for i in bill_items}
    claimed = sum(by_cat.values())
    deductions = []

    limit = policy["room_rent_limit"]
    if limit and room_rate > limit:
        ratio = limit / room_rate
        associated = sum(by_cat.get(c, 0) for c in ("room", "professional_fees", "procedure_charges", "investigations"))
        deductions.append({
            "amount": round(associated * (1 - ratio)),
            "reason": "room_rent_proportionate",
            "insurer_wording": f"Proportionate deduction applied as room rent ₹{room_rate}/day exceeds eligibility of ₹{limit:.0f}/day",
        })

    if "consumables" in by_cat and (not deductions or rng.random() < 0.7):
        deductions.append({
            "amount": round(by_cat["consumables"] * rng.uniform(0.2, 0.6)),
            "reason": "non_payable_item",
            "insurer_wording": "Non-medical expenses deducted as per list of non-payable items",
        })

    if policy["copay_pct"]:
        base = claimed - sum(d["amount"] for d in deductions)
        deductions.append({
            "amount": round(base * policy["copay_pct"] / 100),
            "reason": "co_pay",
            "insurer_wording": f"{policy['copay_pct']}% co-payment applicable as per policy terms",
        })

    if not deductions or rng.random() < 0.15:
        deductions.append({
            "amount": round(rng.uniform(500, 5_000)),
            "reason": "other",
            "insurer_wording": "Registration and administrative charges not payable",
        })

    # Keep it plausible: an insurer rarely deducts more than ~60% of a payable claim.
    total = sum(d["amount"] for d in deductions)
    if total > 0.6 * claimed:
        scale = 0.6 * claimed / total
        for d in deductions:
            d["amount"] = max(1, round(d["amount"] * scale))
    return deductions


def sample_case(seed: int) -> ClaimCase:
    rng = random.Random(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)

    patient = sample_patient(rng, fake)
    condition = rng.choice(CONDITIONS)
    diagnosis, procedure, stay, cost, _ = condition
    days = rng.randint(*stay)
    total = round(rng.uniform(*cost), -2)

    # Hospitals price most charges by room category, so the room stays ~15–30% of the bill
    # and a pricier room pushes the whole bill up.
    room_category = rng.choices(list(ROOM_RATES), weights=[15, 20, 40, 20, 5])[0]
    room_rate = rng.randint(*ROOM_RATES[room_category]) // 100 * 100
    room_amount = room_rate * max(days, 1)
    total = max(total, round(room_amount / rng.uniform(0.15, 0.30), -2))

    admission = date(2023, 1, 1) + timedelta(days=rng.randint(0, 1_365))
    outcome = pick(rng, OUTCOMES)
    rejection_reason = pick(rng, REJECTION_REASONS) if outcome == "rejected" else None
    policy = sample_policy(rng, admission, total, patient["age"], rejection_reason)

    bill_items = sample_bill(rng, condition, room_category, room_rate, days, total)
    claimed = sum(i["amount"] for i in bill_items)

    deductions = []
    if outcome == "partially_settled":
        deductions = sample_deductions(rng, bill_items, room_rate, days, policy)
    approved = 0 if outcome in ("rejected", "query_raised") else claimed - sum(d["amount"] for d in deductions)

    return ClaimCase(
        patient=patient,
        policy=policy,
        hospitalisation={
            "hospital": f"{fake.last_name()} {rng.choice(HOSPITAL_SUFFIXES)}",
            "city": rng.choice(CITIES),
            "mode": "cashless" if rng.random() < 0.6 else "reimbursement",
            "admission_date": admission,
            "discharge_date": admission + timedelta(days=days),
            "diagnosis": diagnosis,
            "procedure": procedure,
            "room_category": room_category,
        },
        bill_items=bill_items,
        claimed=claimed,
        approved=approved,
        deductions=deductions,
        outcome=outcome,
        rejection_reason=rejection_reason,
    )


def summarise(cases):
    outcomes = Counter(c.outcome for c in cases)
    print(f"{len(cases)} cases")
    for outcome, n in outcomes.most_common():
        print(f"  {outcome:<18} {n:>5}  ({n / len(cases):.0%})")
    print(f"  avg claimed   ₹{sum(c.claimed for c in cases) / len(cases):>10,.0f}")
    partial = [c for c in cases if c.outcome == "partially_settled"]
    if partial:
        cut = sum(1 - c.approved / c.claimed for c in partial) / len(partial)
        print(f"  avg deduction on partial settlements: {cut:.1%}")
    reasons = Counter(d.reason for c in cases for d in c.deductions)
    print("  deduction reasons:", dict(reasons.most_common()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000, help="number of cases")
    parser.add_argument("--start-seed", type=int, default=0, help="seed of the first case")
    parser.add_argument("--out", type=Path, default=Path("data/cases.jsonl"))
    parser.add_argument("--show", type=int, metavar="SEED", help="print one case and exit")
    args = parser.parse_args()

    if args.show is not None:
        print(sample_case(args.show).model_dump_json(indent=2))
        return

    cases = [sample_case(seed) for seed in range(args.start_seed, args.start_seed + args.n)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for seed, case in zip(range(args.start_seed, args.start_seed + args.n), cases):
            f.write(json.dumps({"seed": seed, **case.model_dump(mode="json")}) + "\n")
    summarise(cases)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
