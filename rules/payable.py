"""Payable amount (rules engine part B): what the insurer should have paid, step by step.

    from rules.payable import settle
    settlement = settle(coverage, bill)
    settlement.payable, settlement.steps

Six steps, always in this order (docs/RULES_ENGINE.md, agreed 2026-10-29):
1. non-payable items  2. room rent + proportionate deduction  3. sub-limit
4. deductible  5. co-pay  6. sum-insured cap
Every step records its rule ID, the rupees it removed, the running total and a reason. Money is
whole rupees, rounded half-up with exact decimal arithmetic at each step (Python's round() would
send ₹1,234.5 to 1,234; insurers' arithmetic sends it to 1,235).
"""

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel

from rules.terms import Bill, Coverage

# Charges that scale with the room category, so they're paid proportionately when the room
# exceeds eligibility. Pharmacy, consumables, implants, diagnostics and ICU are exempt.
PROPORTIONATE_CATEGORIES = ("professional_fees", "procedure_charges")


class Step(BaseModel):
    rule: str  # ID in rules.catalogue.RULES
    deducted: int  # rupees removed at this step (0 if the rule didn't bite)
    running_total: int  # payable after this step
    reason: str


class Settlement(BaseModel):
    billed: int
    steps: list[Step]
    payable: int
    complete: bool  # False if a step couldn't be computed from the facts given
    notes: list[str] = []

    def deducted(self, rule: str) -> int:
        return sum(s.deducted for s in self.steps if s.rule == rule)


def rupees(x) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fmt(n: int) -> str:
    """Indian digit grouping: 1,52,000."""
    s = str(abs(n))
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail = head[-2:] + "," + tail
        head = head[:-2]
    return ("-" if n < 0 else "") + "₹" + (head + "," + tail if head else tail)


def settle(coverage: Coverage, bill: Bill) -> Settlement:
    by_cat: dict[str, int] = {}
    for line in bill.lines:
        by_cat[line.category] = by_cat.get(line.category, 0) + rupees(line.amount)
    billed = sum(by_cat.values())
    total, steps, notes, complete = billed, [], [], True

    def step(rule, deducted, reason):
        nonlocal total
        deducted = max(0, min(deducted, total))
        total -= deducted
        steps.append(Step(rule=rule, deducted=deducted, running_total=total, reason=reason))

    # 1. Non-payable items
    non_payable = by_cat.get("non_medical", 0) + (0 if coverage.consumables_covered else by_cat.get("consumables", 0))
    step("PAY-NONPAYABLE", non_payable,
         "No non-payable items." if non_payable == 0 else
         f"Non-medical items {fmt(by_cat.get('non_medical', 0))}"
         + ("" if coverage.consumables_covered else f" + consumables {fmt(by_cat.get('consumables', 0))} (not covered)")
         + " are not payable.")

    # 2. Room rent and proportionate deduction
    limit = coverage.room_rent_limit
    if limit is None:
        step("PAY-ROOMRENT", 0, "No room-rent limit in this policy.")
    else:
        if limit.kind == "per_day":
            eligible = Decimal(str(limit.value))
        elif coverage.sum_insured is not None:
            eligible = Decimal(str(coverage.sum_insured)) * Decimal(str(limit.value)) / 100
        else:
            eligible = None
        if eligible is None or bill.room_rate_per_day is None or bill.room_days is None:
            complete = False
            notes.append("Room-rent limit exists but the eligible rate, room rate or number of days is unknown.")
            step("PAY-ROOMRENT", 0, "Can't apply the room-rent limit: rate or days unknown.")
        else:
            actual = Decimal(str(bill.room_rate_per_day))
            eligible_i = rupees(eligible)
            if actual <= eligible:
                step("PAY-ROOMRENT", 0, f"Room {fmt(rupees(actual))}/day is within eligibility of {fmt(eligible_i)}/day.")
            else:
                excess = rupees((actual - eligible) * bill.room_days)
                step("PAY-ROOMRENT", excess,
                     f"Room {fmt(rupees(actual))}/day exceeds eligibility of {fmt(eligible_i)}/day: "
                     f"({fmt(rupees(actual))} − {fmt(eligible_i)}) × {bill.room_days} days.")
                if bill.flat_tariff_hospital:
                    step("PAY-PROPORTIONATE", 0, "Hospital's tariff doesn't vary by room category: no proportionate deduction.")
                else:
                    associated = sum(by_cat.get(c, 0) for c in PROPORTIONATE_CATEGORIES)
                    deduction = rupees(Decimal(associated) * (actual - eligible) / actual)
                    step("PAY-PROPORTIONATE", deduction,
                         f"Doctors' and procedure charges {fmt(associated)} paid at {fmt(eligible_i)} ÷ {fmt(rupees(actual))} "
                         "of their value; ICU, pharmacy, consumables, implants and diagnostics are exempt.")

    # 3. Sub-limit for the condition
    sub = next((s for s in coverage.sub_limits if bill.condition and s.condition == bill.condition), None)
    if sub is None:
        step("PAY-SUBLIMIT", 0, "No sub-limit applies to this condition.")
    else:
        if sub.amount is not None:
            cap = rupees(sub.amount)
        elif coverage.sum_insured is not None:
            cap = rupees(Decimal(str(coverage.sum_insured)) * Decimal(str(sub.pct_si)) / 100)
        else:
            cap = None
        if cap is None:
            complete = False
            notes.append(f"Sub-limit for '{sub.condition}' is a % of an unknown sum insured.")
            step("PAY-SUBLIMIT", 0, "Sub-limit can't be computed: sum insured unknown.")
        else:
            step("PAY-SUBLIMIT", max(0, total - cap), f"'{sub.condition}' is capped at {fmt(cap)}.")

    # 4. Deductible
    step("PAY-DEDUCTIBLE", rupees(coverage.deductible),
         f"Deductible of {fmt(rupees(coverage.deductible))}." if coverage.deductible else "No deductible.")

    # 5. Co-pay on what remains
    copay = rupees(Decimal(total) * Decimal(str(coverage.copay_pct)) / 100)
    step("PAY-COPAY", copay,
         f"{coverage.copay_pct:g}% co-pay on {fmt(total)}." if coverage.copay_pct else "No co-pay.")

    # 6. Sum-insured cap
    if coverage.sum_insured is None:
        step("PAY-SICAP", 0, "Sum insured not limited (or not stated).")
    else:
        headroom = max(0, rupees(coverage.sum_insured) - rupees(coverage.sum_insured_used))
        step("PAY-SICAP", max(0, total - headroom),
             f"Sum insured available: {fmt(rupees(coverage.sum_insured))} − {fmt(rupees(coverage.sum_insured_used))} used = {fmt(headroom)}.")

    return Settlement(billed=billed, steps=steps, payable=total, complete=complete, notes=notes)
