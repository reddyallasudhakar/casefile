"""Eligibility (rules engine part A): policy in force, waiting periods, moratorium, and whether
a rejection on those grounds stands.

    from rules.eligibility import check_eligibility, audit_rejection

Conventions (decided in docs/RULES_ENGINE.md):
- Continuous inception = the earliest valid date among original inception, previous insurer's
  inception and the current term's start. One rule, everywhere.
- Months are calendar months: a month completes on the same day-of-month, so
  2020-01-15 → 2025-01-15 is 60 months and → 2025-01-14 is 59.
- Days are whole days between the dates; a 30-day wait is cleared on day 30.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel

from rules.terms import ClaimFacts, PolicyTerms


class Check(BaseModel):
    rule: str  # ID in rules.catalogue.RULES
    applies: bool  # does this rule bear on this claim at all?
    passed: bool | None  # None = can't decide from the facts given
    elapsed: int | None = None
    threshold: int | None = None
    unit: Literal["days", "months"] | None = None
    detail: str


class Eligibility(BaseModel):
    continuous_inception: date | None
    checks: list[Check]
    eligible: bool | None  # False if any applicable check failed; None if any couldn't be decided

    def check(self, rule: str) -> Check | None:
        return next((c for c in self.checks if c.rule == rule), None)


Verdict = Literal["valid", "invalid", "needs_info"]


class RejectionAudit(BaseModel):
    ground: str
    verdict: Verdict
    rule: str | None
    reason: str


def calendar_months(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(0, months)


def continuous_inception(policy: PolicyTerms) -> date:
    candidates = [policy.policy_start, policy.original_inception, policy.previous_insurer_inception]
    return min(d for d in candidates if d is not None)


def check_eligibility(policy: PolicyTerms, claim: ClaimFacts) -> Eligibility:
    if claim.admission_date is None:
        return Eligibility(
            continuous_inception=continuous_inception(policy), eligible=None,
            checks=[Check(rule="ELIG-INFORCE", applies=True, passed=None, detail="No admission date: nothing can be checked.")],
        )

    admitted = claim.admission_date
    start = continuous_inception(policy)
    days = (admitted - start).days
    months = calendar_months(start, admitted)
    checks = []

    # In force
    in_force = policy.policy_start <= admitted and (policy.policy_end is None or admitted <= policy.policy_end)
    checks.append(Check(
        rule="ELIG-INFORCE", applies=True, passed=in_force,
        detail=f"Admitted {admitted}; current term {policy.policy_start} to {policy.policy_end or 'not stated'}.",
    ))

    # Initial waiting period (accidents exempt)
    if claim.is_accident:
        checks.append(Check(rule="ELIG-INITIAL", applies=False, passed=None, detail="Accident: initial waiting period doesn't apply."))
    else:
        checks.append(Check(
            rule="ELIG-INITIAL", applies=True, passed=days >= policy.initial_waiting_days,
            elapsed=days, threshold=policy.initial_waiting_days, unit="days",
            detail=f"{days} days of continuous cover since {start} (needs {policy.initial_waiting_days}).",
        ))

    # Specific-disease waiting period
    wait = next((w for w in policy.specific_waits if claim.condition and w.condition == claim.condition), None)
    if claim.is_accident or wait is None:
        why = "Accident." if claim.is_accident else (
            "Condition not stated." if claim.condition is None else f"'{claim.condition}' has no specific waiting period in this policy.")
        checks.append(Check(rule="ELIG-SPECIFIC", applies=False, passed=None, detail=why))
    else:
        checks.append(Check(
            rule="ELIG-SPECIFIC", applies=True, passed=months >= wait.months,
            elapsed=months, threshold=wait.months, unit="months",
            detail=f"{months} months of continuous cover for '{wait.condition}' (needs {wait.months}).",
        ))

    # Pre-existing disease waiting period
    if claim.is_accident or policy.ped_wait_waived or claim.condition_is_ped is False:
        why = "Accident." if claim.is_accident else ("PED waiting period waived." if policy.ped_wait_waived else "Condition isn't pre-existing.")
        checks.append(Check(rule="ELIG-PED", applies=False, passed=None, detail=why))
    elif claim.condition_is_ped is None:
        checks.append(Check(
            rule="ELIG-PED", applies=True, passed=None, elapsed=months, threshold=policy.ped_waiting_months, unit="months",
            detail=f"Unknown whether the condition is pre-existing; {months} months elapsed (wait {policy.ped_waiting_months}).",
        ))
    else:
        checks.append(Check(
            rule="ELIG-PED", applies=True, passed=months >= policy.ped_waiting_months,
            elapsed=months, threshold=policy.ped_waiting_months, unit="months",
            detail=f"{months} months of continuous cover for a pre-existing condition (needs {policy.ped_waiting_months}).",
        ))

    # Moratorium: not a gate on payment, a shield against contesting the policy
    checks.append(Check(
        rule="ELIG-MORATORIUM", applies=True, passed=months >= policy.moratorium_months,
        elapsed=months, threshold=policy.moratorium_months, unit="months",
        detail=f"{months} continuous months (moratorium at {policy.moratorium_months}).",
    ))

    gates = [c for c in checks if c.applies and c.rule != "ELIG-MORATORIUM"]
    eligible = False if any(c.passed is False for c in gates) else None if any(c.passed is None for c in gates) else True
    return Eligibility(continuous_inception=start, checks=checks, eligible=eligible)


def audit_rejection(policy: PolicyTerms, claim: ClaimFacts) -> RejectionAudit | None:
    """Does the insurer's rejection ground stand, given the eligibility checks?"""
    ground = claim.rejection_ground
    if ground is None:
        return None
    e = check_eligibility(policy, claim)
    mora = e.check("ELIG-MORATORIUM")
    protected = bool(mora and mora.passed) and not claim.fraud_alleged

    def out(verdict, rule, reason):
        return RejectionAudit(ground=ground, verdict=verdict, rule=rule, reason=reason)

    if claim.admission_date is None:
        return out("needs_info", None, "No admission date.")

    if ground == "initial_waiting_period":
        c = e.check("ELIG-INITIAL")
        if not c.applies:
            return out("invalid", "ELIG-INITIAL", c.detail)
        return out("valid" if c.passed is False else "invalid", "ELIG-INITIAL", c.detail)

    if ground == "specific_disease_waiting_period":
        c = e.check("ELIG-SPECIFIC")
        if claim.is_accident:
            return out("invalid", "ELIG-SPECIFIC", c.detail)
        if not c.applies:
            return out("needs_info", "ELIG-SPECIFIC", c.detail + " Check the policy's list of specified conditions.")
        return out("valid" if c.passed is False else "invalid", "ELIG-SPECIFIC", c.detail)

    if ground == "pre_existing_disease_waiting_period":
        if protected:
            return out("invalid", "ELIG-MORATORIUM", mora.detail + " A pre-existing-disease rejection can't stand after the moratorium.")
        c = e.check("ELIG-PED")
        if policy.ped_wait_waived:
            return out("invalid", "ELIG-PED", c.detail)
        if c.passed is None:
            return out("needs_info", "ELIG-PED", c.detail)
        if not c.applies:  # established not pre-existing
            return out("invalid", "ELIG-PED", c.detail)
        return out("valid" if c.passed is False else "invalid", "ELIG-PED", c.detail)

    if ground == "non_disclosure":
        if protected:
            return out("invalid", "ELIG-MORATORIUM", mora.detail + " Non-disclosure can't be contested after the moratorium.")
        return out("needs_info", "ELIG-MORATORIUM",
                   mora.detail + (" Fraud alleged: the moratorium doesn't protect it." if claim.fraud_alleged
                                  else " Within the moratorium, non-disclosure depends on what was disclosed."))

    return out("needs_info", None, f"'{ground}' isn't decided by eligibility rules (v1).")
