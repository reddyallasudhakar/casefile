"""Run every golden case in tests/golden/ against the rules engine.

    uv run pytest

Each JSON case states the policy, the claim and the expected result. Only the keys present
under "expect" are checked, so a case can pin down just what it's about. The same files can
later test a TypeScript version of the engine.
"""

import json
from pathlib import Path

import pytest

from rules.eligibility import audit_rejection, check_eligibility
from rules.terms import ClaimFacts, PolicyTerms

CASES = sorted(Path(__file__).parent.glob("golden/eligibility/*.json"))


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_eligibility(path):
    case = json.loads(path.read_text())
    policy, claim, expect = PolicyTerms(**case["policy"]), ClaimFacts(**case["claim"]), case["expect"]
    result = check_eligibility(policy, claim)

    if "eligible" in expect:
        assert result.eligible == expect["eligible"], case["name"]
    if "continuous_inception" in expect:
        assert str(result.continuous_inception) == expect["continuous_inception"], case["name"]
    for rule, fields in expect.get("checks", {}).items():
        check = result.check(rule)
        assert check is not None, f"{case['name']}: no {rule} check"
        for key, value in fields.items():
            assert getattr(check, key) == value, f"{case['name']}: {rule}.{key} = {getattr(check, key)}, expected {value}"
    if "rejection" in expect:
        audit = audit_rejection(policy, claim)
        for key, value in expect["rejection"].items():
            assert getattr(audit, key) == value, f"{case['name']}: rejection {key} = {getattr(audit, key)} ({audit.reason})"


def test_every_rule_cited():
    """Every rule the engine can report has a catalogue entry with a source."""
    from rules.catalogue import RULES

    for rule in ["ELIG-INFORCE", "ELIG-CONTINUITY", "ELIG-INITIAL", "ELIG-SPECIFIC", "ELIG-PED", "ELIG-MORATORIUM",
                 "PAY-NONPAYABLE", "PAY-ROOMRENT", "PAY-PROPORTIONATE", "PAY-SUBLIMIT", "PAY-DEDUCTIBLE", "PAY-COPAY", "PAY-SICAP"]:
        assert rule in RULES and RULES[rule]["source"]


# --- R2: payable ---------------------------------------------------------------------

PAYABLE = sorted(Path(__file__).parent.glob("golden/payable/*.json"))


@pytest.mark.parametrize("path", PAYABLE, ids=[p.stem for p in PAYABLE])
def test_payable(path):
    from rules.payable import settle
    from rules.terms import Bill, Coverage

    case = json.loads(path.read_text())
    result, expect = settle(Coverage(**case["coverage"]), Bill(**case["bill"])), case["expect"]
    trail = " | ".join(f"{s.rule} -{s.deducted} ={s.running_total}" for s in result.steps)

    for key in ("billed", "payable", "complete"):
        if key in expect:
            assert getattr(result, key) == expect[key], f"{case['name']}: {key} = {getattr(result, key)}, expected {expect[key]} [{trail}]"
    for rule, amount in expect.get("deducted", {}).items():
        assert result.deducted(rule) == amount, f"{case['name']}: {rule} deducted {result.deducted(rule)}, expected {amount} [{trail}]"
    # Every rupee is accounted for: billed − Σ deductions = payable
    assert result.billed - sum(s.deducted for s in result.steps) == result.payable


def test_steps_always_in_agreed_order():
    from rules.payable import settle
    from rules.terms import Bill, Coverage

    order = [s.rule for s in settle(Coverage(), Bill(lines=[{"category": "pharmacy", "amount": 100}])).steps]
    assert order == ["PAY-NONPAYABLE", "PAY-ROOMRENT", "PAY-SUBLIMIT", "PAY-DEDUCTIBLE", "PAY-COPAY", "PAY-SICAP"]
