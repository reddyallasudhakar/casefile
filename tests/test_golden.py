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

    for rule in ["ELIG-INFORCE", "ELIG-CONTINUITY", "ELIG-INITIAL", "ELIG-SPECIFIC", "ELIG-PED", "ELIG-MORATORIUM"]:
        assert rule in RULES and RULES[rule]["source"]
