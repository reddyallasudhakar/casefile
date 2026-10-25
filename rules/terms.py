"""Inputs to the rules engine: the policy terms and the claim facts it needs.

Deliberately smaller than synth.case.ClaimCase: only what the rules read, with every date and
waiting period explicit, so a golden case states exactly what the engine was told.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SpecificWait(BaseModel):
    """A condition with its own waiting period, as listed in the policy (e.g. cataract, 24 months)."""

    condition: str  # canonical key, e.g. "cataract", "hernia", "joint_replacement"
    months: int = Field(gt=0)


class PolicyTerms(BaseModel):
    policy_start: date  # start of the current policy term
    policy_end: date | None = None  # end of the current term; None = not stated
    original_inception: date | None = None  # first day of continuous cover with this insurer
    previous_insurer_inception: date | None = None  # if ported: first day of cover with the previous insurer
    initial_waiting_days: int = Field(default=30, ge=0)
    specific_waits: list[SpecificWait] = []
    ped_waiting_months: int = Field(default=36, gt=0)
    ped_wait_waived: bool = False  # e.g. a rider that waives the pre-existing-disease wait
    moratorium_months: int = Field(default=60, gt=0)


RejectionGround = Literal[
    "initial_waiting_period",
    "specific_disease_waiting_period",
    "pre_existing_disease_waiting_period",
    "non_disclosure",
    "excluded_treatment",
    "not_requiring_hospitalisation",
    "other",
]


class ClaimFacts(BaseModel):
    admission_date: date | None = None
    is_accident: bool = False
    condition: str | None = None  # canonical key of the condition treated, matched against specific_waits
    condition_is_ped: bool | None = None  # established as pre-existing? None = unknown
    fraud_alleged: bool = False  # the moratorium doesn't protect proven fraud
    rejection_ground: RejectionGround | None = None
