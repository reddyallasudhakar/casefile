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


# --- R2: what the payable calculation needs --------------------------------------------

BillCategory = Literal[
    "room", "icu", "professional_fees", "procedure_charges", "pharmacy", "investigations",
    "consumables", "implants", "non_medical", "other",
]


class BillLine(BaseModel):
    category: BillCategory
    amount: float = Field(ge=0)
    description: str = ""


class RoomRentLimit(BaseModel):
    """Per-day room eligibility: a fixed amount, or a percentage of the sum insured."""

    kind: Literal["per_day", "pct_si_per_day"]
    value: float = Field(gt=0)


class SubLimit(BaseModel):
    """A cap on what's payable for a condition, e.g. cataract ₹40,000 per claim."""

    condition: str
    amount: float | None = Field(default=None, gt=0)
    pct_si: float | None = Field(default=None, gt=0, le=100)


class Coverage(BaseModel):
    sum_insured: float | None = Field(default=None, gt=0)  # None = unlimited
    sum_insured_used: float = Field(default=0, ge=0)  # already consumed this policy year
    room_rent_limit: RoomRentLimit | None = None
    consumables_covered: bool = False
    sub_limits: list[SubLimit] = []
    deductible: float = Field(default=0, ge=0)
    copay_pct: float = Field(default=0, ge=0, le=100)


class Bill(BaseModel):
    lines: list[BillLine]
    room_rate_per_day: float | None = Field(default=None, gt=0)
    room_days: int | None = Field(default=None, ge=0)
    flat_tariff_hospital: bool = False  # charges don't vary by room category: no proportionate deduction
    condition: str | None = None
