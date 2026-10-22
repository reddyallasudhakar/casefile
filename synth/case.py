"""The claim case schema: one health-insurance claim, from policy to insurer decision.

The synthetic generator samples ClaimCase objects first, then writes every document
(bill, discharge summary, settlement letter, ...) from the same case, so the documents
agree with each other and every field is a known label.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RUPEE_TOLERANCE = 1  # money comparisons allow ₹1 of rounding


class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=110)
    gender: Literal["female", "male", "other"]
    relationship: Literal["self", "spouse", "child", "parent"]  # to the policyholder


class Policy(BaseModel):
    insurer: str
    tpa: str | None = None  # some insurers settle in-house
    product: str
    policy_number: str
    inception_date: date
    sum_insured: float = Field(gt=0)
    room_rent_limit: float | None = Field(default=None, gt=0)  # ₹ per day; None = no cap
    copay_pct: float = Field(default=0, ge=0, le=100)
    deductible: float = Field(default=0, ge=0)


class Hospitalisation(BaseModel):
    hospital: str
    city: str
    mode: Literal["cashless", "reimbursement"]
    admission_date: date
    discharge_date: date
    diagnosis: str
    procedure: str | None = None  # medical management has no procedure
    room_category: Literal["general_ward", "shared", "single_private", "deluxe", "suite", "icu"]

    @model_validator(mode="after")
    def discharge_not_before_admission(self):
        if self.discharge_date < self.admission_date:
            raise ValueError("discharge_date is before admission_date")
        return self


class BillItem(BaseModel):
    category: Literal[
        "room", "icu", "professional_fees", "pharmacy", "investigations",
        "consumables", "implants", "procedure_charges", "other",
    ]
    description: str
    amount: float = Field(gt=0)


class Deduction(BaseModel):
    """One amount the insurer didn't pay, and the reason they gave."""

    amount: float = Field(gt=0)
    reason: Literal[
        "room_rent_proportionate", "non_payable_item", "sub_limit", "co_pay",
        "deductible", "not_medically_necessary", "other",
    ]
    insurer_wording: str  # what the letter actually says


class ClaimCase(BaseModel):
    patient: Person
    policy: Policy
    hospitalisation: Hospitalisation
    bill_items: list[BillItem] = Field(min_length=1)
    claimed: float = Field(gt=0)
    approved: float = Field(ge=0)
    deductions: list[Deduction] = []
    outcome: Literal["settled", "partially_settled", "rejected", "query_raised"]

    @model_validator(mode="after")
    def admitted_after_inception(self):
        if self.hospitalisation.admission_date < self.policy.inception_date:
            raise ValueError("admitted before the policy started")
        return self

    @model_validator(mode="after")
    def money_adds_up(self):
        # A rejection or an open query pays nothing, so there is nothing to reconcile.
        if self.outcome in ("rejected", "query_raised"):
            if self.approved != 0:
                raise ValueError(f"outcome is {self.outcome} but approved is {self.approved}")
            return self

        deducted = sum(d.amount for d in self.deductions)
        if abs(self.claimed - deducted - self.approved) > RUPEE_TOLERANCE:
            raise ValueError(
                f"approved {self.approved} ≠ claimed {self.claimed} − deductions {deducted}"
            )
        if self.outcome == "settled" and deducted > RUPEE_TOLERANCE:
            raise ValueError("outcome is settled but there are deductions; use partially_settled")
        return self
