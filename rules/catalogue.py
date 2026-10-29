"""Every rule the engine applies: an ID, the rule in plain English, and where it comes from.

Sources marked "verify" still need checking against the original text of the IRDAI Master
Circular on Health Insurance (2024) and the clause number filled in.
"""

RULES = {
    "ELIG-INFORCE": {
        "rule": "The admission must fall within the current policy period.",
        "source": "Policy schedule (period of insurance)",
    },
    "ELIG-CONTINUITY": {
        "rule": "Waiting periods and the moratorium run from the start of continuous cover: the "
                "earliest of the original inception, the previous insurer's inception (if ported) "
                "and the current term's start.",
        "source": "IRDAI Master Circular on Health Insurance 2024: portability and continuity credit (verify)",
    },
    "ELIG-INITIAL": {
        "rule": "Claims in the first N days of continuous cover (usually 30) are excluded, except "
                "for accidents. Measured from continuous inception, not the renewal date.",
        "source": "IRDAI standard exclusions: initial waiting period (verify)",
    },
    "ELIG-SPECIFIC": {
        "rule": "Conditions the policy lists with their own waiting period (e.g. cataract, hernia) "
                "are excluded until that many months of continuous cover have passed.",
        "source": "Policy schedule; IRDAI standard exclusions: specified disease waiting period (verify)",
    },
    "ELIG-PED": {
        "rule": "Pre-existing diseases are excluded until the PED waiting period (at most 36 months "
                "for products under the 2024 circular) has passed, unless waived.",
        "source": "IRDAI Master Circular on Health Insurance 2024: pre-existing disease waiting period (verify)",
    },
    "ELIG-MORATORIUM": {
        "rule": "After 60 continuous months, no claim can be contested for non-disclosure or "
                "misrepresentation, or rejected as pre-existing, except for established fraud.",
        "source": "IRDAI Master Circular on Health Insurance 2024: moratorium period (verify)",
    },
}

RULES.update({
    "PAY-NONPAYABLE": {
        "rule": "Non-medical items aren't payable; consumables (gloves, masks, kits) are payable only "
                "if the policy covers them.",
        "source": "IRDAI list of non-payable items (Annexure, health insurance guidelines) (verify)",
    },
    "PAY-ROOMRENT": {
        "rule": "Room rent above the per-day eligibility (fixed, or % of sum insured) is deducted: "
                "(actual rate − eligible rate) × days.",
        "source": "Policy schedule: room rent limit",
    },
    "PAY-PROPORTIONATE": {
        "rule": "When the room exceeds eligibility, charges that scale with room category (doctors' "
                "fees, procedure and OT charges) are paid at eligible rate ÷ actual rate. Not applied "
                "to ICU, pharmacy, consumables, implants or diagnostics, nor where the hospital's "
                "tariff doesn't vary by room category.",
        "source": "IRDAI guidelines on proportionate deduction (2020) (verify)",
    },
    "PAY-SUBLIMIT": {
        "rule": "A condition with a sub-limit (e.g. cataract) is paid up to that cap.",
        "source": "Policy schedule: sub-limits",
    },
    "PAY-DEDUCTIBLE": {
        "rule": "The deductible is subtracted from the admissible amount before co-pay.",
        "source": "Policy schedule: deductible",
    },
    "PAY-COPAY": {
        "rule": "Co-pay is the policyholder's percentage share of what remains after all the above.",
        "source": "Policy schedule: co-payment",
    },
    "PAY-SICAP": {
        "rule": "The payable amount can't exceed the sum insured still available this year.",
        "source": "Policy schedule: sum insured",
    },
})
