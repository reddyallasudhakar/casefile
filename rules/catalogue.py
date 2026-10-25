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
