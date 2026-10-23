"""Write claim documents from sampled cases, using Gemini.

Gemini writes the wording and layout; every fact (names, amounts, dates, numbers) comes
from the case, and a check rejects any document that drops the key facts.

    from synth.write_docs import doc_types_for, write_doc
"""

import random
import re
import threading

from google import genai
from google.genai import types

import synth.net  # noqa: F401  (IPv4-first, see synth/net.py)
from synth.case import ClaimCase

# gemini-3.8-flash queued for minutes per call on this project (Oct 2026); 3.5 answers in seconds.
MODEL = "gemini-3.5-flash"
# Writing from given facts needs little reasoning; default thinking tripled the tokens.
CONFIG = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))
_client = None
_client_lock = threading.Lock()


def client():
    """One shared client. The lock matters: parallel workers creating their own clients
    overwrite each other, and the discarded ones close while still in use."""
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client(vertexai=True, project="casefile-ml", location="global")
    return _client


# What each document is, written for the model (mirrors sprints/f2-doc-type/labels.md).
DOC_SPECS = {
    "settlement_letter": "A claim settlement letter from the insurer or its TPA to the policyholder, stating the claim has been paid. Include claim number, policy number, patient, hospital, admission dates, amount claimed, a table of deductions with the insurer's stated reasons, amount approved/paid, and a UTR/NEFT payment reference.",
    "rejection_letter": "A claim repudiation (rejection) letter from the insurer or its TPA to the policyholder. ₹0 is paid. State the claim and policy numbers, patient, hospitalisation, and the reason for repudiation citing a policy clause or exclusion number, and mention the grievance redressal route.",
    "preauth_denial": "A cashless pre-authorisation denial from the TPA/insurer addressed to the hospital's insurance desk, for the planned or ongoing admission. It denies cashless facility and says the patient may submit a reimbursement claim with full documents after discharge.",
    "query_letter": "A claim query / deficiency letter from the insurer or TPA asking for additional documents or clarification before the claim can be processed (e.g. indoor case papers, past consultation records, original bills, investigation reports). Give a deadline in days and say the claim may be closed if not received. No decision is made.",
    "policy_schedule": "The policy schedule page: insurer name, product name, policy number, period of insurance, insured member details, sum insured, premium paid, room-rent eligibility, co-payment, and key waiting periods.",
    "hospital_bill": "The hospital's final inpatient (IP) bill: hospital letterhead, bill number, patient, IP number, admission/discharge dates, itemised line items with amounts grouped by category, GST lines where relevant, and the grand total.",
    "discharge_summary": "The hospital's discharge summary: patient details, IP number, admission and discharge dates, diagnosis, presenting complaints, history, clinical findings, procedure performed (if any), course in hospital, condition at discharge, medications and advice on discharge, signed by the treating doctor.",
    "medical_report": "A single clinical report from this hospitalisation that is NOT a discharge summary: a laboratory report (tests with values and reference ranges) or a radiology report (findings and impression) relevant to the diagnosis.",
    "health_card": "The insurer or TPA health e-card: very short. Member name, member/card ID, policy number, insurer and TPA name, validity dates, and the TPA helpline. No coverage details or premium.",
    "other": "A claim-related page that is none of: settlement/rejection/query letter, pre-auth denial, policy schedule, hospital bill, discharge summary, lab/radiology report, or health card.",
}

OTHER_SUBTYPES = [
    "the claim form (Part A, filled by the insured: patient and bank details left as [REDACTED], declaration and signature line)",
    "one page of the policy wording booklet (general terms and conditions, numbered clauses, no member details)",
    "the patient's covering email to the insurer listing documents enclosed for reimbursement",
    "the hospital's admission/registration form",
    "a nearly blank page with only a scanned page number and a faint 'continued' note",
]

STYLES = [
    "large private insurer: formal, structured, with a reference block and footer disclaimers",
    "TPA: terse and templated, dense tables, small font, many reference numbers",
    "corporate hospital chain: polished letterhead, clean tables",
    "small nursing home in a tier-2 city: plain layout, typed in a hurry, slightly inconsistent formatting",
    "public-sector insurer: old-fashioned bureaucratic wording, long sentences",
]


def doc_types_for(case: ClaimCase, rng: random.Random) -> list[str]:
    """The documents a real claim file for this case would plausibly contain."""
    docs = ["policy_schedule", "health_card", "hospital_bill", "discharge_summary", "medical_report"]
    docs.append({
        "settled": "settlement_letter",
        "partially_settled": "settlement_letter",
        "rejected": "rejection_letter",
        "query_raised": "query_letter",
    }[case.outcome])
    # Many reimbursement claims start as a denied cashless request.
    if case.hospitalisation.mode == "reimbursement" and rng.random() < 0.4:
        docs.append("preauth_denial")
    if rng.random() < 0.5:
        docs.append("other")
    return docs


def key_facts(case: ClaimCase, doc_type: str) -> dict[str, str | int]:
    """Facts the document must contain; checked after generation."""
    facts: dict[str, str | int] = {}
    if doc_type != "other":
        facts["patient"] = case.patient.name.split()[0]
    if doc_type in ("settlement_letter", "rejection_letter", "query_letter", "preauth_denial", "policy_schedule", "health_card"):
        facts["policy_number"] = case.policy.policy_number
    if doc_type in ("settlement_letter", "hospital_bill"):
        facts["claimed"] = int(case.claimed)
    if doc_type == "settlement_letter":
        facts["approved"] = int(case.approved)
    if doc_type == "policy_schedule":
        facts["sum_insured"] = int(case.policy.sum_insured)
    return facts


def _numbers(text: str) -> set[int]:
    return {int(float(n.replace(",", ""))) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def missing_facts(html: str, facts: dict) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", html)
    numbers = _numbers(text)
    missing = []
    for name, value in facts.items():
        if isinstance(value, int):
            if value not in numbers:
                missing.append(name)
        elif value.lower() not in text.lower():
            missing.append(name)
    return missing


def build_prompt(case: ClaimCase, doc_type: str, rng: random.Random) -> str:
    spec = DOC_SPECS[doc_type]
    if doc_type == "other":
        spec += " Make it: " + rng.choice(OTHER_SUBTYPES) + "."
    hindi = rng.random() < 0.15
    return f"""You are generating a SYNTHETIC document for training document-understanding models on Indian health-insurance claims. All names and organisations are fictional.

Document to write: {doc_type}
What it is: {spec}
House style: {rng.choice(STYLES)}
{"Include one or two short lines in Hindi (Devanagari) where an Indian document plausibly would, e.g. a heading or a note." if hindi else ""}

The claim case (the ONLY source of facts):
{case.model_dump_json(indent=1)}

Rules:
- Use the case's names, numbers, dates and amounts exactly. Write amounts in Indian style (₹1,32,500). Invent only incidental details the case doesn't fix (claim number, IP number, addresses, doctor names, phone numbers, reference numbers), and keep them fictional.
- Never mention that the document is synthetic, and never include the document type label itself as a heading unless a real document would.
- Output ONE complete HTML document for ONE A4 page: inline CSS only, no external resources, no images, no JavaScript. Width 794px. Use a realistic letterhead, fonts and tables for this style.
- Output only the HTML, starting with <!doctype html>."""


def write_doc(case: ClaimCase, doc_type: str, seed: int, retries: int = 2) -> tuple[str, list[str]]:
    """Return (html, missing_facts). missing_facts is empty when every key fact made it in."""
    rng = random.Random(f"{seed}-{doc_type}")
    facts = key_facts(case, doc_type)
    for _ in range(retries + 1):
        reply = client().models.generate_content(
            model=MODEL, contents=build_prompt(case, doc_type, rng), config=CONFIG
        )
        html = re.sub(r"^```(?:html)?\s*|\s*```\s*$", "", (reply.text or "").strip())
        missing = missing_facts(html, facts)
        if not missing:
            return html, []
    return html, missing


if __name__ == "__main__":
    from synth.sample import sample_case

    case = sample_case(7)
    html, missing = write_doc(case, "settlement_letter", 7)
    print(html[:1500])
    print("missing facts:", missing)
