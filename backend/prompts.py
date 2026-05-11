OCR_PROMPT = (
    "Extract all readable text from this document. "
    "Return only the raw text, with no commentary."
)

def _build_analysis_prompt(contract_label: str, focus_areas: str, requirements: str) -> str:
  requirements_text = requirements.strip() or "Not provided."
  return f"""
You are a legal expert specializing in {contract_label} contracts.
Analyze this contract and return a JSON with exactly these fields:
{{
  "risk_score": <integer 0-100>,
  "compatibility_score": <integer 0-100>,
  "verdict": "ACCEPT" | "NEGOTIATE" | "REJECT",
  "verdict_reason": "<1-2 sentence explanation referencing the user's requirements and key clauses>",
  "red_flags": [{{"clause_title": "...", "clause_text": "...", "plain_english_explanation": "...", "severity": "high|medium|low"}}],
  "safe_clauses": [{{"clause_title": "...", "plain_english_explanation": "..."}}]
}}
Return only valid JSON. No preamble. Do not use markdown blocks, just raw JSON.
For red flags, include the exact clause text in "clause_text".
The risk score is objective and should not use the user's requirements.
Focus on {focus_areas}.

User requirements:
{requirements_text}

Compatibility scoring rules:
- If requirements are missing or empty, set "compatibility_score" to 50.
- Otherwise, score based on alignment with the requirements (conflicts lower the score).

Verdict guidance:
- ACCEPT when risk <= 30 and compatibility >= 70.
- REJECT when risk >= 61 or compatibility <= 30.
- Otherwise NEGOTIATE.
""".strip()

CONTRACT_TYPE_LABELS = {
  "employment": "Employment Bond",
  "rental": "Rental Lease",
  "freelance": "Freelance NDA",
  "vc": "VC Term Sheet",
  "tos": "Terms of Service",
  "general": "General Contract",
}

CONTRACT_FOCUS_AREAS = {
  "employment": "bonds, notice periods, training costs, repayment clauses, penalties, non-competes, IP assignment, and exit restrictions",
  "rental": "eviction clauses, deposits, repair obligations, late fees, landlord access, unilateral changes, and auto-renewals",
  "freelance": "IP ownership, confidentiality scope, non-solicitation, exclusivity, liability limits, indemnity, and termination",
  "vc": "liquidation preference, anti-dilution, control rights, board seats, vesting, drag-along, and option pool clauses",
  "tos": "data sharing or selling, arbitration, auto-renewals, unilateral changes, liability limits, subscription traps, and cancellation",
  "general": "termination, renewal, fees, indemnity, liability limits, arbitration, assignment, and unusual obligations",
}

def get_contract_prompt(contract_type: str, requirements: str | None) -> str:
  label = CONTRACT_TYPE_LABELS.get(contract_type, CONTRACT_TYPE_LABELS["general"])
  focus_areas = CONTRACT_FOCUS_AREAS.get(contract_type, CONTRACT_FOCUS_AREAS["general"])
  return _build_analysis_prompt(label, focus_areas, requirements or "")

EXPLANATION_SYSTEM_PROMPT_TEMPLATE = "You are a legal expert specializing in {contract_label} contracts."

EXPLANATION_PROMPT_TEMPLATE = """
Rewrite the clause in plain English for a non-lawyer.
Keep it concise (2-4 sentences) and focus on practical impact.
Return only the explanation text.

Clause title: {clause_title}
Clause text: {clause_text}
""".strip()
SUMMARY_CHUNK_PROMPT = """
Summarize the contract text below into concise plain text.
Preserve key clauses, obligations, penalties, and timelines.
Do not analyze or assign a risk score. Do not use bullet points or JSON.
""".strip()

SUMMARY_COMBINE_PROMPT = """
Combine the summaries below into a single concise contract text.
Preserve key clauses and details. Do not analyze or use bullet points.
""".strip()
