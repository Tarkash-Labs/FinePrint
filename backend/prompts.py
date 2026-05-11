OCR_PROMPT = (
    "Extract all readable text from this document. "
    "Return only the raw text, with no commentary."
)

def _build_analysis_prompt(contract_label: str, focus_areas: str) -> str:
  return f"""
You are a legal expert specializing in {contract_label} contracts.
Analyze this contract and return a JSON with exactly these fields:
{{
  "risk_score": <integer 0-100>,
  "red_flags": [{{"clause_title": "...", "clause_text": "...", "plain_english_explanation": "...", "severity": "high|medium|low"}}],
  "safe_clauses": [{{"clause_title": "...", "plain_english_explanation": "..."}}]
}}
Return only valid JSON. No preamble. Do not use markdown blocks, just raw JSON.
For red flags, include the exact clause text in "clause_text".
Focus on {focus_areas}.
""".strip()

CONTRACT_TYPE_LABELS = {
  "employment": "Employment Bond",
  "rental": "Rental Lease",
  "freelance": "Freelance NDA",
  "vc": "VC Term Sheet",
  "tos": "Terms of Service",
  "general": "General Contract",
}

CONTRACT_ANALYSIS_PROMPTS = {
  "employment": _build_analysis_prompt(
    "Employment Bond",
    "bonds, notice periods, training costs, repayment clauses, penalties, non-competes, IP assignment, and exit restrictions",
  ),
  "rental": _build_analysis_prompt(
    "Rental Lease",
    "eviction clauses, deposits, repair obligations, late fees, landlord access, unilateral changes, and auto-renewals",
  ),
  "freelance": _build_analysis_prompt(
    "Freelance NDA",
    "IP ownership, confidentiality scope, non-solicitation, exclusivity, liability limits, indemnity, and termination",
  ),
  "vc": _build_analysis_prompt(
    "VC Term Sheet",
    "liquidation preference, anti-dilution, control rights, board seats, vesting, drag-along, and option pool clauses",
  ),
  "tos": _build_analysis_prompt(
    "Terms of Service",
    "data sharing or selling, arbitration, auto-renewals, unilateral changes, liability limits, subscription traps, and cancellation",
  ),
  "general": _build_analysis_prompt(
    "General Contract",
    "termination, renewal, fees, indemnity, liability limits, arbitration, assignment, and unusual obligations",
  ),
}

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
