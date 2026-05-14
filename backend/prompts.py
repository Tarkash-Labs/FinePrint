OCR_PROMPT = (
    "Extract all text from this document. If the document has tables, preserve column structure with pipes. "
    "If it is a multi-column layout, extract the left column first, then the right. "
    "Return only the extracted raw text, with no commentary."
)

def _build_analysis_prompt(contract_label: str, focus_areas: str, requirements: str) -> str:
  requirements_text = requirements.strip() or "None provided."
  return f"""
You are a ruthless, detail-oriented legal expert specializing in {contract_label} contracts. Your job is to protect the user.
Analyze this contract and return a JSON with exactly these fields:
{{
  "risk_score": <integer 0-100>,
  "compatibility_score": <integer 0-100>,
  "verdict": "ACCEPT" | "NEGOTIATE" | "REJECT",
  "verdict_reason": "<1-2 sentence explanation referencing requirements and clauses>",
  "requirement_breakdown": [
    {{"requirement": "<specific user requirement>", "met": true/false, "explanation": "<why it was or wasn't met>"}}
  ],
  "red_flags": [
    {{
      "clause_title": "...", 
      "clause_text": "<EXACT text from the contract>", 
      "plain_english_explanation": "<Briefly state the risk>", 
      "negotiation_tip": "<Actionable advice on what the user should ask to change>",
      "suggested_rewrite": "<Provide a safer, alternative 1-2 sentence rewrite for this clause that the user can propose>",
      "severity": "high|medium|low"
    }}
  ],
  "safe_clauses": [{{"clause_title": "...", "plain_english_explanation": "..."}}]
}}

CRITICAL RULES:
1. Return ONLY valid JSON. No preamble. No markdown blocks.
2. DETECT ALL RED FLAGS. Do not summarize them into one. If there are 5 bad clauses, list 5 red flags.
3. You MUST extract the exact original text for "clause_text". 
4. The "risk_score" is objective based on standard legal risks. Focus heavily on: {focus_areas}.
5. The "compatibility_score" MUST directly reflect the User Requirements below. If a requirement is completely violated, score drops.

User Requirements to evaluate against:
{requirements_text}

Verdict guidance:
- ACCEPT when risk <= 30 and compatibility >= 70.
- REJECT when risk >= 61 or compatibility <= 30.
- Otherwise NEGOTIATE.
""".strip()

CONTRACT_TYPE_LABELS = {
  "employment": "Employment Bond",
  "internship": "Internship Agreement",
  "rental": "Rental Lease",
  "freelance": "Freelance NDA",
  "vc": "VC Term Sheet",
  "tos": "Terms of Service",
  "general": "General Contract",
}

CONTRACT_FOCUS_AREAS = {
  "employment": "bonds, notice periods, training costs, repayment clauses, penalties, non-competes, IP assignment, and exit restrictions",
  "internship": "unpaid work mandates, intellectual property assignment for side projects, conversion clauses, post-internship bonds, unreasonable working hours",
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

SUMMARY_CHUNK_PROMPT = """
Summarize the contract text below into concise plain text.
Preserve key clauses, obligations, penalties, and timelines.
Do not analyze or assign a risk score. Do not use bullet points or JSON.
""".strip()

SUMMARY_COMBINE_PROMPT = """
Combine the summaries below into a single concise contract text.
Preserve key clauses and details. Do not analyze or use bullet points.
""".strip()

CONSOLIDATED_DENSE_PROMPT = """
You are an elite legal negotiator. A contract ({contract_label}) has been analyzed, and red flags were found.
Your task is to generate a final summary and a negotiation email.

Risk Score: {risk_score}/100
Verdict: {verdict}
Red Flags Identified: {red_flags_summary}

Return ONLY a JSON object with exactly these keys:
{{
  "tldr": "<Exactly 3 sentences. 1: Overall risk/verdict. 2: Top red flags. 3: Next steps for the user. Plain English.>",
  "negotiation_email": "<A polite, professional, and firm email to {company_name} from {user_name} requesting changes to the red flags. No subject line. Leave blank if no red flags.>"
}}
""".strip()

BATCH_EXPLAIN_PROMPT = """
You are a legal expert translating complex contract clauses into plain English.
For each of the flagged clauses below, provide a highly polished, simple 1-2 sentence explanation of the risk for a non-lawyer, a negotiation tip, and a suggested rewrite.

Contract Type: {contract_label}

Clauses to explain:
{clauses_json}

Return ONLY a JSON array of objects with EXACTLY these fields:
[
  {{
    "clause_title": "<Title of the clause EXACTLY matching the input>",
    "plain_english_explanation": "<Refined Plain English explanation>",
    "negotiation_tip": "<Negotiation tip>",
    "suggested_rewrite": "<A 1-2 sentence safer alternative text to propose>"
  }}
]
""".strip()

CLAUSE_QA_PROMPT = """
You are a legal expert helping a non-lawyer understand a specific contract clause.
Respond in plain English with 2-4 sentences. Be concise and practical.

Contract type: {contract_label}
Clause title: {clause_title}
Clause text: {clause_text}

Conversation so far:
{history}

User question: {question}
""".strip()

COMPARE_PROMPT = """
You are an expert legal AI. The user has uploaded two versions of a contract (Original and Revised).
Compare the extracted Red Flags from both and determine what changed.

Return ONLY a JSON object with exactly these fields:
{{
  "summary": "<2-3 sentence plain English summary of what improved or got worse>",
  "resolved_flags": ["<Title of flag from Original that is fixed in Revised>"],
  "new_flags": ["<Title of new dangerous flag found in Revised>"],
  "remaining_flags": ["<Title of flag from Original still present in Revised>"],
  "overall_change": "IMPROVED" | "WORSE" | "UNCHANGED"
}}

Original Analysis Flags:
{v1_flags}

Revised Analysis Flags:
{v2_flags}
""".strip()