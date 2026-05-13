OCR_PROMPT = (
  "Extract all readable text from this document. "
  "If the document has tables, preserve column structure with pipes. "
  "If it is multi-column, extract the left column first, then the right. "
  "Return only the extracted text."
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
      "plain_english_explanation": "...", 
      "negotiation_tip": "<Actionable advice on what the user should ask to change>",
      "suggested_rewrite": "<1-2 sentence fairer replacement clause>",
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
5. The "compatibility_score" MUST directly reflect the User Requirements below. If a requirement is completely violated, the compatibility score must drop significantly.
6. For each red flag, include suggested_rewrite: a fairer version of the clause in 1-2 sentences.

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
  "internship": "unpaid work mandates, intellectual property assignment for side projects, conversion clauses, post-internship bonds, unreasonable working hours, and non-competes",
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
Rewrite the following clause in plain English for a non-lawyer.
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

NEGOTIATION_EMAIL_PROMPT = """
You are a professional negotiator. Based on the following contract red flags, write a polite, professional, and firm email to the counterparty requesting changes to the contract.
Do not be overly aggressive, but stand your ground. 
Format it with clear spacing. Do not include a subject line.
Start with "Dear {company_name}," (or "Dear Hiring Manager," if no company name is provided).
Sign off the email with "{user_name}" (or "[Your Name]" if no name is provided).

Red Flags to address:
{flags_text}
""".strip()

TLDR_SUMMARY_PROMPT = """
Write a TL;DR summary in exactly 3 sentences. Use plain English and avoid legal jargon.
Sentence 1: Overall risk and verdict.
Sentence 2: 1-2 key red flags.
Sentence 3: Any notable safe clauses or what the user should do next.

Contract type: {contract_label}
Risk score: {risk_score}
Verdict: {verdict}
Top red flags: {red_flags}
Safe clauses: {safe_clauses}
User requirements: {requirements}
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

FINAL_ENRICHMENT_PROMPT = """
You are a legal expert helping a non-lawyer understand a contract and negotiate better terms.
Return ONLY valid JSON with exactly these fields:
{{
  "tldr": "<exactly 3 sentences>",
  "negotiation_email": "<professional email>",
  "explanations": [
    {{"index": 1, "plain_english_explanation": "<2-4 sentences>"}}
  ]
}}

Rules:
1. Return ONLY valid JSON. No preamble. No markdown.
2. If explanation_targets is empty, return an empty explanations array.
3. Each explanations entry must include the same index from explanation_targets.
4. negotiation_email must start with "Dear {company_name}," (or "Dear Hiring Manager," if company_name is empty).
5. negotiation_email must end with "{user_name}" (or "[Your Name]" if user_name is empty).

Contract type: {contract_label}
Risk score: {risk_score}
Verdict: {verdict}
User requirements: {requirements}
Company name: {company_name}
User name: {user_name}

Red flags (context):
{red_flags_json}

Safe clauses (context):
{safe_clauses_json}

Explanation targets (rewrite these only):
{explanation_targets_json}
""".strip()