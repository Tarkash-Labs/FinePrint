OCR_PROMPT = (
    "Extract all readable text from this document. "
    "Return only the raw text, with no commentary."
)

EMPLOYMENT_BOND_SYSTEM_PROMPT = """
You are a legal expert specializing in Employment Bond agreements.
Analyze this contract and return a JSON with exactly these fields:
{
  "risk_score": <integer 0-100>,
  "red_flags": [{"clause_title": "...", "plain_english_explanation": "...", "severity": "high|medium|low"}],
  "safe_clauses": [{"clause_title": "...", "plain_english_explanation": "..."}]
}
Return only valid JSON. No preamble. Do not use markdown blocks, just raw JSON.
Focus on bonds, notice periods, training costs, repayment clauses, penalties,
non-competes, and IP assignment.
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
