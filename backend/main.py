import base64
import binascii
import json
import logging
import os
import re
import uuid
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, AsyncIterator, List, Optional, Literal

from config import settings
from gemma_client import GemmaClient, GemmaClientError
from prompts import (
    CONTRACT_TYPE_LABELS,
    EXPLANATION_PROMPT_TEMPLATE,
    EXPLANATION_SYSTEM_PROMPT_TEMPLATE,
    OCR_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_COMBINE_PROMPT,
    NEGOTIATION_EMAIL_PROMPT,
    TLDR_SUMMARY_PROMPT,
    CLAUSE_QA_PROMPT,
    get_contract_prompt,
)
from utils import extract_json_object, clamp_text, chunk_text

logger = logging.getLogger(__name__)

APP_START_TIME = time.monotonic()
EXPLANATION_MIN_CHARS = 40
ASK_CLAUSE_RATE_LIMIT = 20
ASK_CLAUSE_WINDOW_SECONDS = 60 * 5
ASK_CLAUSE_BUCKETS: dict[str, list[float]] = {}

def is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    times = ASK_CLAUSE_BUCKETS.get(ip, [])
    times = [t for t in times if now - t < ASK_CLAUSE_WINDOW_SECONDS]
    if len(times) >= ASK_CLAUSE_RATE_LIMIT:
        ASK_CLAUSE_BUCKETS[ip] = times
        return True
    times.append(now)
    ASK_CLAUSE_BUCKETS[ip] = times
    return False

def get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

app = FastAPI(title="FinePrint API", description="Analyzing contracts with Gemma 4")

SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "application/pdf",
}

CONTRACT_LABELS = CONTRACT_TYPE_LABELS

REPORT_TTL_SECONDS = 60 * 60 * 24 * 7
REPORT_MAX_ITEMS = 200

STORE_PATH = Path("report_store.json")

def _load_store() -> OrderedDict:
    if STORE_PATH.exists():
        try:
            return OrderedDict(json.loads(STORE_PATH.read_text()))
        except Exception:
            pass
    return OrderedDict()

def _save_store() -> None:
    try:
        STORE_PATH.write_text(json.dumps(dict(REPORT_STORE)))
    except Exception:
        pass

REPORT_STORE: OrderedDict = _load_store()

def store_report(report: dict) -> str:
    report_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    payload = {
        "report_id": report_id,
        "created_at": created_at.isoformat(),
        **report,
    }
    REPORT_STORE[report_id] = {
        "created_at": created_at.timestamp(),
        "payload": payload,
    }
    _prune_reports()
    _save_store()
    return report_id

def _prune_reports() -> None:
    now = datetime.now(timezone.utc).timestamp()
    expired = [key for key, value in REPORT_STORE.items() if now - value["created_at"] > REPORT_TTL_SECONDS]
    for key in expired:
        REPORT_STORE.pop(key, None)
    while len(REPORT_STORE) > REPORT_MAX_ITEMS:
        REPORT_STORE.popitem(last=False)

def _call_with_retry(fn, retries: int = 2, delay: float = 2.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except GemmaClientError as exc:
            if "429" in str(exc) and attempt < retries:
                time.sleep(delay * (attempt + 1))
                continue
            raise

gemma_client = GemmaClient(api_key=settings.api_key)

def decode_base64_payload(payload: str, mime_type: str | None) -> tuple[bytes, str]:
    trimmed = payload.strip()
    detected_mime = mime_type
    if trimmed.startswith("data:"):
        header, _, data = trimmed.partition(",")
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 payload")
        header_mime = header[5:].split(";")[0]
        detected_mime = header_mime or detected_mime
        trimmed = data

    try:
        decoded = base64.b64decode(trimmed, validate=True)
    except binascii.Error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 payload")

    if not decoded:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Base64 payload is empty")
    if not detected_mime:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing MIME type")

    return decoded, detected_mime

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def normalize_severity(value: Any) -> str:
    normalized = normalize_text(value).lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "medium"

def compute_verdict(risk_score: int, compatibility_score: int) -> str:
    if risk_score >= 61 or compatibility_score <= 30:
        return "REJECT"
    if risk_score <= 30 and compatibility_score >= 70:
        return "ACCEPT"
    return "NEGOTIATE"

def parse_analysis_payload(
    payload: dict,
) -> tuple[int, int, str, str, list[dict], list[dict], list[dict]]:
    risk_score = max(0, min(100, int(payload.get("risk_score", payload.get("riskScore", 0)))))
    compatibility_score = max(0, min(100, int(payload.get("compatibility_score", payload.get("compatibilityScore", 50)))))

    verdict = normalize_text(payload.get("verdict") or payload.get("decision")).upper()
    if verdict not in ("ACCEPT", "NEGOTIATE", "REJECT"):
        verdict = compute_verdict(risk_score, compatibility_score)

    verdict_reason = normalize_text(payload.get("verdict_reason") or payload.get("analysis_summary"))
    if not verdict_reason:
        verdict_reason = "Verdict based on overall risk and compatibility."

    raw_reqs = payload.get("requirement_breakdown", [])
    requirements: list[dict] = []
    if isinstance(raw_reqs, list):
        for req in raw_reqs:
            if isinstance(req, dict):
                requirements.append({
                    "requirement": normalize_text(req.get("requirement")),
                    "met": bool(req.get("met", False)),
                    "explanation": normalize_text(req.get("explanation"))
                })

    raw_red_flags = payload.get("red_flags", payload.get("redFlags", []))
    red_flags: list[dict] = []
    if isinstance(raw_red_flags, list):
        for entry in raw_red_flags:
            if isinstance(entry, dict):
                red_flags.append({
                    "clause_title": normalize_text(entry.get("clause_title") or entry.get("title")) or "Untitled clause",
                    "clause_text": normalize_text(entry.get("clause_text") or entry.get("clause")),
                    "plain_english_explanation": normalize_text(entry.get("plain_english_explanation") or entry.get("explanation")),
                    "negotiation_tip": normalize_text(entry.get("negotiation_tip") or "Consult a legal professional regarding this clause."),
                    "suggested_rewrite": normalize_text(entry.get("suggested_rewrite")),
                    "severity": normalize_severity(entry.get("severity")),
                })

    raw_safe_clauses = payload.get("safe_clauses", payload.get("safeClauses", []))
    safe_clauses: list[dict] = []
    if isinstance(raw_safe_clauses, list):
        for entry in raw_safe_clauses:
            if isinstance(entry, dict):
                safe_clauses.append({
                    "clause_title": normalize_text(entry.get("clause_title") or entry.get("title")) or "Safe clause",
                    "plain_english_explanation": normalize_text(entry.get("plain_english_explanation") or entry.get("explanation")),
                })

    return risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, requirements

def summarize_contract_text(text: str) -> str:
    chunks = chunk_text(text, settings.summary_chunk_chars)
    if len(chunks) <= 1: return text

    summaries = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        summary_prompt = f"{SUMMARY_CHUNK_PROMPT}\n\nChunk {index}/{total}:\n{chunk}"
        summary_text = _call_with_retry(lambda: gemma_client.generate_content(model=settings.moe_model, contents=summary_prompt, temperature=0.2))
        summaries.append(summary_text.strip())

    combined = "\n\n".join(summaries)
    return _call_with_retry(lambda: gemma_client.generate_content(model=settings.moe_model, contents=f"{SUMMARY_COMBINE_PROMPT}\n\n{combined}", temperature=0.2))

def maybe_summarize_contract_text(text: str) -> str:
    if len(text) <= settings.summary_trigger_chars: return text
    try:
        return summarize_contract_text(text)
    except GemmaClientError:
        return clamp_text(text, settings.summary_chunk_chars)

async def read_contract_text(file: UploadFile | None, text: str | None, base64_image: str | None, base64_mime_type: str | None) -> str:
    trimmed_text = (text or "").strip()
    if trimmed_text: return clamp_text(trimmed_text, settings.max_contract_chars)

    if file is None and not base64_image:
        raise HTTPException(status_code=400, detail="Provide a file, base64 payload, or contract text")

    content_type: str | None = None
    if file is not None:
        content_type = file.content_type
        contents = await file.read()
    else:
        contents, content_type = decode_base64_payload(base64_image or "", base64_mime_type)

    if content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    from google.genai import types
    ocr_contents = [types.Part.from_bytes(data=contents, mime_type=content_type), OCR_PROMPT]
    try:
        extracted_text = _call_with_retry(lambda: gemma_client.generate_content(model=settings.e4b_model, contents=ocr_contents, temperature=0.0))
    except GemmaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return clamp_text(extracted_text, settings.max_contract_chars)

def run_moe_analysis(contract_text: str, contract_type: str, requirements: str | None):
    analysis_contents = f"Contract text:\n{contract_text}"
    analysis_prompt = get_contract_prompt(contract_type, requirements)

    def generate_response(use_mime: bool, inline_prompt: bool) -> str:
        return _call_with_retry(lambda: gemma_client.generate_content(
            model=settings.moe_model,
            contents=analysis_contents if not inline_prompt else f"{analysis_prompt}\n\n{analysis_contents}",
            system_instruction=None if inline_prompt else analysis_prompt,
            temperature=0.2,
            response_mime_type="application/json" if use_mime else None,
        ))

    try:
        analysis_text = generate_response(use_mime=True, inline_prompt=False)
        payload = extract_json_object(analysis_text)
    except Exception:
        analysis_text = generate_response(use_mime=False, inline_prompt=True)
        payload = extract_json_object(analysis_text)

    return parse_analysis_payload(payload)

def explain_red_flag(contract_type: str, flag: dict) -> str:
    clause_title = flag.get("clause_title", "")
    clause_text = flag.get("clause_text") or flag.get("plain_english_explanation") or clause_title
    system_prompt = EXPLANATION_SYSTEM_PROMPT_TEMPLATE.format(contract_label=CONTRACT_LABELS.get(contract_type, contract_type))
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(clause_title=clause_title, clause_text=clause_text)

    return gemma_client.generate_content(model=settings.dense_model, contents=prompt, system_instruction=system_prompt, temperature=0.2).strip()

def generate_negotiation_email(red_flags: list[dict], company_name: str, user_name: str) -> str:
    if not red_flags:
        return "No major red flags detected. You are good to proceed!"
    
    flags_summary = "\n\n".join([
        f"Issue: {f.get('clause_title')}\nProblematic Text: {f.get('clause_text')}\nRequested Change: {f.get('negotiation_tip')}" 
        for f in red_flags
    ])
    
    prompt = NEGOTIATION_EMAIL_PROMPT.format(
        flags_text=flags_summary,
        company_name=company_name or "Hiring Manager",
        user_name=user_name or "[Your Name]"
    )
    
    system_prompt = "You are an elite legal negotiator drafting professional pushback emails."
    return gemma_client.generate_content(
        model=settings.dense_model, 
        contents=prompt, 
        system_instruction=system_prompt, 
        temperature=0.4
    ).strip()

def generate_tldr_summary(
    contract_type: str,
    risk_score: int,
    verdict: str,
    red_flags: list[dict],
    safe_clauses: list[dict],
    requirements: str | None,
) -> str:
    label = CONTRACT_LABELS.get(contract_type, contract_type)
    red_flag_titles = ", ".join([f.get("clause_title", "") for f in red_flags[:2] if f.get("clause_title")])
    safe_titles = ", ".join([s.get("clause_title", "") for s in safe_clauses[:2] if s.get("clause_title")])
    prompt = TLDR_SUMMARY_PROMPT.format(
        contract_label=label,
        risk_score=risk_score,
        verdict=verdict,
        red_flags=red_flag_titles or "None identified",
        safe_clauses=safe_titles or "None identified",
        requirements=(requirements or "None provided").strip(),
    )

    return gemma_client.generate_content(
        model=settings.dense_model,
        contents=prompt,
        temperature=0.3,
    ).strip()

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequirementMatch(BaseModel):
    requirement: str
    met: bool
    explanation: str

class FlaggedClause(BaseModel):
    clause_title: str
    clause_text: str
    plain_english_explanation: str
    negotiation_tip: str
    suggested_rewrite: Optional[str] = None
    severity: str

class SafeClause(BaseModel):
    clause_title: str
    plain_english_explanation: str

class AnalyzeResponse(BaseModel):
    risk_score: int
    compatibility_score: int
    verdict: str
    verdict_reason: str
    summary: Optional[str] = None
    requirement_breakdown: List[RequirementMatch]
    red_flags: List[FlaggedClause]
    safe_clauses: List[SafeClause]
    negotiation_email: Optional[str] = None

class ClauseChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AskClauseRequest(BaseModel):
    contract_type: str
    clause_title: str
    clause_text: str
    question: str
    history: List[ClauseChatMessage] = []

class AskClauseResponse(BaseModel):
    answer: str

def build_fallback_analysis(text: str, requirements: str | None = None) -> tuple:
    flags = []
    lower_text = text.lower()
    
    if "bond" in lower_text or "penalty" in lower_text or "lakh" in lower_text:
        flags.append({
            "clause_title": "Employment Bond / Financial Penalty",
            "clause_text": "Clause containing financial penalties for early exit.",
            "plain_english_explanation": "The contract forces you to pay a penalty if you resign early.",
            "negotiation_tip": "Refuse any financial penalty for leaving the company. Training is a cost of business.",
            "suggested_rewrite": "Employment shall be at-will with no financial penalties for resignation.",
            "severity": "high"
        })
    if "overtime" in lower_text or "weekend" in lower_text or "hours" in lower_text:
        flags.append({
            "clause_title": "Unpaid Overtime Risks",
            "clause_text": "Clause mentioning weekends, overtime, or extended availability.",
            "plain_english_explanation": "You may be forced to work weekends or after-hours without extra pay.",
            "negotiation_tip": "Ask for strict working hours to be defined and a Right to Disconnect.",
            "suggested_rewrite": "Standard working hours are strictly 10 AM to 7 PM. Any required overtime shall be compensated.",
            "severity": "medium"
        })
    
    risk = 85 if flags else 20
    comp = 30 if flags else 80
    verdict = "REJECT" if flags else "ACCEPT"
    reason = "Fallback analysis triggered. Found predatory keywords." if flags else "Fallback analysis found no immediate red flags."

    return (risk, comp, verdict, reason, flags, [{"clause_title": "Fallback", "plain_english_explanation": "Used keyword fallback due to API timeout."}], [])

@app.post("/analyze/stream")
async def analyze_contract_stream(
    contract_type: str = Form(...),
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    requirements: str | None = Form(None),
    company_name: str | None = Form(None),
    user_name: str | None = Form(None),
    base64_image: str | None = Form(None),
    base64_mime_type: str | None = Form(None),
):
    if not settings.api_key:
        return StreamingResponse(iter([sse_event("error", {"detail": "Missing API Key"})]), media_type="text/event-stream")

    try:
        contract_text = await read_contract_text(file, text, base64_image, base64_mime_type)
    except HTTPException as exc:
        return StreamingResponse(iter([sse_event("error", {"detail": str(exc.detail)})]), media_type="text/event-stream")

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield sse_event("status", {"stage": "extract", "message": "Text extracted. Starting analysis..."})
            working_text = maybe_summarize_contract_text(contract_text)
            yield sse_event("status", {"stage": "analyze", "message": "Scoring risk and parsing user requirements... (Gemma MoE)"})

            try:
                (risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, req_breakdown) = run_moe_analysis(working_text, contract_type, requirements)
            except Exception as exc:
                (risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, req_breakdown) = build_fallback_analysis(working_text, requirements)

            yield sse_event("risk_score", {"risk_score": risk_score})
            yield sse_event("compatibility_score", {"compatibility_score": compatibility_score})
            yield sse_event("verdict", {"verdict": verdict, "verdict_reason": verdict_reason})

            try:
                summary_text = _call_with_retry(lambda: generate_tldr_summary(
                    contract_type,
                    risk_score,
                    verdict,
                    red_flags,
                    safe_clauses,
                    requirements,
                ))
            except Exception as exc:
                summary_text = f"Summary unavailable: {str(exc)}"
            yield sse_event("summary", {"summary": summary_text})
            
            for req in req_breakdown:
                yield sse_event("requirement_match", req)

            for clause in safe_clauses:
                yield sse_event("safe_clause", clause)

            total_flags = len(red_flags)
            final_flags: list[dict] = []
            for index, flag in enumerate(red_flags, start=1):
                yield sse_event("status", {"stage": "explain", "message": f"Drafting plain English for clause {index}/{total_flags}... (Gemma Dense)"})
                
                explanation = flag.get("plain_english_explanation", "")
                try:
                    explanation = _call_with_retry(lambda: explain_red_flag(contract_type, flag))
                except Exception:
                    pass

                yield sse_event("red_flag", {
                    "clause_title": flag.get("clause_title"),
                    "clause_text": flag.get("clause_text"),
                    "plain_english_explanation": explanation,
                    "negotiation_tip": flag.get("negotiation_tip"),
                    "suggested_rewrite": flag.get("suggested_rewrite"),
                    "severity": flag.get("severity"),
                })
                final_flags.append({
                    "clause_title": flag.get("clause_title"),
                    "clause_text": flag.get("clause_text"),
                    "plain_english_explanation": explanation,
                    "negotiation_tip": flag.get("negotiation_tip"),
                    "suggested_rewrite": flag.get("suggested_rewrite"),
                    "severity": flag.get("severity"),
                })

            yield sse_event("status", {"stage": "email", "message": "Drafting negotiation email... (Gemma Dense)"})
            email_text = ""
            try:
                email_text = _call_with_retry(lambda: generate_negotiation_email(red_flags, company_name or "", user_name or ""))
            except Exception:
                email_text = "Dear Hiring Manager,\n\nPlease review the clauses regarding bonds and overtime as we discussed.\n\nBest,\n[Your Name]"
                
            yield sse_event("negotiation_email", {"email": email_text})

            report_id = store_report({
                "contract_type": contract_type,
                "analysis": {
                    "risk_score": risk_score,
                    "compatibility_score": compatibility_score,
                    "verdict": verdict,
                    "verdict_reason": verdict_reason,
                    "summary": summary_text,
                    "requirement_breakdown": req_breakdown,
                    "red_flags": final_flags or red_flags,
                    "safe_clauses": safe_clauses,
                    "negotiation_email": email_text,
                },
            })
            yield sse_event("share_report", {"report_id": report_id})

            yield sse_event("done", {"ok": True})
        except Exception as exc:
            yield sse_event("error", {"detail": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.get("/report/{report_id}")
def get_report(report_id: str):
    _prune_reports()
    record = REPORT_STORE.get(report_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found or expired")
    return record["payload"]

@app.post("/ask-clause", response_model=AskClauseResponse)
def ask_clause(payload: AskClauseRequest, request: Request):
    if not settings.api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing API Key")

    client_ip = get_client_ip(request)
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded. Please try again soon.")

    if payload.contract_type not in CONTRACT_TYPE_LABELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported contract type")

    history_turns = payload.history[-6:] if payload.history else []
    history_text = "\n".join(
        [
            f"User: {turn.content}" if turn.role == "user" else f"Assistant: {turn.content}"
            for turn in history_turns
        ]
    ) or "None"

    clause_text = payload.clause_text or payload.clause_title
    prompt = CLAUSE_QA_PROMPT.format(
        contract_label=CONTRACT_LABELS.get(payload.contract_type, payload.contract_type),
        clause_title=payload.clause_title,
        clause_text=clause_text,
        history=history_text,
        question=payload.question.strip(),
    )

    answer = _call_with_retry(lambda: gemma_client.generate_content(
        model=settings.dense_model,
        contents=prompt,
        temperature=0.2,
    )).strip()

    return AskClauseResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)