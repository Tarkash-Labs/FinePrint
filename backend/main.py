import base64
import binascii
import json
import logging
import os
import uuid
import time
import asyncio
from collections import OrderedDict
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, AsyncIterator, List, Optional, Literal

from config import settings
from gemma_client import GemmaClient, GemmaClientError
from prompts import (
    CONTRACT_TYPE_LABELS,
    OCR_PROMPT,
    CONSOLIDATED_DENSE_PROMPT,
    BATCH_EXPLAIN_PROMPT,
    CLAUSE_QA_PROMPT,
    COMPARE_PROMPT,
    get_contract_prompt,
)
from utils import extract_json_object, clamp_text, chunk_text

logger = logging.getLogger(__name__)

# --- Initialization & Guards ---
APP_START_TIME = time.monotonic()
EXPLANATION_MIN_CHARS = 40

if settings.cors_allow_origins == ("*",) and settings.cors_allow_credentials:
    raise ValueError("FATAL: CORS_ALLOW_CREDENTIALS cannot be true when CORS_ALLOW_ORIGINS is '*'. Browsers will block this.")

app = FastAPI(title="FinePrint API", description="Multimodal Contract Analysis with Gemma 4")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=list(settings.cors_allow_origins), 
    allow_credentials=settings.cors_allow_credentials, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "application/pdf",
}

CONTRACT_LABELS = CONTRACT_TYPE_LABELS

# --- Persistence Setup ---
REPORT_TTL_SECONDS = 60 * 60 * 24 * 7
REPORT_MAX_ITEMS = 200
REPORT_STORE: "OrderedDict[str, dict]" = OrderedDict()

def load_reports_from_disk():
    if os.path.exists(settings.report_db_path):
        try:
            with open(settings.report_db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    REPORT_STORE[k] = v
        except Exception as e:
            logger.error(f"Error loading reports: {e}")

def save_reports_to_disk():
    try:
        with open(settings.report_db_path, "w", encoding="utf-8") as f:
            json.dump(REPORT_STORE, f)
    except Exception as e:
        logger.error(f"Error saving reports: {e}")

load_reports_from_disk()

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
    save_reports_to_disk()
    return report_id

def _prune_reports() -> None:
    now = datetime.now(timezone.utc).timestamp()
    expired = [key for key, value in REPORT_STORE.items() if now - value["created_at"] > REPORT_TTL_SECONDS]
    for key in expired:
        REPORT_STORE.pop(key, None)
    while len(REPORT_STORE) > REPORT_MAX_ITEMS:
        REPORT_STORE.popitem(last=False)

gemma_client = GemmaClient(api_key=settings.api_key)

# --- Rate Limiter Setup ---
ASK_CLAUSE_RATE_LIMIT = 20
ASK_CLAUSE_WINDOW_SECONDS = 3600  # 1 hour
ASK_CLAUSE_BUCKETS: dict[str, list[float]] = {}

def check_rate_limit(ip: str):
    """Evicts old IPs to prevent memory leaks and checks rate limits."""
    now = time.time()
    
    keys_to_delete = []
    for k in list(ASK_CLAUSE_BUCKETS.keys()):
        valid_times = [t for t in ASK_CLAUSE_BUCKETS[k] if now - t < ASK_CLAUSE_WINDOW_SECONDS]
        if not valid_times:
            keys_to_delete.append(k)
        else:
            ASK_CLAUSE_BUCKETS[k] = valid_times
            
    for k in keys_to_delete:
        ASK_CLAUSE_BUCKETS.pop(k, None)

    timestamps = ASK_CLAUSE_BUCKETS.get(ip, [])
    if len(timestamps) >= ASK_CLAUSE_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for /ask-clause")
        
    timestamps.append(now)
    ASK_CLAUSE_BUCKETS[ip] = timestamps

# --- Helper Functions ---

async def _call_with_retry(async_func, max_retries=3, base_delay=1):
    """Async retry logic that yields the event loop during delays."""
    for attempt in range(max_retries):
        try:
            return await async_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logger.warning(f"Gemma API call failed (attempt {attempt + 1}), retrying...")
            await asyncio.sleep(base_delay * (2 ** attempt))

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

    estimated_size = len(trimmed) * 0.75
    if estimated_size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Base64 payload exceeds size limit.")

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
    return str(value).strip() if value is not None else ""

def compute_verdict(risk_score: int, compatibility_score: int) -> str:
    if risk_score >= 61 or compatibility_score <= 30:
        return "REJECT"
    if risk_score <= 30 and compatibility_score >= 70:
        return "ACCEPT"
    return "NEGOTIATE"

def parse_analysis_payload(payload: dict) -> tuple[int, int, str, str, list[dict], list[dict], list[dict]]:
    risk_score = max(0, min(100, int(payload.get("risk_score", payload.get("riskScore", 0)))))
    compatibility_score = max(0, min(100, int(payload.get("compatibility_score", payload.get("compatibilityScore", 50)))))

    verdict = normalize_text(payload.get("verdict") or payload.get("decision")).upper()
    if verdict not in ("ACCEPT", "NEGOTIATE", "REJECT"):
        verdict = compute_verdict(risk_score, compatibility_score)

    verdict_reason = normalize_text(payload.get("verdict_reason") or payload.get("analysis_summary"))
    if not verdict_reason:
        verdict_reason = "Verdict based on overall risk and compatibility."

    raw_reqs = payload.get("requirement_breakdown", [])
    requirements = [
        {
            "requirement": normalize_text(req.get("requirement")),
            "met": bool(req.get("met", False)),
            "explanation": normalize_text(req.get("explanation"))
        }
        for req in raw_reqs if isinstance(req, dict)
    ] if isinstance(raw_reqs, list) else []

    raw_red_flags = payload.get("red_flags", payload.get("redFlags", []))
    red_flags = [
        {
            "clause_title": normalize_text(entry.get("clause_title") or entry.get("title")) or "Untitled clause",
            "clause_text": normalize_text(entry.get("clause_text") or entry.get("clause")),
            "plain_english_explanation": normalize_text(entry.get("plain_english_explanation") or entry.get("explanation")),
            "negotiation_tip": normalize_text(entry.get("negotiation_tip") or "Consult a legal professional."),
            "suggested_rewrite": normalize_text(entry.get("suggested_rewrite") or ""),
            "severity": normalize_text(entry.get("severity")).lower() if normalize_text(entry.get("severity")).lower() in {"high", "medium", "low"} else "medium",
        }
        for entry in raw_red_flags if isinstance(entry, dict)
    ] if isinstance(raw_red_flags, list) else []

    raw_safe_clauses = payload.get("safe_clauses", payload.get("safeClauses", []))
    safe_clauses = [
        {
            "clause_title": normalize_text(entry.get("clause_title") or entry.get("title")) or "Safe clause",
            "plain_english_explanation": normalize_text(entry.get("plain_english_explanation") or entry.get("explanation")),
        }
        for entry in raw_safe_clauses if isinstance(entry, dict)
    ] if isinstance(raw_safe_clauses, list) else []

    return risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, requirements

async def read_contract_text(file: UploadFile | None, text: str | None, base64_image: str | None, base64_mime_type: str | None) -> tuple[str, int]:
    trimmed_text = (text or "").strip()
    if trimmed_text: 
        return clamp_text(trimmed_text, settings.max_contract_chars), 0

    if file is None and not base64_image:
        raise HTTPException(status_code=400, detail="Provide a file, base64 payload, or contract text")

    content_type: str | None = None
    if file is not None:
        content_type = file.content_type
        if content_type not in SUPPORTED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
            
        if file.size and file.size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")
            
        contents = await file.read()
    else:
        contents, content_type = decode_base64_payload(base64_image or "", base64_mime_type)

    if content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    from google.genai import types
    ocr_contents = [types.Part.from_bytes(data=contents, mime_type=content_type), OCR_PROMPT]
    
    t0 = time.perf_counter()
    try:
        extracted_text = await _call_with_retry(
            lambda: asyncio.to_thread(
                gemma_client.generate_content,
                model=settings.e4b_model, 
                contents=ocr_contents, 
                temperature=0.0
            )
        )
    except GemmaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    
    return clamp_text(extracted_text, settings.max_contract_chars), int((time.perf_counter() - t0) * 1000)

async def run_moe_analysis(contract_text: str, contract_type: str, requirements: str | None):
    analysis_contents = f"Contract text:\n{contract_text}"
    analysis_prompt = get_contract_prompt(contract_type, requirements)

    def generate_response(use_mime: bool, inline_prompt: bool) -> str:
        return gemma_client.generate_content(
            model=settings.moe_model,
            contents=analysis_contents if not inline_prompt else f"{analysis_prompt}\n\n{analysis_contents}",
            system_instruction=None if inline_prompt else analysis_prompt,
            temperature=0.2,
            response_mime_type="application/json" if use_mime else None,
        )

    t0 = time.perf_counter()
    try:
        analysis_text = await _call_with_retry(lambda: asyncio.to_thread(generate_response, use_mime=True, inline_prompt=False))
        payload = extract_json_object(analysis_text)
    except Exception:
        analysis_text = await _call_with_retry(lambda: asyncio.to_thread(generate_response, use_mime=False, inline_prompt=True))
        payload = extract_json_object(analysis_text)

    moe_ms = int((time.perf_counter() - t0) * 1000)
    return (*parse_analysis_payload(payload), moe_ms)

async def batch_explain_flags(contract_type: str, red_flags: list[dict]) -> list[dict]:
    """Single dense call to explain all flags at once."""
    flags_to_explain = []
    for flag in red_flags:
        explanation = flag.get("plain_english_explanation", "")
        if len(explanation) < EXPLANATION_MIN_CHARS:
            flags_to_explain.append(flag)
    
    if not flags_to_explain:
        return red_flags
        
    clauses_json = json.dumps([{
        "clause_title": f["clause_title"], 
        "clause_text": f["clause_text"]
    } for f in flags_to_explain], indent=2)
    
    prompt = BATCH_EXPLAIN_PROMPT.format(
        contract_label=CONTRACT_LABELS.get(contract_type, contract_type),
        clauses_json=clauses_json
    )
    
    try:
        response_text = await _call_with_retry(
            lambda: asyncio.to_thread(
                gemma_client.generate_content,
                model=settings.dense_model,
                contents=prompt,
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        explained_flags = extract_json_object(response_text)
        
        explained_map = {f.get("clause_title"): f for f in explained_flags if isinstance(f, dict)}
        for flag in red_flags:
            title = flag.get("clause_title")
            if title in explained_map:
                update_data = explained_map[title]
                flag["plain_english_explanation"] = update_data.get("plain_english_explanation", flag["plain_english_explanation"])
                flag["negotiation_tip"] = update_data.get("negotiation_tip", flag.get("negotiation_tip", ""))
                flag["suggested_rewrite"] = update_data.get("suggested_rewrite", flag.get("suggested_rewrite", ""))
    except Exception as e:
        logger.error(f"Batch explain failed: {e}")
        
    return red_flags

async def run_consolidated_dense(
    contract_type: str, risk_score: int, verdict: str, 
    red_flags: list[dict], company_name: str, user_name: str
) -> tuple[str, str, int]:
    
    if not red_flags:
        return "No major red flags detected. You are good to proceed!", "", 0

    flags_summary = "\n".join([f"- {f.get('clause_title')}: {f.get('plain_english_explanation')}" for f in red_flags])
    
    prompt = CONSOLIDATED_DENSE_PROMPT.format(
        contract_label=CONTRACT_LABELS.get(contract_type, contract_type),
        risk_score=risk_score,
        verdict=verdict,
        red_flags_summary=flags_summary,
        company_name=company_name or "Hiring Manager",
        user_name=user_name or "[Your Name]"
    )

    t0 = time.perf_counter()
    try:
        response_text = await _call_with_retry(
            lambda: asyncio.to_thread(
                gemma_client.generate_content,
                model=settings.dense_model,
                contents=prompt,
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        payload = extract_json_object(response_text)
        dense_ms = int((time.perf_counter() - t0) * 1000)
        return payload.get("tldr", "Summary unavailable."), payload.get("negotiation_email", ""), dense_ms
    except Exception as e:
        logger.error(f"Consolidated dense call failed: {e}")
        dense_ms = int((time.perf_counter() - t0) * 1000)
        return "Summary failed.", "Email failed.", dense_ms

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

# --- Pydantic Models ---
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
            "negotiation_tip": "Refuse any financial penalty for leaving the company.",
            "suggested_rewrite": "Employee may resign with standard notice, without financial penalty for training.",
            "severity": "high"
        })
    
    risk = 85 if flags else 20
    comp = 30 if flags else 80
    verdict = "REJECT" if flags else "ACCEPT"
    reason = "Fallback triggered."
    return (risk, comp, verdict, reason, flags, [{"clause_title": "Fallback", "plain_english_explanation": "Safe"}], [], 0)

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "api_key_set": bool(settings.api_key),
        "models": {
            "ocr": settings.e4b_model,
            "classify": settings.moe_model,
            "explain": settings.dense_model
        },
        "reports_cached": len(REPORT_STORE),
        "uptime_s": time.monotonic() - APP_START_TIME
    }

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

    async def event_stream() -> AsyncIterator[str]:
        timing_metrics = {}
        try:
            yield sse_event("status", {"stage": "extract", "message": f"Extracting text... ({settings.e4b_model})"})
            contract_text, e4b_ms = await read_contract_text(file, text, base64_image, base64_mime_type)
            timing_metrics["e4b_ms"] = e4b_ms
            
            yield sse_event("status", {"stage": "analyze", "message": f"Scoring risk... ({settings.moe_model})"})
            
            try:
                (risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, req_breakdown, moe_ms) = await run_moe_analysis(contract_text, contract_type, requirements)
                timing_metrics["moe_ms"] = moe_ms
            except Exception as exc:
                logger.error(f"MoE Failed: {exc}")
                (risk_score, compatibility_score, verdict, verdict_reason, red_flags, safe_clauses, req_breakdown, moe_ms) = build_fallback_analysis(contract_text, requirements)
                timing_metrics["moe_ms"] = moe_ms

            yield sse_event("risk_score", {"risk_score": risk_score})
            yield sse_event("compatibility_score", {"compatibility_score": compatibility_score})
            yield sse_event("verdict", {"verdict": verdict, "verdict_reason": verdict_reason})

            for req in req_breakdown:
                yield sse_event("requirement_match", req)

            for clause in safe_clauses:
                yield sse_event("safe_clause", clause)

            # Batch explain red flags efficiently
            if red_flags:
                yield sse_event("status", {"stage": "explain", "message": f"Explaining {len(red_flags)} clauses... ({settings.dense_model})"})
                t0 = time.perf_counter()
                final_flags = await batch_explain_flags(contract_type, red_flags)
                timing_metrics["dense_explain_ms"] = int((time.perf_counter() - t0) * 1000)
                
                for flag in final_flags:
                    yield sse_event("red_flag", flag)
            else:
                final_flags = []

            # Consolidated Dense Call (TL;DR + Email) runs AFTER all explanations are finalized
            yield sse_event("status", {"stage": "email", "message": f"Drafting TL;DR & Email... ({settings.dense_model})"})
            await asyncio.sleep(0.5) 
            tldr, email_text, dense_email_ms = await run_consolidated_dense(contract_type, risk_score, verdict, final_flags, company_name or "", user_name or "")
            timing_metrics["dense_email_ms"] = dense_email_ms

            yield sse_event("summary", {"summary": tldr})
            yield sse_event("negotiation_email", {"email": email_text})

            report_id = store_report({
                "contract_type": contract_type,
                "analysis": {
                    "risk_score": risk_score,
                    "compatibility_score": compatibility_score,
                    "verdict": verdict,
                    "verdict_reason": verdict_reason,
                    "summary": tldr,
                    "requirement_breakdown": req_breakdown,
                    "red_flags": final_flags,
                    "safe_clauses": safe_clauses,
                    "negotiation_email": email_text,
                },
            })
            yield sse_event("share_report", {"report_id": report_id})

            yield sse_event("done", {"ok": True, "timing": timing_metrics})
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
async def ask_clause(request: Request, payload: AskClauseRequest):
    check_rate_limit(request.client.host)
    
    if not settings.api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing API Key")

    history_turns = payload.history[-6:] if payload.history else []
    history_text = "\n".join(
        [f"{turn.role.capitalize()}: {turn.content}" for turn in history_turns]
    ) or "None"

    prompt = CLAUSE_QA_PROMPT.format(
        contract_label=CONTRACT_LABELS.get(payload.contract_type, payload.contract_type),
        clause_title=payload.clause_title,
        clause_text=payload.clause_text or payload.clause_title,
        history=history_text,
        question=payload.question.strip(),
    )

    answer = await _call_with_retry(
        lambda: asyncio.to_thread(
            gemma_client.generate_content,
            model=settings.dense_model,
            contents=prompt,
            temperature=0.2,
        )
    )

    return AskClauseResponse(answer=answer.strip())

@app.post("/compare")
async def compare_contracts(
    contract_type: str = Form(...),
    requirements: str | None = Form(None),
    file_v1: UploadFile | None = File(None),
    text_v1: str | None = Form(None),
    file_v2: UploadFile | None = File(None),
    text_v2: str | None = Form(None),
):
    """
    Feature 1: Contract Comparison Mode.
    Upload two versions of the same contract (Original vs Revised),
    extract flags concurrently, and compute the delta.
    """
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="Missing API Key")
        
    try:
        text1, _ = await read_contract_text(file_v1, text_v1, None, None)
        text2, _ = await read_contract_text(file_v2, text_v2, None, None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from files: {e}")
        
    # Run MoE analysis on both concurrently
    task1 = run_moe_analysis(text1, contract_type, requirements)
    task2 = run_moe_analysis(text2, contract_type, requirements)
    
    res1, res2 = await asyncio.gather(task1, task2)

    # Unpack both results — tuple from parse_analysis_payload + moe_ms at index -1
    # Indices: 0=risk, 1=compat, 2=verdict, 3=verdict_reason, 4=red_flags, 5=safe_clauses, 6=req_breakdown, 7=moe_ms
    risk_v1, compat_v1, verdict_v1, reason_v1, flags_v1, safe_v1, reqs_v1, _ = res1
    risk_v2, compat_v2, verdict_v2, reason_v2, flags_v2, safe_v2, reqs_v2, _ = res2

    prompt = COMPARE_PROMPT.format(
        v1_flags=json.dumps([f.get("clause_title") for f in flags_v1]),
        v2_flags=json.dumps([f.get("clause_title") for f in flags_v2])
    )
    
    response_text = await _call_with_retry(
        lambda: asyncio.to_thread(
            gemma_client.generate_content,
            model=settings.dense_model,
            contents=prompt,
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    compare_summary = extract_json_object(response_text)

    # Attach the full individual analysis results so the frontend can compute
    # risk_delta, score_before, score_after, and the severity heatmap
    # without any extra API calls.
    result_v1 = {
        "risk_score": risk_v1,
        "compatibility_score": compat_v1,
        "verdict": verdict_v1,
        "verdict_reason": reason_v1,
        "red_flags": flags_v1,
        "safe_clauses": safe_v1,
        "requirement_breakdown": reqs_v1,
    }
    result_v2 = {
        "risk_score": risk_v2,
        "compatibility_score": compat_v2,
        "verdict": verdict_v2,
        "verdict_reason": reason_v2,
        "red_flags": flags_v2,
        "safe_clauses": safe_v2,
        "requirement_breakdown": reqs_v2,
    }

    return {**compare_summary, "result_v1": result_v1, "result_v2": result_v2}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)