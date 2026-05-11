import base64
import binascii
import json
import os
import re

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Iterator, List

from config import settings
from gemma_client import GemmaClient, GemmaClientError
from prompts import (
    CONTRACT_TYPE_LABELS,
    EXPLANATION_PROMPT_TEMPLATE,
    EXPLANATION_SYSTEM_PROMPT_TEMPLATE,
    OCR_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_COMBINE_PROMPT,
    get_contract_prompt,
)
from utils import extract_json_object, clamp_text, chunk_text

app = FastAPI(title="FinePrint API", description="Analyzing contracts with Gemma 4")

SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "application/pdf",
}

CONTRACT_LABELS = CONTRACT_TYPE_LABELS

gemma_client = GemmaClient(
    api_key=settings.api_key,
)

def format_megabytes(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.1f} MB"

def get_upload_size(upload: UploadFile) -> int | None:
    try:
        file_obj = upload.file
        current_pos = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()
        file_obj.seek(current_pos)
        return size
    except Exception:
        return None

def decode_base64_payload(payload: str, mime_type: str | None) -> tuple[bytes, str]:
    trimmed = payload.strip()
    detected_mime = mime_type
    if trimmed.startswith("data:"):
        header, _, data = trimmed.partition(",")
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 payload",
            )
        header_mime = header[5:].split(";")[0]
        detected_mime = header_mime or detected_mime
        trimmed = data

    try:
        decoded = base64.b64decode(trimmed, validate=True)
    except binascii.Error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 payload",
        )

    if not decoded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Base64 payload is empty",
        )

    if not detected_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing MIME type for base64 payload",
        )

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
) -> tuple[int, int, str, str, list[dict], list[dict]]:
    raw_score = payload.get("risk_score", payload.get("riskScore", 0))
    try:
        risk_score = int(raw_score)
    except (TypeError, ValueError):
        risk_score = 0
    risk_score = max(0, min(100, risk_score))

    raw_comp = payload.get("compatibility_score", payload.get("compatibilityScore", 50))
    try:
        compatibility_score = int(raw_comp)
    except (TypeError, ValueError):
        compatibility_score = 50
    compatibility_score = max(0, min(100, compatibility_score))

    verdict = normalize_text(
        payload.get("verdict")
        or payload.get("final_verdict")
        or payload.get("decision")
    ).upper()
    if verdict not in ("ACCEPT", "NEGOTIATE", "REJECT"):
        verdict = compute_verdict(risk_score, compatibility_score)

    verdict_reason = normalize_text(
        payload.get("verdict_reason")
        or payload.get("verdictReason")
        or payload.get("final_verdict_reason")
        or payload.get("decision_reason")
        or payload.get("analysis_summary")
    )
    if not verdict_reason:
        verdict_reason = "Verdict based on overall risk and compatibility."

    raw_red_flags = payload.get("red_flags", payload.get("redFlags", []))
    raw_safe_clauses = payload.get("safe_clauses", payload.get("safeClauses", []))

    red_flags: list[dict] = []
    if isinstance(raw_red_flags, list):
        for entry in raw_red_flags:
            if isinstance(entry, str):
                red_flags.append(
                    {
                        "clause_title": entry.strip() or "Untitled clause",
                        "clause_text": "",
                        "plain_english_explanation": entry.strip(),
                        "severity": "medium",
                    }
                )
                continue
            if not isinstance(entry, dict):
                continue
            red_flags.append(
                {
                    "clause_title": normalize_text(
                        entry.get("clause_title") or entry.get("title")
                    )
                    or "Untitled clause",
                    "clause_text": normalize_text(
                        entry.get("clause_text") or entry.get("clause")
                    ),
                    "plain_english_explanation": normalize_text(
                        entry.get("plain_english_explanation")
                        or entry.get("explanation")
                    ),
                    "severity": normalize_severity(entry.get("severity")),
                }
            )

    safe_clauses: list[dict] = []
    if isinstance(raw_safe_clauses, list):
        for entry in raw_safe_clauses:
            if isinstance(entry, str):
                safe_clauses.append(
                    {
                        "clause_title": entry.strip() or "Safe clause",
                        "plain_english_explanation": entry.strip(),
                    }
                )
                continue
            if not isinstance(entry, dict):
                continue
            safe_clauses.append(
                {
                    "clause_title": normalize_text(
                        entry.get("clause_title") or entry.get("title")
                    )
                    or "Safe clause",
                    "plain_english_explanation": normalize_text(
                        entry.get("plain_english_explanation")
                        or entry.get("explanation")
                    ),
                }
            )

    return (
        risk_score,
        compatibility_score,
        verdict,
        verdict_reason,
        red_flags,
        safe_clauses,
    )

def summarize_contract_text(text: str) -> str:
    chunks = chunk_text(text, settings.summary_chunk_chars)
    if len(chunks) <= 1:
        return text

    summaries: list[str] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        summary_prompt = f"{SUMMARY_CHUNK_PROMPT}\n\nChunk {index}/{total}:\n{chunk}"
        summary_text = gemma_client.generate_content(
            model=settings.moe_model,
            contents=summary_prompt,
            temperature=0.2,
        )
        summaries.append(summary_text.strip())

    combined = "\n\n".join(summaries)
    combine_prompt = f"{SUMMARY_COMBINE_PROMPT}\n\n{combined}"
    return gemma_client.generate_content(
        model=settings.moe_model,
        contents=combine_prompt,
        temperature=0.2,
    )

def maybe_summarize_contract_text(text: str) -> str:
    if len(text) <= settings.summary_trigger_chars:
        return text
    try:
        return summarize_contract_text(text)
    except GemmaClientError:
        return clamp_text(text, settings.summary_chunk_chars)

async def read_contract_text(
    file: UploadFile | None,
    text: str | None,
    base64_image: str | None,
    base64_mime_type: str | None,
) -> str:
    trimmed_text = (text or "").strip()
    if trimmed_text:
        return clamp_text(trimmed_text, settings.max_contract_chars)

    if file is None and not base64_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a file, base64 payload, or contract text",
        )

    content_type: str | None = None
    contents: bytes

    if file is not None:
        content_type = file.content_type
        if not content_type:
            ext = os.path.splitext(file.filename or "")[1].lower()
            ext_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".pdf": "application/pdf",
            }
            content_type = ext_map.get(ext, "application/octet-stream")

        max_upload_bytes = settings.max_upload_bytes
        upload_size = get_upload_size(file)
        if upload_size is not None and upload_size > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "Uploaded file is too large. "
                    f"Max size is {format_megabytes(max_upload_bytes)}."
                ),
            )

        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )
    else:
        contents, content_type = decode_base64_payload(
            base64_image or "",
            base64_mime_type,
        )

    if content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Use PNG, JPG, WEBP, or PDF.",
        )

    max_upload_bytes = settings.max_upload_bytes
    if len(contents) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Uploaded file is too large. "
                f"Max size is {format_megabytes(max_upload_bytes)}."
            ),
        )

    from google.genai import types
    ocr_contents = [
        types.Part.from_bytes(data=contents, mime_type=content_type),
        OCR_PROMPT,
    ]

    try:
        extracted_text = gemma_client.generate_content(
            model=settings.e4b_model,
            contents=ocr_contents,
            temperature=0.0,
        )
    except GemmaClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    return clamp_text(extracted_text, settings.max_contract_chars)

def run_moe_analysis(
    contract_text: str,
    contract_type: str,
    requirements: str | None,
) -> tuple[int, int, str, str, list[dict], list[dict]]:
    analysis_contents = f"Contract text:\n{contract_text}"
    analysis_prompt = get_contract_prompt(contract_type, requirements)

    try:
        analysis_text = gemma_client.generate_content(
            model=settings.moe_model,
            contents=analysis_contents,
            system_instruction=analysis_prompt,
            temperature=0.2,
        )
    except GemmaClientError as exc:
        error_text = str(exc)
        if "INTERNAL" in error_text or "500" in error_text:
            analysis_text = gemma_client.generate_content(
                model=settings.moe_model,
                contents=f"{analysis_prompt}\n\n{analysis_contents}",
                temperature=0.2,
            )
        else:
            raise

    payload = extract_json_object(analysis_text)
    (
        risk_score,
        compatibility_score,
        verdict,
        verdict_reason,
        red_flags,
        safe_clauses,
    ) = parse_analysis_payload(payload)
    if not red_flags and not safe_clauses:
        raise ValueError("Empty analysis payload")
    if risk_score >= 50 and not red_flags:
        raise ValueError("High risk without red flags")
    return (
        risk_score,
        compatibility_score,
        verdict,
        verdict_reason,
        red_flags,
        safe_clauses,
    )

def explain_red_flag(contract_type: str, flag: dict) -> str:
    clause_title = flag.get("clause_title", "")
    clause_text = flag.get("clause_text") or flag.get("plain_english_explanation") or clause_title
    label = CONTRACT_LABELS.get(contract_type, contract_type)
    system_prompt = EXPLANATION_SYSTEM_PROMPT_TEMPLATE.format(contract_label=label)
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        clause_title=clause_title,
        clause_text=clause_text,
    )

    response = gemma_client.generate_content(
        model=settings.dense_model,
        contents=prompt,
        system_instruction=system_prompt,
        temperature=0.2,
    )
    return response.strip()

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FlaggedClause(BaseModel):
    clause_title: str
    plain_english_explanation: str
    severity: str

class SafeClause(BaseModel):
    clause_title: str
    plain_english_explanation: str

class AnalyzeResponse(BaseModel):
    risk_score: int
    compatibility_score: int
    verdict: str
    verdict_reason: str
    red_flags: List[FlaggedClause]
    safe_clauses: List[SafeClause]

def build_fallback_analysis(text: str, requirements: str | None = None) -> AnalyzeResponse:
    lowered = text.lower()
    red_flags: list[FlaggedClause] = []
    safe_clauses: list[SafeClause] = []
    risk_score = 20

    def add_flag(title: str, explanation: str, severity: str, weight: int) -> None:
        nonlocal risk_score
        red_flags.append(
            FlaggedClause(
                clause_title=title,
                plain_english_explanation=explanation,
                severity=severity,
            )
        )
        risk_score += weight

    def add_safe(title: str, explanation: str, weight: int = 5) -> None:
        nonlocal risk_score
        safe_clauses.append(
            SafeClause(
                clause_title=title,
                plain_english_explanation=explanation,
            )
        )
        risk_score -= weight

    if "service agreement" in lowered or "bond" in lowered:
        add_flag(
            "Service agreement or bond",
            "Requires signing a service agreement or bond, which can limit flexibility.",
            "high",
            25,
        )

    duration_match = re.search(r"\b(\d{1,2})\s*(year|years|month|months)\b", lowered)
    if duration_match and ("service agreement" in lowered or "bond" in lowered):
        duration = duration_match.group(0)
        add_flag(
            "Minimum service duration",
            f"Mentions a required commitment of {duration}; leaving early may trigger penalties.",
            "high",
            20,
        )

    if any(keyword in lowered for keyword in ["penalty", "repay", "repayment", "training cost"]):
        add_flag(
            "Repayment or penalty clauses",
            "Mentions repayment or penalties; confirm the exact amounts and triggers.",
            "high",
            20,
        )

    if any(keyword in lowered for keyword in ["night shift", "rotational shift"]):
        add_flag(
            "Shift requirements",
            "Mentions night or rotational shifts, which can affect work-life balance.",
            "low",
            5,
        )

    if "apply by" in lowered or "deadline" in lowered:
        add_flag(
            "Short deadline",
            "Contains a short application deadline; confirm timeline and requirements.",
            "low",
            5,
        )

    if "backlog" in lowered:
        add_flag(
            "Eligibility restrictions",
            "States restrictions on backlogs; ensure you meet eligibility criteria.",
            "low",
            5,
        )

    if any(keyword in lowered for keyword in ["ctc", "lpa", "salary"]):
        add_safe(
            "Compensation stated",
            "The posting specifies salary or CTC details.",
        )

    if "insurance" in lowered:
        add_safe(
            "Insurance benefit",
            "Mentions health or accident insurance benefits.",
        )

    if "5 days" in lowered or "5-day" in lowered:
        add_safe(
            "Work week specified",
            "Defines a five-day work week.",
        )

    if "learning" in lowered or "training" in lowered:
        add_safe(
            "Learning support",
            "Mentions learning or training opportunities.",
        )

    if not red_flags and not safe_clauses:
        add_flag(
            "Manual review recommended",
            "Could not reliably extract structured clauses; review the text manually.",
            "low",
            5,
        )

    requirements_text = normalize_text(requirements).lower()
    compatibility_score = 50
    if requirements_text:
        if (
            any(term in requirements_text for term in ["leave", "switch", "short term", "short-term", "1 year", "12 month", "12 months"])
            and any(term in lowered for term in ["service agreement", "bond", "minimum service"])
        ):
            compatibility_score -= 20
        if "side project" in requirements_text and any(
            term in lowered for term in ["ip assignment", "inventions", "work for hire", "ownership"]
        ):
            compatibility_score -= 15
        if "remote" in requirements_text and any(term in lowered for term in ["on-site", "onsite", "office"]):
            compatibility_score -= 10

    risk_score = max(0, min(100, risk_score))
    compatibility_score = max(0, min(100, compatibility_score))
    verdict = compute_verdict(risk_score, compatibility_score)
    if requirements_text:
        verdict_reason = "Compatibility is estimated from fallback rules; review clauses against your goals."
    else:
        verdict_reason = "No personal requirements provided; verdict based on risk score."

    return AnalyzeResponse(
        risk_score=risk_score,
        compatibility_score=compatibility_score,
        verdict=verdict,
        verdict_reason=verdict_reason,
        red_flags=red_flags,
        safe_clauses=safe_clauses,
    )

@app.get("/")
def read_root():
    return {"message": "FinePrint API is running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_contract(
    contract_type: str = Form(...),
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    requirements: str | None = Form(None),
    base64_image: str | None = Form(None),
    base64_mime_type: str | None = Form(None),
):
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMMA_API_KEY is not configured",
        )

    if contract_type not in CONTRACT_TYPE_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported contract type",
        )
    contract_text = await read_contract_text(
        file=file,
        text=text,
        base64_image=base64_image,
        base64_mime_type=base64_mime_type,
    )
    contract_text = maybe_summarize_contract_text(contract_text)

    try:
        (
            risk_score,
            compatibility_score,
            verdict,
            verdict_reason,
            red_flags,
            safe_clauses,
        ) = run_moe_analysis(
            contract_text,
            contract_type,
            requirements,
        )
    except Exception:
        return build_fallback_analysis(contract_text, requirements)

    final_red_flags: list[FlaggedClause] = []
    for flag in red_flags:
        explanation = flag.get("plain_english_explanation", "")
        try:
            explanation = explain_red_flag(contract_type, flag)
        except GemmaClientError:
            if not explanation:
                explanation = "Could not generate a plain-English explanation."

        final_red_flags.append(
            FlaggedClause(
                clause_title=flag.get("clause_title", "Untitled clause"),
                plain_english_explanation=explanation,
                severity=flag.get("severity", "medium"),
            )
        )

    final_safe_clauses: list[SafeClause] = []
    for clause in safe_clauses:
        explanation = clause.get("plain_english_explanation") or "Appears reasonable."
        final_safe_clauses.append(
            SafeClause(
                clause_title=clause.get("clause_title", "Safe clause"),
                plain_english_explanation=explanation,
            )
        )

    return AnalyzeResponse(
        risk_score=risk_score,
        compatibility_score=compatibility_score,
        verdict=verdict,
        verdict_reason=verdict_reason,
        red_flags=final_red_flags,
        safe_clauses=final_safe_clauses,
    )

@app.post("/analyze/stream")
async def analyze_contract_stream(
    contract_type: str = Form(...),
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    requirements: str | None = Form(None),
    base64_image: str | None = Form(None),
    base64_mime_type: str | None = Form(None),
):
    if not settings.api_key:
        return StreamingResponse(
            iter([
                sse_event(
                    "error",
                    {"detail": "GEMMA_API_KEY is not configured"},
                )
            ]),
            media_type="text/event-stream",
        )

    if contract_type not in CONTRACT_TYPE_LABELS:
        return StreamingResponse(
            iter([
                sse_event(
                    "error",
                    {"detail": "Unsupported contract type"},
                )
            ]),
            media_type="text/event-stream",
        )

    try:
        contract_text = await read_contract_text(
            file=file,
            text=text,
            base64_image=base64_image,
            base64_mime_type=base64_mime_type,
        )
    except HTTPException as exc:
        return StreamingResponse(
            iter([
                sse_event(
                    "error",
                    {"detail": str(exc.detail)},
                )
            ]),
            media_type="text/event-stream",
        )

    def event_stream() -> Iterator[str]:
        try:
            if len(contract_text) > settings.summary_trigger_chars:
                yield sse_event(
                    "status",
                    {
                        "stage": "summarize",
                        "message": "Summarizing long contract...",
                    },
                )
            working_text = maybe_summarize_contract_text(contract_text)

            yield sse_event(
                "status",
                {
                    "stage": "classify",
                    "message": "Scoring risk and identifying clauses...",
                },
            )

            try:
                (
                    risk_score,
                    compatibility_score,
                    verdict,
                    verdict_reason,
                    red_flags,
                    safe_clauses,
                ) = run_moe_analysis(
                    working_text,
                    contract_type,
                    requirements,
                )
            except Exception:
                fallback = build_fallback_analysis(working_text, requirements)
                yield sse_event("risk_score", {"risk_score": fallback.risk_score})
                yield sse_event(
                    "compatibility_score",
                    {"compatibility_score": fallback.compatibility_score},
                )
                yield sse_event(
                    "verdict",
                    {
                        "verdict": fallback.verdict,
                        "verdict_reason": fallback.verdict_reason,
                    },
                )
                for clause in fallback.safe_clauses:
                    yield sse_event(
                        "safe_clause",
                        {
                            "clause_title": clause.clause_title,
                            "plain_english_explanation": clause.plain_english_explanation,
                        },
                    )
                for flag in fallback.red_flags:
                    yield sse_event(
                        "red_flag",
                        {
                            "clause_title": flag.clause_title,
                            "plain_english_explanation": flag.plain_english_explanation,
                            "severity": flag.severity,
                        },
                    )
                yield sse_event("done", {"ok": True})
                return

            yield sse_event("risk_score", {"risk_score": risk_score})
            yield sse_event(
                "compatibility_score",
                {"compatibility_score": compatibility_score},
            )
            yield sse_event(
                "verdict",
                {"verdict": verdict, "verdict_reason": verdict_reason},
            )
            for clause in safe_clauses:
                yield sse_event(
                    "safe_clause",
                    {
                        "clause_title": clause.get("clause_title"),
                        "plain_english_explanation": clause.get("plain_english_explanation"),
                    },
                )

            total_flags = len(red_flags)
            for index, flag in enumerate(red_flags, start=1):
                yield sse_event(
                    "status",
                    {
                        "stage": "explain",
                        "message": f"Explaining clause {index}/{total_flags}...",
                    },
                )

                explanation = flag.get("plain_english_explanation", "")
                try:
                    explanation = explain_red_flag(contract_type, flag)
                except GemmaClientError:
                    if not explanation:
                        explanation = "Could not generate a plain-English explanation."

                yield sse_event(
                    "red_flag",
                    {
                        "clause_title": flag.get("clause_title"),
                        "plain_english_explanation": explanation,
                        "severity": flag.get("severity"),
                    },
                )

            yield sse_event("done", {"ok": True})
        except Exception as exc:
            yield sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
