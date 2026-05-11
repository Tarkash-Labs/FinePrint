import base64
import binascii
import os
import re

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List

from config import settings
from gemma_client import GemmaClient, GemmaClientError
from prompts import (
    OCR_PROMPT,
    EMPLOYMENT_BOND_SYSTEM_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_COMBINE_PROMPT,
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

CONTRACT_PROMPTS = {
    "employment": EMPLOYMENT_BOND_SYSTEM_PROMPT,
}

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
    red_flags: List[FlaggedClause]
    safe_clauses: List[SafeClause]

def build_fallback_analysis(text: str) -> AnalyzeResponse:
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

    risk_score = max(0, min(100, risk_score))
    return AnalyzeResponse(
        risk_score=risk_score,
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
    base64_image: str | None = Form(None),
    base64_mime_type: str | None = Form(None),
):
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMMA_API_KEY is not configured",
        )

    if contract_type not in CONTRACT_PROMPTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only employment bond analysis is supported right now",
        )

    trimmed_text = (text or "").strip()
    if trimmed_text:
        contract_text = clamp_text(trimmed_text, settings.max_contract_chars)
    else:
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
            OCR_PROMPT
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

        contract_text = clamp_text(extracted_text, settings.max_contract_chars)

    if len(contract_text) > settings.summary_trigger_chars:
        try:
            contract_text = summarize_contract_text(contract_text)
        except GemmaClientError:
            contract_text = clamp_text(contract_text, settings.summary_chunk_chars)

    analysis_contents = f"Contract text:\n{contract_text}"
    analysis_prompt = f"{CONTRACT_PROMPTS[contract_type]}\n\n{analysis_contents}"

    try:
        analysis_text = gemma_client.generate_content(
            model=settings.moe_model,
            contents=analysis_contents,
            system_instruction=CONTRACT_PROMPTS[contract_type],
            temperature=0.2,
        )
    except GemmaClientError as exc:
        error_text = str(exc)
        if "INTERNAL" in error_text or "500" in error_text:
            # Retry by inlining the system prompt to avoid model/system config issues.
            try:
                analysis_text = gemma_client.generate_content(
                    model=settings.moe_model,
                    contents=analysis_prompt,
                    temperature=0.2,
                )
            except GemmaClientError:
                return build_fallback_analysis(contract_text)
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=error_text,
            )

    try:
        payload = extract_json_object(analysis_text)
    except Exception:
        return build_fallback_analysis(contract_text)

    try:
        return AnalyzeResponse(**payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Invalid JSON schema: {exc}",
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
