import base64

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="FinePrint API", description="Analyzing contracts with Gemma 4")

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

@app.get("/")
def read_root():
    return {"message": "FinePrint API is running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_contract(
    file: UploadFile = File(...),
    contract_type: str = Form(...)
):
    # Read the file content and convert to base64 for Gemma 4 routing
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    
    # Placeholder logic to return a simulated response
    _ = base64_image

    return AnalyzeResponse(
        risk_score=85,
        red_flags=[
            FlaggedClause(
                clause_title="Uncapped Liability",
                plain_english_explanation="You are taking on unlimited financial risk if anything goes wrong.",
                severity="high"
            ),
            FlaggedClause(
                clause_title="Vague Non-Compete",
                plain_english_explanation="You cannot work for any competitor anywhere for 3 years after leaving.",
                severity="high"
            )
        ],
        safe_clauses=[
            SafeClause(
                clause_title="Standard Severability",
                plain_english_explanation="If one part of the contract is invalid, the rest still applies."
            )
        ]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
