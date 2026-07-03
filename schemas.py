"""Shared Pydantic request/response models for the fact-checking API."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paired PDF + Excel verification
# ---------------------------------------------------------------------------


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The factual claim to verify against the database.")


class DocumentRequest(BaseModel):
    document: str = Field(..., min_length=1, description="Long-form text to scan for checkable factual claims.")


class VerifyClaimResponse(BaseModel):
    status: Literal["Entailed", "Refuted", "Inconclusive"]
    sql_query_used: str
    reasoning: str


class ClaimVerificationResult(BaseModel):
    claim: str
    status: Literal["Entailed", "Refuted", "Inconclusive", "Error"]
    sql_query_used: Optional[str] = None
    reasoning: str


class VerifyDocumentResponse(BaseModel):
    total_claims: int
    entailed_count: int
    refuted_count: int
    inconclusive_count: int
    error_count: int
    summary: str
    results: List[ClaimVerificationResult]


class UploadExcelSourceResponse(BaseModel):
    filename: str
    n_sheets: int
    n_facts: int
    auto_aggregate: int
    auto_not_aggregate: int
    llm_escalated: int
    defaulted: int


class FactVerificationResult(BaseModel):
    metric_label: str
    matched_excel_label: Optional[str] = None
    matched_excel_source: Optional[str] = None  # "filename / sheet" of the source that matched
    year: int
    month: str
    claim_type: Literal["absolute", "growth_yoy"]
    pdf_value: float
    pdf_unit: str
    excel_value: Optional[float] = None
    excel_unit: str
    delta: Optional[float] = None
    verdict: Literal["Entailed", "Refuted", "Inconclusive"]
    reasoning: str
    context_quote: str
    page_number: Optional[int] = None


class PairedVerificationResponse(BaseModel):
    pdf_filename: str
    excel_filenames: List[str]
    excel_sheets: List[str]
    excel_units: List[str]
    total_facts: int
    entailed_count: int
    refuted_count: int
    inconclusive_count: int
    results: List[FactVerificationResult]


class TableListResponse(BaseModel):
    tables: List[str]


class TableDataResponse(BaseModel):
    table: str
    columns: List[str]
    rows: List[List[Any]]
    total_rows: int
    limit: int
    offset: int
