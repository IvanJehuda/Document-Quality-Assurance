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


class PeriodResult(BaseModel):
    """One (metric, year, month) data point used to compute a FactVerificationResult."""
    metric_label: str
    year: int
    month: str
    excel_value: Optional[float] = None


class FactVerificationResult(BaseModel):
    operation: Literal[
        "value", "yoy_growth", "average", "sum", "diff", "ratio",
        "is_increasing", "is_decreasing", "is_stable",
    ]
    metric_label: str  # display label - shared metric name, or "A / B" for cross-metric ops
    matched_excel_source: Optional[str] = None  # "filename / sheet" of the source that matched
    periods: List[PeriodResult] = Field(default_factory=list)
    claimed_value: Optional[float] = None
    claimed_unit: Optional[str] = None
    computed_value: Optional[float] = None
    computed_unit: Optional[str] = None
    delta: Optional[float] = None
    verdict: Literal["Entailed", "Refuted", "Inconclusive"]
    reasoning: str
    context_quote: str
    page_number: Optional[int] = None


class TypoIssue(BaseModel):
    word: str
    start: int
    end: int
    category: Literal["ejaan", "tidak_baku", "grammar"]
    suggestion: str
    explanation: str
    page_number: Optional[int] = None


class TypoCheckResponse(BaseModel):
    total_issues: int
    ejaan_count: int
    tidak_baku_count: int
    grammar_count: int
    summary: str
    issues: List[TypoIssue]


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
    typo_check: Optional[TypoCheckResponse] = None


class TableListResponse(BaseModel):
    tables: List[str]


class TableDataResponse(BaseModel):
    table: str
    columns: List[str]
    rows: List[List[Any]]
    total_rows: int
    limit: int
    offset: int
