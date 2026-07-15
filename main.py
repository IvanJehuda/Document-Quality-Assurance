"""
Fact-Checking PoC: Text-to-SQL pipeline + LLM-as-a-Judge.

POST /api/verify-claim
    Request : {"claim": "<text claim>"}
    Response: {"status": "Entailed" | "Refuted" | "Inconclusive", "sql_query_used": "...", "reasoning": "..."}

POST /api/verify-document
    Request : {"document": "<long-form text>"}
    Response: per-claim verdicts plus a short aggregate summary, e.g.
        {
          "total_claims": 4,
          "entailed_count": 1, "refuted_count": 1, "inconclusive_count": 1, "error_count": 1,
          "summary": "4 claim(s) extracted: 1 entailed, 1 refuted, 1 inconclusive, 1 could not be verified due to a pipeline error.",
          "results": [
            {"claim": "...", "status": "Entailed", "sql_query_used": "...", "reasoning": "..."},
            ...
          ]
        }

POST /api/verify-document-pdf
    Request : multipart/form-data, field "file" = a digital PDF (must have a real text layer;
               scanned/image-only PDFs are not supported - no OCR).
    Response: same shape as /api/verify-document.

POST /api/upload-excel-source
    Request : multipart/form-data, field "file" = a .xlsx workbook to add as a new data source.
    Response: {"filename": "...", "n_sheets": 1, "n_facts": 16, "auto_aggregate": 1,
               "auto_not_aggregate": 7, "llm_escalated": 0, "defaulted": 0}
    Parses the workbook (style-aware header/aggregate detection - see excel_ingestion.py) and
    writes it into the same database the other endpoints query, under a long/tidy `excel_facts`
    table. Trusted-uploader endpoint, no access control - intended for the developer/a small team
    adding their own source files, not for arbitrary public users (whoever can call this controls
    what "ground truth" later fact-checks are compared against).

GET /api/tables
    Response: {"tables": ["excel_facts", "indikator_ekonomi", "neraca_dagang"]}
    Lists every table in the database, so the UI's "Lihat Data" tab can browse raw ground-truth
    data - including rows ingested from Excel uploads - without going through SQL.

GET /api/tables/{table_name}?limit=200&offset=0
    Response: {"table": "...", "columns": [...], "rows": [[...], ...], "total_rows": N,
               "limit": 200, "offset": 0}
    Returns a page of raw rows for one table. `table_name` is validated against `/api/tables`'
    own list before being interpolated into SQL (SQLite has no parameterized identifiers).

Pipeline:
    1. Claim extraction - an LLM call pulls out atomic, checkable claims about data tracked in the
                       database from a free-text document (verify-document only).
    2. Text-to-SQL  - a single LLM call translates each claim into one SQL query (given the full
                       DB schema), executes it against a READ-ONLY SQLite connection, and returns
                       the data. Retries once with the error fed back if the query fails.
    3. LLM-as-a-Judge - a second LLM call compares the claim against the actual retrieved data
                       and issues a structured Entailed/Refuted verdict.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from pathlib import Path

from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from db import DB_PATH, fetch_table_rows, get_readonly_db, list_tables
from excel_parser_bi import list_sheet_names
from excel_ingestion import ingest_bytes
from llm_provider import get_llm, get_vision_llm
from orchestrator import verify_document
from paired_verifier import verify_paired
from pdf_extraction import (
    MIN_USEFUL_CHARS,
    extract_narrative_text,
    extract_text_from_pdf,
    extract_text_from_pdf_vision_async,
)
from schemas import (
    ClaimRequest,
    DocumentRequest,
    FactVerificationResult,
    PairedVerificationResponse,
    TableDataResponse,
    TableListResponse,
    UploadExcelSourceResponse,
    VerifyClaimResponse,
    VerifyDocumentResponse,
)
from typo_checker import check_typos
from verifier import build_judge_chain, build_sql_chain, verify_claim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fact-checker")

_PAGE_MARKER_RE = re.compile(r'\[== Halaman \d+ ==\]')

app = FastAPI(
    title="Fact-Checker PoC",
    description="Text-to-SQL + LLM-as-a-Judge fact-checking pipeline over a structured statistical database.",
)

# --- Access control (cookie session, login overlay lives in the shell) -----
# Only the data APIs require a session; the app shell (index.html) is served
# openly so the frontend can render the real UI *blurred* behind a login card
# and reveal it in place once the user logs in (no page redirect). Credentials
# come from the environment (APP_USERNAME / APP_PASSWORD) - never hard-code
# them, since this repo is public. If either is unset, auth is OFF (handy for
# local dev and tests); set both as Secrets on the host to enable it in prod.
#
# /api/login sets a signed, HttpOnly session cookie; the middleware then lets
# /api/* calls through only if that cookie is present and valid (else 401).
# The shell, /health, and the auth endpoints themselves stay open.
SESSION_COOKIE = "dqa_session"
SESSION_MAX_AGE = 12 * 3600  # re-login required after 12 hours
_OPEN_API_PATHS = {"/api/auth-status", "/api/login", "/api/logout"}


def _auth_enabled() -> bool:
    return bool(os.getenv("APP_USERNAME") and os.getenv("APP_PASSWORD"))


def _session_secret() -> bytes:
    """Key used to sign session cookies. Uses SESSION_SECRET if set, otherwise a
    stable value derived from the credentials, so no extra Secret is required
    (changing the password invalidates existing sessions, which is fine)."""
    explicit = os.getenv("SESSION_SECRET")
    if explicit:
        return explicit.encode()
    seed = f"{os.getenv('APP_USERNAME', '')}:{os.getenv('APP_PASSWORD', '')}"
    return hashlib.sha256(seed.encode()).digest()


def _issue_session() -> str:
    payload = f"{os.getenv('APP_USERNAME', '')}:{int(time.time())}"
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _valid_session(cookie: str) -> bool:
    payload, sep, sig = cookie.rpartition(".")
    if not sep:
        return False
    expected = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(sig, expected):
        return False
    user, sep, issued = payload.partition(":")
    if not sep or user != os.getenv("APP_USERNAME", ""):
        return False
    try:
        return (time.time() - int(issued)) < SESSION_MAX_AGE
    except ValueError:
        return False


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    # Everything except the data APIs is open (the shell page, /health, static,
    # and the auth endpoints). Only gate /api/* that isn't an auth endpoint.
    if not _auth_enabled() or not path.startswith("/api/") or path in _OPEN_API_PATHS:
        return await call_next(request)
    if _valid_session(request.cookies.get(SESSION_COOKIE, "")):
        return await call_next(request)
    return JSONResponse({"detail": "Not authenticated"}, status_code=401)
# --------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def ui_root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    """Browser-tab icon — the sidebar brand logo as a standalone SVG."""
    return FileResponse(STATIC_DIR / "favicon.svg")


@app.get("/api/auth-status")
async def auth_status(request: Request) -> dict:
    """Lets the shell decide whether to show the login overlay. Open endpoint."""
    authed = (not _auth_enabled()) or _valid_session(request.cookies.get(SESSION_COOKIE, ""))
    return {"auth_required": _auth_enabled(), "authenticated": authed}


@app.post("/api/login")
async def api_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    ok = (
        _auth_enabled()
        and secrets.compare_digest(username, os.getenv("APP_USERNAME", ""))
        and secrets.compare_digest(password, os.getenv("APP_PASSWORD", ""))
    )
    if not ok:
        return JSONResponse({"ok": False, "detail": "Invalid credentials"}, status_code=401)

    # Behind the host's TLS proxy the internal scheme is http, so trust the
    # forwarded proto to decide the cookie's Secure/SameSite attributes.
    is_https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    # Hugging Face Spaces embeds the app in a cross-site <iframe>, so a SameSite=Lax
    # cookie is withheld on subsequent /api/* requests (login "works" but the next
    # call is 401). SameSite=None lets the cookie ride in that iframe, and it
    # requires Secure. Over plain http (local dev) fall back to Lax without Secure.
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        SESSION_COOKIE,
        _issue_session(),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="none" if is_https else "lax",
        secure=is_https,
    )
    return resp


@app.post("/api/logout")
async def api_logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.post("/api/excel-sheets")
async def excel_sheets_endpoint(file: UploadFile = File(...)) -> dict:
    """List the worksheet names in an uploaded .xls/.xlsx file, for the sheet picker."""
    data = await file.read()
    try:
        return {"sheets": list_sheet_names(data)}
    except Exception as exc:
        logger.warning("Could not list sheets for %r: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=f"Tidak bisa membaca sheet Excel: {exc}") from exc


@app.post("/api/verify-claim", response_model=VerifyClaimResponse)
async def verify_claim_endpoint(payload: ClaimRequest) -> VerifyClaimResponse:
    claim = payload.claim.strip()

    try:
        database = get_readonly_db()
    except FileNotFoundError as exc:
        logger.error("Database file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # connection-level failures (corrupt file, locked, etc.)
        logger.exception("Database connection error")
        raise HTTPException(status_code=500, detail=f"Database connection error: {exc}") from exc

    sql_chain = build_sql_chain(get_llm(temperature=0.0))
    judge_chain = build_judge_chain(get_llm(temperature=0.0))

    try:
        return verify_claim(claim, database, sql_chain, judge_chain)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/verify-document", response_model=VerifyDocumentResponse)
async def verify_document_endpoint(payload: DocumentRequest) -> VerifyDocumentResponse:
    document = payload.document.strip()

    try:
        return verify_document(document)
    except FileNotFoundError as exc:
        logger.error("Database file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document verification pipeline failed")
        raise HTTPException(status_code=502, detail=f"Document verification failed: {exc}") from exc


@app.post("/api/verify-document-pdf", response_model=VerifyDocumentResponse)
async def verify_document_pdf_endpoint(file: UploadFile = File(...)) -> VerifyDocumentResponse:
    pdf_bytes = await file.read()

    try:
        document = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.warning("PDF text extraction failed for upload %r: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    content_chars = len(_PAGE_MARKER_RE.sub("", document).strip())
    if content_chars < MIN_USEFUL_CHARS:
        logger.warning(
            "PDF upload %r yielded only %d usable chars - no text layer / scanned PDF (no OCR here).",
            file.filename, content_chars,
        )
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF. Scanned/image-only PDFs are not "
                   "supported by this endpoint (no OCR).",
        )

    try:
        return verify_document(document)
    except FileNotFoundError as exc:
        logger.error("Database file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document verification pipeline failed")
        raise HTTPException(status_code=502, detail=f"Document verification failed: {exc}") from exc


@app.post("/api/extract-pdf-text")
async def extract_pdf_text_endpoint(
    file: UploadFile = File(...),
    use_vision: bool = False,
) -> dict:
    """Debug endpoint: extract and return raw text from a PDF without running verification.

    Returns the extracted text plus metadata so you can inspect exactly what the
    pipeline sees before fact extraction begins.

    Query params:
      use_vision=false  (default) — pypdf only, fast, no LLM cost
      use_vision=true   — force vision extraction even if pypdf succeeds
    """
    pdf_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal baca PDF: {exc}") from exc

    content_chars = len(_PAGE_MARKER_RE.sub("", text).strip())
    page_markers = _PAGE_MARKER_RE.findall(text)
    method = "pypdf"

    needs_vision = use_vision or content_chars < MIN_USEFUL_CHARS
    if needs_vision:
        try:
            vision_llm = get_vision_llm()
        except RuntimeError as exc:
            if content_chars < MIN_USEFUL_CHARS:
                raise HTTPException(status_code=400, detail=f"PDF butuh vision extraction tapi: {exc}") from exc
            vision_llm = None

        if vision_llm is not None:
            try:
                text = await extract_text_from_pdf_vision_async(pdf_bytes, vision_llm)
                content_chars = len(_PAGE_MARKER_RE.sub("", text).strip())
                page_markers = _PAGE_MARKER_RE.findall(text)
                method = f"vision ({type(vision_llm).__name__})"
            except Exception as exc:
                logger.exception("Vision extraction failed for %r", filename)
                raise HTTPException(status_code=502, detail=f"Vision extraction gagal: {exc}") from exc

    return {
        "filename": filename,
        "method": method,
        "page_count": len(page_markers),
        "char_count": content_chars,
        "text": text,
    }


# --- DISABLED for shared/multi-division deployment ------------------------
# /api/upload-excel-source writes uploaded workbooks into the shared "ground
# truth" database (excel_facts), i.e. the reference data every later fact-check
# is compared against. With the app exposed to other divisions and no auth,
# any caller could alter that reference data ("data poisoning"), so the
# endpoint is intentionally turned off. The everyday PDF+Excel flow
# (/api/verify-paired) is unaffected - it never touches this DB.
#
# To restore (ideally gate behind an admin-only token first), uncomment below.
# The imports it needs (ingest_bytes, UploadExcelSourceResponse) are kept.
#
# @app.post("/api/upload-excel-source", response_model=UploadExcelSourceResponse)
# async def upload_excel_source_endpoint(file: UploadFile = File(...)) -> UploadExcelSourceResponse:
#     file_bytes = await file.read()
#     filename = file.filename or "upload.xlsx"
#
#     try:
#         summary = ingest_bytes(file_bytes, filename, llm=get_llm(temperature=0.0))
#     except Exception as exc:
#         logger.warning("Excel ingestion failed for upload %r: %s", filename, exc)
#         raise HTTPException(status_code=400, detail=f"Could not ingest Excel file: {exc}") from exc
#
#     return UploadExcelSourceResponse(
#         filename=filename,
#         n_sheets=summary.n_sheets,
#         n_facts=summary.n_facts,
#         auto_aggregate=summary.auto_aggregate,
#         auto_not_aggregate=summary.auto_not_aggregate,
#         llm_escalated=summary.llm_escalated,
#         defaulted=summary.defaulted,
#     )
# --------------------------------------------------------------------------


@app.post("/api/verify-paired", response_model=PairedVerificationResponse)
async def verify_paired_endpoint(
    pdf_file: UploadFile = File(..., description="PDF report to verify."),
    excel_file: List[UploadFile] = File(..., description="One or more .xls/.xlsx statistical tables."),
    sheet_names: str = "I.1",
) -> PairedVerificationResponse:
    """Verify all quantitative claims in a PDF report against one or more BI Excel tables.

    No SQL is generated — comparisons are direct lookups into the parsed Excel tables.
    When multiple Excel files are uploaded, claims are checked against each source in order;
    the first source that contains the matching metric label is used for the verdict.

    sheet_names: comma-separated sheet names, one per Excel file (e.g. "I.1,II.1").
    If fewer sheet names than files are provided, the last sheet name is reused for remaining files.
    """
    pdf_bytes = await pdf_file.read()

    sheets = [s.strip() for s in sheet_names.split(",") if s.strip()] or ["I.1"]

    excel_sources = []
    for i, ef in enumerate(excel_file):
        sheet = sheets[i] if i < len(sheets) else sheets[-1]
        excel_sources.append((await ef.read(), sheet, ef.filename or f"table_{i+1}.xls"))

    try:
        vision_llm = None
        try:
            vision_llm = get_vision_llm()
        except RuntimeError:
            logger.warning("Vision LLM not available (GOOGLE_API_KEY missing); vision fallback disabled.")

        # Extract the narrative text once and share it between fact-verification and the
        # typo/grammar check — the vision fallback is an LLM call, so re-extracting per
        # consumer would double its cost and rate-limit exposure for no benefit.
        narrative_text = await extract_narrative_text(pdf_bytes, vision_llm)

        # Prefer Gemini for the typo/grammar escalation call when available - it judges
        # domain jargon (e.g. "kartal", "inflasi") more reliably than the Groq text model,
        # which was observed hallucinating a false-positive correction for "kartal" during
        # real-document testing.
        typo_llm = vision_llm if vision_llm is not None else get_llm(temperature=0.0)

        # Fact verification and the typo/grammar check both only depend on narrative_text,
        # so run them concurrently to overlap their LLM round-trips instead of waiting for
        # the whole fact check to finish before the (single-call) typo pass even starts.
        fact_result, typo_result = await asyncio.gather(
            verify_paired(
                narrative_text=narrative_text,
                excel_sources=excel_sources,
                llm=get_llm(temperature=0.0),
                pdf_filename=pdf_file.filename or "report.pdf",
                vision_llm=vision_llm,
            ),
            asyncio.to_thread(check_typos, narrative_text, typo_llm),
        )
        return fact_result.model_copy(update={"typo_check": typo_result})
    except Exception as exc:
        logger.exception("Paired verification failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tables", response_model=TableListResponse)
async def list_tables_endpoint() -> TableListResponse:
    try:
        return TableListResponse(tables=list_tables())
    except FileNotFoundError as exc:
        logger.error("Database file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/tables/{table_name}", response_model=TableDataResponse)
async def get_table_data_endpoint(table_name: str, limit: int = 200, offset: int = 0) -> TableDataResponse:
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    try:
        valid_tables = list_tables()
    except FileNotFoundError as exc:
        logger.error("Database file missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if table_name not in valid_tables:
        raise HTTPException(status_code=404, detail=f"Tabel '{table_name}' tidak ditemukan.")

    columns, rows, total_rows = fetch_table_rows(table_name, limit=limit, offset=offset)
    return TableDataResponse(
        table=table_name,
        columns=columns,
        rows=[list(row) for row in rows],
        total_rows=total_rows,
        limit=limit,
        offset=offset,
    )


# Accept HEAD as well as GET: uptime pingers (e.g. UptimeRobot's free tier) probe with HEAD,
# and a GET-only route answers HEAD with 405, which flags a false "down" incident even though
# the request still reaches the server.
@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict:
    return {"status": "ok", "database_exists": DB_PATH.exists()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
