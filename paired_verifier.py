"""Paired PDF + Excel verification pipeline.

Given one PDF report and one or more Excel statistical tables (BI format), verifies every
quantitative claim in the PDF narrative against the authoritative values in the Excel sources.

No SQL is generated — comparison is a direct dict lookup into the parsed Excel tables.
Unit conversion is applied automatically (e.g. triliun Rp  ↔  miliar Rp).
YoY growth rates are computed from the Excel itself (current month vs same month prior year).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.language_models import BaseChatModel

import re

from excel_parser_bi import BITableData, parse_bi_table
from pdf_extraction import (
    MIN_USEFUL_CHARS,
    extract_text_from_pdf,
    extract_text_from_pdf_vision,
    extract_text_from_pdf_vision_async,
)

_PAGE_MARKER_RE = re.compile(r'\[== Halaman \d+ ==\]')
from schemas import FactVerificationResult, PairedVerificationResponse
from structured_extractor import ExtractedFact, extract_structured_facts, extract_structured_facts_async

logger = logging.getLogger("fact-checker")

# ── Rounding tolerance ────────────────────────────────────────────────────────
# Values in the PDF are printed to 1 decimal place (e.g. 10.415,9 triliun or 10,8%).
# Half of one rounding unit = 0.05 in either scale.
# False positives are more dangerous than false negatives → use strict tolerance.
MATCH_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Excel source container
# ---------------------------------------------------------------------------

@dataclass
class _ExcelSource:
    table: BITableData
    filename: str
    sheet: str

    @property
    def label(self) -> str:
        return f"{self.filename} / {self.sheet}"


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

_UNIT_FACTORS: Dict[Tuple[str, str], float] = {
    # (pdf_unit_normalised, excel_unit_normalised) -> factor such that pdf * factor = excel
    # Rupiah
    ("triliun rp", "miliar rp"): 1_000.0,
    ("miliar rp",  "miliar rp"): 1.0,
    ("triliun rp", "triliun rp"): 1.0,
    ("miliar rp",  "triliun rp"): 0.001,
    # USD — LLM may say "miliar USD", "miliar dolar AS", "miliar dolar", or "billion USD"
    ("miliar usd",      "juta usd"): 1_000.0,
    ("miliar dolar as", "juta usd"): 1_000.0,
    ("miliar dolar",    "juta usd"): 1_000.0,
    ("billion usd",     "juta usd"): 1_000.0,
    ("juta usd",        "juta usd"): 1.0,
    ("miliar usd",      "miliar usd"): 1.0,
    ("miliar dolar as", "miliar dolar as"): 1.0,
    ("juta usd",        "miliar usd"): 0.001,
    ("juta usd",        "miliar dolar as"): 0.001,
}


def _unit_factor(pdf_unit: str, excel_unit: str) -> Optional[float]:
    """Return the multiplier to convert a PDF absolute value to Excel units, or None if unknown."""
    key = (pdf_unit.lower().strip(), excel_unit.lower().strip())
    return _UNIT_FACTORS.get(key)


# ---------------------------------------------------------------------------
# Per-fact comparison (multi-source)
# ---------------------------------------------------------------------------

def _compare_absolute(
    fact: ExtractedFact,
    sources: List[_ExcelSource],
) -> FactVerificationResult:
    """Compare a level/stock claim against multiple Excel sources; first match wins."""
    for src in sources:
        factor = _unit_factor(fact.unit, src.table.unit)
        if factor is None:
            continue  # unit incompatible with this source, try next

        matched_label, excel_raw = src.table.lookup_fuzzy(fact.metric_label, fact.year, fact.month)
        if excel_raw is None:
            continue  # metric not found in this source, try next

        excel_in_pdf_unit = excel_raw / factor
        delta = round(abs(fact.value - excel_in_pdf_unit), 4)
        is_match = delta <= MATCH_TOLERANCE

        return FactVerificationResult(
            metric_label=fact.metric_label,
            matched_excel_label=matched_label,
            matched_excel_source=src.label,
            year=fact.year,
            month=fact.month,
            claim_type=fact.claim_type,
            pdf_value=fact.value,
            pdf_unit=fact.unit,
            excel_value=round(excel_in_pdf_unit, 4),
            excel_unit=fact.unit,
            delta=delta,
            verdict="Entailed" if is_match else "Refuted",
            reasoning=(
                f"PDF: {fact.value} {fact.unit} | "
                f"Excel [{src.label}] ({matched_label}): {round(excel_in_pdf_unit, 4)} {fact.unit} | "
                f"Δ = {delta} → {'within' if is_match else 'exceeds'} tolerance {MATCH_TOLERANCE}"
            ),
            context_quote=fact.context_quote,
            page_number=fact.page_number,
        )

    # Nothing found across all sources — determine best error message
    all_incompatible = all(_unit_factor(fact.unit, src.table.unit) is None for src in sources)
    if all_incompatible:
        units_tried = ", ".join(f"'{src.table.unit}'" for src in sources)
        reasoning = (
            f"Unit conversion from '{fact.unit}' to any of [{units_tried}] is not supported. "
            "Cannot compare."
        )
    else:
        # Check if the metric label exists at all across sources (different period)
        all_available: set = set()
        for src in sources:
            if _unit_factor(fact.unit, src.table.unit) is not None:
                for period in src.table.available_periods(fact.metric_label):
                    all_available.add(period)
        if all_available:
            recent = sorted(all_available, reverse=True)[:6]
            periods_str = ", ".join(f"{m} {y}" for y, m in recent)
            reasoning = (
                f"Metric '{fact.metric_label}' found in Excel but no data for "
                f"{fact.month} {fact.year}. "
                f"Most recent available: {periods_str}."
            )
        else:
            reasoning = (
                f"No data found in any Excel source for metric '{fact.metric_label}' "
                f"at {fact.month} {fact.year}."
            )

    return FactVerificationResult(
        metric_label=fact.metric_label,
        matched_excel_label=None,
        matched_excel_source=None,
        year=fact.year,
        month=fact.month,
        claim_type=fact.claim_type,
        pdf_value=fact.value,
        pdf_unit=fact.unit,
        excel_value=None,
        excel_unit=sources[0].table.unit if sources else "",
        delta=None,
        verdict="Inconclusive",
        reasoning=reasoning,
        context_quote=fact.context_quote,
        page_number=fact.page_number,
    )


def _compare_growth_yoy(
    fact: ExtractedFact,
    sources: List[_ExcelSource],
) -> FactVerificationResult:
    """Compare a yoy growth-rate claim against multiple Excel sources; first match wins."""
    for src in sources:
        matched_label, curr_raw = src.table.lookup_fuzzy(fact.metric_label, fact.year, fact.month)
        if curr_raw is None:
            continue  # metric not found in this source

        prior_year = fact.year - 1
        _, prev_raw = src.table.lookup_fuzzy(fact.metric_label, prior_year, fact.month)

        if prev_raw is None:
            return FactVerificationResult(
                metric_label=fact.metric_label,
                matched_excel_label=matched_label,
                matched_excel_source=src.label,
                year=fact.year,
                month=fact.month,
                claim_type=fact.claim_type,
                pdf_value=fact.value,
                pdf_unit=fact.unit,
                excel_value=None,
                excel_unit="persen_yoy",
                delta=None,
                verdict="Inconclusive",
                reasoning=(
                    f"No data found in Excel [{src.label}] for metric '{fact.metric_label}' "
                    f"at {fact.month} {prior_year} (needed for yoy denominator)."
                ),
                context_quote=fact.context_quote,
                page_number=fact.page_number,
            )

        if prev_raw == 0:
            return FactVerificationResult(
                metric_label=fact.metric_label,
                matched_excel_label=matched_label,
                matched_excel_source=src.label,
                year=fact.year,
                month=fact.month,
                claim_type=fact.claim_type,
                pdf_value=fact.value,
                pdf_unit=fact.unit,
                excel_value=None,
                excel_unit="persen_yoy",
                delta=None,
                verdict="Inconclusive",
                reasoning=f"Prior-year value for '{matched_label}' at {fact.month} {prior_year} is zero; yoy undefined.",
                context_quote=fact.context_quote,
                page_number=fact.page_number,
            )

        computed_yoy = round((curr_raw - prev_raw) / abs(prev_raw) * 100, 4)
        delta = round(abs(fact.value - computed_yoy), 4)
        is_match = delta <= MATCH_TOLERANCE

        return FactVerificationResult(
            metric_label=fact.metric_label,
            matched_excel_label=matched_label,
            matched_excel_source=src.label,
            year=fact.year,
            month=fact.month,
            claim_type=fact.claim_type,
            pdf_value=fact.value,
            pdf_unit="persen_yoy",
            excel_value=computed_yoy,
            excel_unit="persen_yoy",
            delta=delta,
            verdict="Entailed" if is_match else "Refuted",
            reasoning=(
                f"PDF: {fact.value}% yoy | "
                f"Excel [{src.label}] ({matched_label}): {computed_yoy}% yoy "
                f"({curr_raw:.2f} vs {prev_raw:.2f} {src.table.unit}) | "
                f"Δ = {delta}% → {'within' if is_match else 'exceeds'} tolerance {MATCH_TOLERANCE}%"
            ),
            context_quote=fact.context_quote,
            page_number=fact.page_number,
        )

    # Not found in any source
    return FactVerificationResult(
        metric_label=fact.metric_label,
        matched_excel_label=None,
        matched_excel_source=None,
        year=fact.year,
        month=fact.month,
        claim_type=fact.claim_type,
        pdf_value=fact.value,
        pdf_unit=fact.unit,
        excel_value=None,
        excel_unit="persen_yoy",
        delta=None,
        verdict="Inconclusive",
        reasoning=(
            f"No data found in any Excel source for metric '{fact.metric_label}' "
            f"at {fact.month} {fact.year} (needed for yoy numerator)."
        ),
        context_quote=fact.context_quote,
        page_number=fact.page_number,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _deduplicate_facts(facts: List[ExtractedFact]) -> List[ExtractedFact]:
    """Remove duplicate (metric, year, month, claim_type) entries — keep first occurrence."""
    seen = set()
    unique = []
    for f in facts:
        key = (f.metric_label, f.year, f.month, f.claim_type)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


async def verify_paired(
    pdf_bytes: bytes,
    excel_sources: List[Tuple[bytes, str, str]],
    llm: BaseChatModel,
    pdf_filename: str = "report.pdf",
    vision_llm: Optional[BaseChatModel] = None,
) -> PairedVerificationResponse:
    """Verify all quantitative claims in a PDF narrative against one or more Excel statistical tables.

    Args:
        pdf_bytes:      Raw bytes of the PDF report.
        excel_sources:  List of (excel_bytes, sheet_name, filename) tuples.
                        Claims are checked against each source in order; the first source
                        that contains the metric label is used for the verdict.
        llm:            Fallback chat model (used when vision_llm is unavailable).
        pdf_filename:   Display name for the PDF (metadata only).
        vision_llm:     Vision-capable model (Gemini). When provided, used as the PRIMARY
                        model for both PDF image extraction and structured fact extraction,
                        since it handles Indonesian number formats more reliably than Groq.

    Returns:
        PairedVerificationResponse with per-fact verdicts.
    """
    # Step 1: Parse all Excel sources
    parsed_sources: List[_ExcelSource] = []
    for excel_bytes, sheet_name, filename in excel_sources:
        logger.info("Parsing Excel sheet '%s' from '%s'", sheet_name, filename)
        table = parse_bi_table(excel_bytes, sheet_name)
        logger.info("Excel parsed: %d metrics, unit='%s'", len(table.row_labels), table.unit)
        parsed_sources.append(_ExcelSource(table=table, filename=filename, sheet=sheet_name))

    # Per-source label groups with table title context (used by the LLM to understand
    # what generic rows like 'Total' represent in each table).
    source_labels_for_extractor = [
        (f"{src.table.title} / {src.filename}", src.table.row_labels)
        for src in parsed_sources
    ]

    # Combined flat list for de-duplication (required by extract_structured_facts_async signature)
    all_row_labels: List[str] = []
    seen_labels: set = set()
    for src in parsed_sources:
        for label in src.table.row_labels:
            if label not in seen_labels:
                all_row_labels.append(label)
                seen_labels.add(label)

    # Step 2: Extract PDF text — vision fallback when pypdf returns too little.
    logger.info("Extracting text from PDF '%s'", pdf_filename)
    narrative_text = extract_text_from_pdf(pdf_bytes)
    content_chars = len(_PAGE_MARKER_RE.sub('', narrative_text).strip())
    if content_chars < MIN_USEFUL_CHARS:
        if vision_llm is None:
            logger.warning(
                "PDF content too short (%d chars excl. markers) and no vision_llm provided — "
                "pass vision_llm=get_vision_llm() to enable the fallback.",
                content_chars,
            )
        else:
            logger.info(
                "PDF content too short (%d chars excl. markers), falling back to vision extraction",
                content_chars,
            )
            narrative_text = await extract_text_from_pdf_vision_async(pdf_bytes, vision_llm)

    # Step 3: Extract structured facts.
    logger.info("Running structured fact extraction (combined row labels from %d source(s))", len(parsed_sources))
    extraction_primary = vision_llm if vision_llm is not None else llm
    extraction_fallback = llm if vision_llm is not None else None
    raw_facts = await extract_structured_facts_async(
        narrative_text,
        all_row_labels,
        extraction_primary,
        fallback_llm=extraction_fallback,
        source_labels=source_labels_for_extractor,
    )
    facts = _deduplicate_facts(raw_facts)
    logger.info("%d unique facts after deduplication (was %d)", len(facts), len(raw_facts))

    excel_filenames = [src.filename for src in parsed_sources]
    excel_sheets = [src.sheet for src in parsed_sources]
    excel_units = [src.table.unit for src in parsed_sources]

    if not facts:
        return PairedVerificationResponse(
            pdf_filename=pdf_filename,
            excel_filenames=excel_filenames,
            excel_sheets=excel_sheets,
            excel_units=excel_units,
            total_facts=0,
            entailed_count=0,
            refuted_count=0,
            inconclusive_count=0,
            results=[],
        )

    # Step 4: Direct comparison (no SQL) — each fact is checked across all sources
    results: List[FactVerificationResult] = []
    for fact in facts:
        if fact.claim_type == "absolute":
            result = _compare_absolute(fact, parsed_sources)
        elif fact.claim_type == "growth_yoy":
            result = _compare_growth_yoy(fact, parsed_sources)
        else:
            result = FactVerificationResult(
                metric_label=fact.metric_label,
                matched_excel_label=None,
                matched_excel_source=None,
                year=fact.year,
                month=fact.month,
                claim_type=fact.claim_type,
                pdf_value=fact.value,
                pdf_unit=fact.unit,
                excel_value=None,
                excel_unit=parsed_sources[0].table.unit if parsed_sources else "",
                delta=None,
                verdict="Inconclusive",
                reasoning=f"Unknown claim_type '{fact.claim_type}'.",
                context_quote=fact.context_quote,
            )
        results.append(result)

    entailed = sum(1 for r in results if r.verdict == "Entailed")
    refuted = sum(1 for r in results if r.verdict == "Refuted")
    inconclusive = sum(1 for r in results if r.verdict == "Inconclusive")

    return PairedVerificationResponse(
        pdf_filename=pdf_filename,
        excel_filenames=excel_filenames,
        excel_sheets=excel_sheets,
        excel_units=excel_units,
        total_facts=len(results),
        entailed_count=entailed,
        refuted_count=refuted,
        inconclusive_count=inconclusive,
        results=results,
    )
