"""Paired PDF + Excel verification pipeline.

Given one PDF report and one or more Excel statistical tables (BI format), verifies every
quantitative claim in the PDF narrative against the authoritative values in the Excel sources.

Claims are represented as an OPERATION over one or more (metric, year, month) data points
(see structured_extractor.py for the full operation list: value, yoy_growth, average, sum,
diff, ratio, is_increasing, is_decreasing, is_stable) rather than a fixed claim-type enum, so
claims more complex than a single point-in-time value are supported without a schema change per
pattern. No SQL is generated — every data point is a direct dict lookup into the parsed Excel
tables; the operation itself (average, sum, diff, ratio, monotonic-trend check, unit conversion,
YoY growth) is computed in plain Python, never by an LLM, to avoid hallucination risk in the
comparison step.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from excel_parser_bi import BITableData, parse_bi_table
from schemas import FactVerificationResult, PairedVerificationResponse, PeriodResult
from structured_extractor import (
    ExtractedFact,
    PeriodPoint,
    extract_structured_facts,
    extract_structured_facts_async,
)

logger = logging.getLogger("fact-checker")

# ── Rounding tolerance ────────────────────────────────────────────────────────
# Values in the PDF are printed to 1 decimal place (e.g. 10.415,9 triliun or 10,8%).
# Half of one rounding unit = 0.05 in either scale.
# False positives are more dangerous than false negatives → use strict tolerance.
MATCH_TOLERANCE = 0.05

# Operations whose comparison is a level value in fact.unit's scale, needing unit conversion
# against the matched Excel source's unit. The rest (yoy_growth, ratio, trend checks) either
# compare percentages directly or cancel/ignore units entirely.
_LEVEL_OPS = {"value", "average", "sum", "diff"}


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
# Period resolution against a single Excel source
# ---------------------------------------------------------------------------

def _try_resolve(
    periods: List[PeriodPoint], src: _ExcelSource
) -> Tuple[List[Tuple[str, float]], List[PeriodPoint]]:
    """Look up every period in one Excel source.

    Returns (resolved, missing): resolved has (matched_label, raw_value) for periods found in
    this source (in the same order as `periods`); missing has the PeriodPoint objects that
    weren't found. A fact's periods must ALL resolve from the SAME source (kept unit-consistent) -
    callers should treat a non-empty `missing` as "this source can't be used for this fact".
    """
    resolved: List[Tuple[str, float]] = []
    missing: List[PeriodPoint] = []
    for p in periods:
        label, raw = src.table.lookup_fuzzy(p.metric_label, p.year, p.month)
        if raw is None:
            missing.append(p)
        else:
            resolved.append((label, raw))
    return resolved, missing


def _build_periods(
    fact_periods: List[PeriodPoint], resolved: List[Tuple[str, float]], values: List[float]
) -> List[PeriodResult]:
    return [
        PeriodResult(metric_label=resolved[i][0], year=p.year, month=p.month, excel_value=round(values[i], 4))
        for i, p in enumerate(fact_periods)
    ]


def _numeric_verdict(claimed: float, computed: float) -> Tuple[float, str]:
    delta = round(abs(claimed - computed), 4)
    return delta, ("Entailed" if delta <= MATCH_TOLERANCE else "Refuted")


def _make_result(
    fact: ExtractedFact,
    periods: List[PeriodResult],
    matched_source: Optional[str],
    claimed_value: Optional[float],
    claimed_unit: Optional[str],
    computed_value: Optional[float],
    computed_unit: Optional[str],
    delta: Optional[float],
    verdict: str,
    reasoning: str,
) -> FactVerificationResult:
    return FactVerificationResult(
        operation=fact.operation,
        metric_label=fact.display_label,
        matched_excel_source=matched_source,
        periods=periods,
        claimed_value=claimed_value,
        claimed_unit=claimed_unit,
        computed_value=computed_value,
        computed_unit=computed_unit,
        delta=delta,
        verdict=verdict,
        reasoning=reasoning,
        context_quote=fact.context_quote,
        page_number=fact.page_number,
    )


# ---------------------------------------------------------------------------
# Per-operation computation (all arithmetic here is plain Python, never LLM output)
# ---------------------------------------------------------------------------

def _compute_yoy_growth(fact: ExtractedFact, resolved: List[Tuple[str, float]], src: _ExcelSource) -> FactVerificationResult:
    p = fact.periods[0]
    matched_label, curr_raw = resolved[0]
    matched_source = src.label
    prior_year = p.year - 1
    prior_label, prior_raw = src.table.lookup_fuzzy(matched_label, prior_year, p.month)
    current_period = PeriodResult(metric_label=matched_label, year=p.year, month=p.month, excel_value=round(curr_raw, 4))

    if prior_raw is None:
        return _make_result(
            fact, [current_period], matched_source, fact.claimed_value, "persen_yoy", None, "persen_yoy", None,
            "Inconclusive",
            reasoning=(
                f"No data found in Excel [{matched_source}] for metric '{matched_label}' "
                f"at {p.month} {prior_year} (needed for yoy denominator)."
            ),
        )
    if prior_raw == 0:
        return _make_result(
            fact, [current_period], matched_source, fact.claimed_value, "persen_yoy", None, "persen_yoy", None,
            "Inconclusive",
            reasoning=f"Prior-year value for '{matched_label}' at {p.month} {prior_year} is zero; yoy undefined.",
        )

    computed = round((curr_raw - prior_raw) / abs(prior_raw) * 100, 4)
    delta, verdict = _numeric_verdict(fact.claimed_value, computed)
    periods = [current_period, PeriodResult(metric_label=prior_label, year=prior_year, month=p.month, excel_value=round(prior_raw, 4))]
    return _make_result(
        fact, periods, matched_source, fact.claimed_value, "persen_yoy", computed, "persen_yoy", delta, verdict,
        reasoning=(
            f"PDF: {fact.claimed_value}% yoy | "
            f"Excel [{matched_source}] ({matched_label}): {computed}% yoy "
            f"({curr_raw:.2f} vs {prior_raw:.2f} {src.table.unit}) | "
            f"Δ = {delta}% → {'within' if verdict == 'Entailed' else 'exceeds'} tolerance {MATCH_TOLERANCE}%"
        ),
    )


def _compute_ratio(fact: ExtractedFact, resolved: List[Tuple[str, float]], src: _ExcelSource) -> FactVerificationResult:
    (label_a, raw_a), (label_b, raw_b) = resolved
    matched_source = src.label
    p_a, p_b = fact.periods
    periods = [
        PeriodResult(metric_label=label_a, year=p_a.year, month=p_a.month, excel_value=round(raw_a, 4)),
        PeriodResult(metric_label=label_b, year=p_b.year, month=p_b.month, excel_value=round(raw_b, 4)),
    ]
    if raw_b == 0:
        return _make_result(
            fact, periods, matched_source, fact.claimed_value, fact.unit, None, fact.unit, None, "Inconclusive",
            reasoning=f"Nilai penyebut '{label_b}' bernilai nol; rasio tidak terdefinisi.",
        )
    computed = raw_a / raw_b
    if fact.unit and "persen" in fact.unit.lower():
        computed *= 100
    computed = round(computed, 4)
    delta, verdict = _numeric_verdict(fact.claimed_value, computed)
    return _make_result(
        fact, periods, matched_source, fact.claimed_value, fact.unit, computed, fact.unit, delta, verdict,
        reasoning=(
            f"PDF: rasio = {fact.claimed_value} {fact.unit} | "
            f"Excel [{matched_source}]: {label_a}={round(raw_a, 4)} / {label_b}={round(raw_b, 4)} = {computed} {fact.unit} | "
            f"Δ = {delta} → {'within' if verdict == 'Entailed' else 'exceeds'} tolerance {MATCH_TOLERANCE}"
        ),
    )


def _compute_trend(fact: ExtractedFact, resolved: List[Tuple[str, float]], src: _ExcelSource) -> FactVerificationResult:
    values = [raw for (_, raw) in resolved]
    periods = _build_periods(fact.periods, resolved, values)
    matched_source = src.label

    if fact.operation == "is_increasing":
        ok = all(values[i + 1] >= values[i] for i in range(len(values) - 1))
    elif fact.operation == "is_decreasing":
        ok = all(values[i + 1] <= values[i] for i in range(len(values) - 1))
    else:  # is_stable
        ok = all(abs(values[i + 1] - values[i]) <= MATCH_TOLERANCE for i in range(len(values) - 1))

    verdict = "Entailed" if ok else "Refuted"
    breakdown = ", ".join(f"{p.month} {p.year}={round(v, 4)}" for p, v in zip(fact.periods, values))
    return _make_result(
        fact, periods, matched_source, None, None, None, src.table.unit, None, verdict,
        reasoning=(
            f"Klaim tren '{fact.operation}' untuk '{fact.display_label}' | "
            f"Excel [{matched_source}]: {breakdown} | "
            f"{'sesuai' if ok else 'tidak sesuai'} dengan klaim"
        ),
    )


def _compute_operation(
    fact: ExtractedFact, resolved: List[Tuple[str, float]], factor: float, src: _ExcelSource
) -> FactVerificationResult:
    op = fact.operation
    matched_source = src.label

    if op == "value":
        computed = round(resolved[0][1] / factor, 4)
        periods = _build_periods(fact.periods, resolved, [resolved[0][1] / factor])
        delta, verdict = _numeric_verdict(fact.claimed_value, computed)
        return _make_result(
            fact, periods, matched_source, fact.claimed_value, fact.unit, computed, fact.unit, delta, verdict,
            reasoning=(
                f"PDF: {fact.claimed_value} {fact.unit} | "
                f"Excel [{matched_source}] ({resolved[0][0]}): {computed} {fact.unit} | "
                f"Δ = {delta} → {'within' if verdict == 'Entailed' else 'exceeds'} tolerance {MATCH_TOLERANCE}"
            ),
        )

    if op == "yoy_growth":
        return _compute_yoy_growth(fact, resolved, src)

    if op in ("average", "sum"):
        converted = [raw / factor for (_, raw) in resolved]
        computed = round(sum(converted) / len(converted), 4) if op == "average" else round(sum(converted), 4)
        periods = _build_periods(fact.periods, resolved, converted)
        delta, verdict = _numeric_verdict(fact.claimed_value, computed)
        label = "rata-rata" if op == "average" else "total"
        breakdown = ", ".join(f"{p.month} {p.year}={round(v, 4)}" for p, v in zip(fact.periods, converted))
        return _make_result(
            fact, periods, matched_source, fact.claimed_value, fact.unit, computed, fact.unit, delta, verdict,
            reasoning=(
                f"PDF: {label} = {fact.claimed_value} {fact.unit} | "
                f"Excel [{matched_source}] ({label} dari {breakdown}) = {computed} {fact.unit} | "
                f"Δ = {delta} → {'within' if verdict == 'Entailed' else 'exceeds'} tolerance {MATCH_TOLERANCE}"
            ),
        )

    if op == "diff":
        converted = [raw / factor for (_, raw) in resolved]
        computed = round(converted[-1] - converted[0], 4)
        periods = _build_periods(fact.periods, resolved, converted)
        delta, verdict = _numeric_verdict(fact.claimed_value, computed)
        return _make_result(
            fact, periods, matched_source, fact.claimed_value, fact.unit, computed, fact.unit, delta, verdict,
            reasoning=(
                f"PDF: selisih = {fact.claimed_value} {fact.unit} | "
                f"Excel [{matched_source}]: {round(converted[-1], 4)} - {round(converted[0], 4)} = {computed} {fact.unit} | "
                f"Δ = {delta} → {'within' if verdict == 'Entailed' else 'exceeds'} tolerance {MATCH_TOLERANCE}"
            ),
        )

    if op == "ratio":
        return _compute_ratio(fact, resolved, src)

    return _compute_trend(fact, resolved, src)  # is_increasing / is_decreasing / is_stable


def _inconclusive_result(
    fact: ExtractedFact, best_missing: Optional[List[PeriodPoint]], best_reason: Optional[str]
) -> FactVerificationResult:
    if best_reason:
        reasoning = best_reason
    elif best_missing:
        missing_str = ", ".join(f"{p.metric_label} {p.month} {p.year}" for p in best_missing)
        reasoning = f"Data tidak ditemukan di sumber Excel manapun untuk: {missing_str}."
    else:
        reasoning = f"Tidak ada sumber Excel yang cocok untuk operasi '{fact.operation}' pada '{fact.display_label}'."
    return _make_result(
        fact, [], None, fact.claimed_value, fact.unit, None, fact.unit, None, "Inconclusive", reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Per-fact evaluation (multi-source): tries each source in order, first source with ALL
# periods resolved AND a compatible unit (if the operation needs one) wins.
# ---------------------------------------------------------------------------

def _evaluate_fact(fact: ExtractedFact, sources: List[_ExcelSource]) -> FactVerificationResult:
    needs_unit = fact.operation in _LEVEL_OPS
    best_missing: Optional[List[PeriodPoint]] = None
    best_reason: Optional[str] = None

    for src in sources:
        resolved, missing = _try_resolve(fact.periods, src)
        if missing:
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
            continue

        factor = 1.0
        if needs_unit:
            factor = _unit_factor(fact.unit, src.table.unit)
            if factor is None:
                if best_reason is None:
                    best_reason = (
                        f"Unit conversion from '{fact.unit}' to '{src.table.unit}' "
                        f"([{src.label}]) is not supported. Cannot compare."
                    )
                continue

        return _compute_operation(fact, resolved, factor, src)

    return _inconclusive_result(fact, best_missing, best_reason)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _deduplicate_facts(facts: List[ExtractedFact]) -> List[ExtractedFact]:
    """Remove duplicate (operation, periods) entries — keep first occurrence."""
    seen = set()
    unique = []
    for f in facts:
        key = (f.operation, tuple((p.metric_label, p.year, p.month) for p in f.periods))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


async def verify_paired(
    narrative_text: str,
    excel_sources: List[Tuple[bytes, str, str]],
    llm: BaseChatModel,
    pdf_filename: str = "report.pdf",
    vision_llm: Optional[BaseChatModel] = None,
) -> PairedVerificationResponse:
    """Verify all quantitative claims in a PDF narrative against one or more Excel statistical tables.

    Args:
        narrative_text: Already-extracted PDF narrative text (with [== Halaman N ==] page markers),
                        e.g. from pdf_extraction.extract_narrative_text(). Extraction is the
                        caller's responsibility so the same text can be reused for other checks
                        (e.g. typo_checker.check_typos) without re-running the vision LLM fallback.
        excel_sources:  List of (excel_bytes, sheet_name, filename) tuples.
                        Claims are checked against each source in order; the first source
                        that contains every data point a claim references is used for the verdict.
        llm:            Fallback chat model (used when vision_llm is unavailable).
        pdf_filename:   Display name for the PDF (metadata only).
        vision_llm:     Vision-capable model (Gemini). When provided, used as the PRIMARY
                        model for structured fact extraction, since it handles Indonesian
                        number formats more reliably than Groq.

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

    # Step 2: Extract structured facts.
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

    # Step 3: Direct comparison (no SQL) — each fact is checked across all sources
    results: List[FactVerificationResult] = [_evaluate_fact(fact, parsed_sources) for fact in facts]

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
