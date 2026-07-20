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
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from excel_parser_bi import BITableData, parse_bi_table
from table_parser_generic import parse_generic_table
from table_parser_llm import parse_table_with_llm
from schemas import (
    FactVerificationResult,
    PairedVerificationResponse,
    PeriodResult,
    TableSuggestion,
)
from structured_extractor import (
    ExtractedFact,
    PeriodPoint,
    extract_structured_facts,
    extract_structured_facts_async,
)

logger = logging.getLogger("fact-checker")

# ── Progress reporting ────────────────────────────────────────────────────────
# A callback receiving one stage event dict per pipeline step, so a streaming caller (see
# main.py's /api/verify-paired-stream) can tell the user which step is running during the
# ~40s wait instead of showing an opaque spinner. Events look like:
#   {"type": "stage", "stage": "excel", "status": "running", "current": 1, "total": 2,
#    "detail": "TABEL1_1.xls / I.1"}
# `status` is "running" or "done". Called synchronously on the event loop thread, so an
# implementation must never block or await. None (the default) disables reporting.
ProgressCb = Optional[Callable[[Dict[str, Any]], None]]

# ── Rounding tolerance ────────────────────────────────────────────────────────
# Values in the PDF are printed to 1 decimal place (e.g. 10.415,9 triliun or 10,8%).
# Half of one rounding unit = 0.05 in either scale.
# False positives are more dangerous than false negatives → use strict tolerance.
MATCH_TOLERANCE = 0.05

# Operations whose comparison is a level value in fact.unit's scale, needing unit conversion
# against the matched Excel source's unit. The rest (yoy_growth, ratio, trend checks) either
# compare percentages directly or cancel/ignore units entirely.
_LEVEL_OPS = {"value", "average", "sum", "diff"}

# Operations that are only meaningful along a time axis: yoy needs the prior-year point,
# trend checks need chronological ordering. A claim whose data points are categorical
# (col_label instead of year/month) can never satisfy them.
_TEMPORAL_ONLY_OPS = {"yoy_growth", "is_increasing", "is_decreasing", "is_stable"}


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


# Decimal scale words and currency tokens for units the explicit _UNIT_FACTORS table does not
# list. Parsing these keeps unit handling deterministic for arbitrary sources (the generic
# parser can encounter any wording) without enumerating every pair by hand.
_SCALE_WORDS: Dict[str, float] = {
    "ribu": 1e3, "thousand": 1e3,
    "juta": 1e6, "million": 1e6,
    "miliar": 1e9, "milyar": 1e9, "billion": 1e9,
    "triliun": 1e12, "trillion": 1e12,
}
_CURRENCY_TOKENS: Dict[str, str] = {
    "rp": "rp", "rupiah": "rp", "idr": "rp",
    "usd": "usd", "dolar": "usd", "dollar": "usd",
}


def _parse_scale_unit(unit: str) -> Optional[Tuple[float, Optional[str]]]:
    """Parse a level unit into (decimal scale, currency token or None).

    'juta Rp' -> (1e6, 'rp') | 'Rp' -> (1.0, 'rp') | 'miliar' -> (1e9, None).
    Returns None for percentages and units with neither a scale word nor a currency
    ('unit', 'buah'), where scaling would be meaningless.
    """
    if "%" in unit:
        return None
    words = re.findall(r"[a-z]+", unit.lower())
    if "persen" in words:
        return None
    scale, currency, saw_scale = 1.0, None, False
    for w in words:
        if w in _SCALE_WORDS:
            scale *= _SCALE_WORDS[w]
            saw_scale = True
        elif w in _CURRENCY_TOKENS and currency is None:
            currency = _CURRENCY_TOKENS[w]
    if not saw_scale and currency is None:
        return None
    return scale, currency


def _unit_factor(pdf_unit: Optional[str], excel_unit: Optional[str]) -> Optional[float]:
    """Return the multiplier to convert a PDF absolute value to Excel units, or None if unknown."""
    if not pdf_unit or not excel_unit:
        return None
    key = (pdf_unit.lower().strip(), excel_unit.lower().strip())
    factor = _UNIT_FACTORS.get(key)
    if factor is not None:
        return factor
    # Identical units (after normalisation) never need conversion, whatever they are —
    # keeps the explicit table for CROSS-unit pairs only.
    if key[0] == key[1]:
        return 1.0
    # General fallback: both units parse to a decimal scale of the SAME currency
    # ('ribu Rp' vs 'miliar Rp' -> 1e3/1e9). pdf * factor = excel, hence the ratio.
    pdf_parsed, excel_parsed = _parse_scale_unit(key[0]), _parse_scale_unit(key[1])
    if pdf_parsed and excel_parsed and pdf_parsed[1] and pdf_parsed[1] == excel_parsed[1]:
        return pdf_parsed[0] / excel_parsed[0]
    return None


# Trailing parenthetical of a categorical column name, where such tables usually declare
# the column's unit: 'Harga (Rp)' -> 'Rp', 'Omzet (juta Rp)' -> 'juta Rp'.
_COL_UNIT_RE = re.compile(r"\(([^)]+)\)\s*$")


def _col_unit(col_label: str) -> Optional[str]:
    m = _COL_UNIT_RE.search(col_label)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Period resolution against a single Excel source
# ---------------------------------------------------------------------------

def _try_resolve(
    periods: List[PeriodPoint], src: _ExcelSource
) -> Tuple[List[Tuple[str, float]], List[PeriodPoint], List[Optional[str]]]:
    """Look up every period in one Excel source.

    Returns (resolved, missing, col_units): resolved has (matched_label, raw_value) for periods
    found in this source (in the same order as `periods`); missing has the PeriodPoint objects
    that weren't found; col_units has, per resolved categorical point, the unit declared in the
    matched column's trailing parenthetical ('Harga (Rp)' -> 'Rp'), or None. A fact's periods
    must ALL resolve from the SAME source (kept unit-consistent) - callers should treat a
    non-empty `missing` as "this source can't be used for this fact".

    Each point only resolves against a source of the matching axis kind: temporal points
    (year+month) against temporal tables, categorical points (col_label) against categorical
    tables. A mismatched source simply counts the point as missing so the next source is tried.
    """
    resolved: List[Tuple[str, float]] = []
    missing: List[PeriodPoint] = []
    col_units: List[Optional[str]] = []
    for p in periods:
        if p.col_label is not None:
            if src.table.axis_type != "categorical":
                missing.append(p)
                continue
            row, col, raw = src.table.lookup_cell_fuzzy(p.metric_label, p.col_label)
            if raw is None:
                missing.append(p)
            else:
                resolved.append((f"{row} — {col}", raw))
                col_units.append(_col_unit(col))
        else:
            if src.table.axis_type != "temporal":
                missing.append(p)
                continue
            label, raw = src.table.lookup_fuzzy(p.metric_label, p.year, p.month)
            if raw is None:
                missing.append(p)
            else:
                resolved.append((label, raw))
                col_units.append(None)
    return resolved, missing, col_units


def _build_periods(
    fact_periods: List[PeriodPoint], resolved: List[Tuple[str, float]], values: List[float]
) -> List[PeriodResult]:
    return [
        PeriodResult(
            metric_label=resolved[i][0], year=p.year, month=p.month,
            col_label=p.col_label, excel_value=round(values[i], 4),
        )
        for i, p in enumerate(fact_periods)
    ]


def _point_desc(p: PeriodPoint) -> str:
    """Short human-readable identity of a data point for reasoning strings."""
    return p.col_label if p.col_label is not None else f"{p.month} {p.year}"


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
        breakdown = ", ".join(f"{_point_desc(p)}={round(v, 4)}" for p, v in zip(fact.periods, converted))
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
        missing_str = ", ".join(
            f"{p.metric_label} ({p.col_label})" if p.col_label is not None
            else f"{p.metric_label} {p.month} {p.year}"
            for p in best_missing
        )
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
    # A time-only operation over categorical data points can never be computed — fail fast
    # with an explanation instead of scanning sources for data that cannot qualify.
    if fact.operation in _TEMPORAL_ONLY_OPS and any(p.col_label is not None for p in fact.periods):
        return _make_result(
            fact, [], None, fact.claimed_value, fact.unit, None, fact.unit, None, "Inconclusive",
            reasoning=(
                f"Operasi '{fact.operation}' memerlukan data deret waktu (tahun/bulan), "
                f"tetapi klaim ini merujuk atribut non-waktu ('{fact.display_label}')."
            ),
        )

    needs_unit = fact.operation in _LEVEL_OPS
    best_missing: Optional[List[PeriodPoint]] = None
    best_reason: Optional[str] = None

    for src in sources:
        resolved, missing, col_units = _try_resolve(fact.periods, src)
        if missing:
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
            continue

        factor = 1.0
        if needs_unit:
            # Categorical tables usually declare their unit in the COLUMN name
            # ('Harga (Rp)') rather than a table-wide unit row — prefer the matched
            # column's declared unit when there is one.
            excel_unit = src.table.unit
            if src.table.axis_type == "categorical":
                excel_unit = next((u for u in col_units if u), None) or src.table.unit
            factor = _unit_factor(fact.unit, excel_unit)
            if factor is None and src.table.axis_type == "categorical" and not excel_unit:
                # Source declares no unit anywhere. Spreadsheet cells without a declared
                # unit hold BASE-currency numbers (an item list stores 7500000, not 7.5) —
                # so normalise the CLAIM to base units via its own scale word: a claim of
                # 7,5 'juta Rp' becomes 7 500 000 before comparing. Claims whose unit has
                # no scale to apply ('Rp', 'unit', or none) compare raw, as before.
                parsed = _parse_scale_unit(fact.unit) if fact.unit else None
                factor = parsed[0] if parsed else 1.0
            if factor is None:
                if best_reason is None:
                    best_reason = (
                        f"Unit conversion from '{fact.unit}' to '{excel_unit}' "
                        f"([{src.label}]) is not supported. Cannot compare."
                    )
                continue

        return _compute_operation(fact, resolved, factor, src)

    return _inconclusive_result(fact, best_missing, best_reason)


# ---------------------------------------------------------------------------
# Table-family suggestions for Inconclusive claims
# ---------------------------------------------------------------------------

# Known BI statistical-table families and the metric keywords that point at them. A BI M2
# report cites series from ~4 different SEKI tables; users typically upload only I.1 and then
# see the rest come back Inconclusive with no hint of WHICH table would cover them. Keyword
# matching is on the extracted metric label (lowercased substring), deterministic on purpose.
_TABLE_FAMILY_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "Uang Primer / M0 (SEKI Tabel 1.2)",
        ("m0", "uang primer", "uang kartal yang diedarkan", "uyd", "giro bank umum"),
    ),
    (
        "Posisi Simpanan Masyarakat / DPK (mis. TABEL1_19 — SEKI Tabel 1.19)",
        ("dpk", "dana pihak ketiga", "deposito", "tabungan masyarakat"),
    ),
    (
        "Posisi Kredit Bank Umum & BPR (SEKI Tabel 1.5 dst — KMK/KI/KK per jenis penggunaan)",
        (
            "kredit", "kmk", "kredit modal kerja", "kredit investasi", "kredit konsumsi",
            "kepemilikan rumah", "kendaraan bermotor", "multiguna", "debitur",
        ),
    ),
]


def _build_table_suggestions(results: List[FactVerificationResult]) -> List[TableSuggestion]:
    """Map Inconclusive claims to the BI table family that likely carries their data.

    Only claims that resolved against NO source at all (matched_excel_source is None) are
    considered — an Inconclusive caused by e.g. a unit mismatch already found its table.
    """
    metrics_by_family: Dict[str, List[str]] = {}
    for r in results:
        if r.verdict != "Inconclusive" or r.matched_excel_source is not None:
            continue
        label_lower = r.metric_label.lower()
        for family, keywords in _TABLE_FAMILY_HINTS:
            if any(kw in label_lower for kw in keywords):
                bucket = metrics_by_family.setdefault(family, [])
                if r.metric_label not in bucket:
                    bucket.append(r.metric_label)
                break  # first matching family wins; families are ordered specific-first

    return [
        TableSuggestion(table=family, metrics=metrics)
        for family, metrics in metrics_by_family.items()
    ]


# ---------------------------------------------------------------------------
# Parser cascade: deterministic BI layout first, generic heuristic parser as fallback
# ---------------------------------------------------------------------------

def _parse_table_with_fallback(
    excel_bytes: bytes, sheet_name: str, llm: Optional[BaseChatModel] = None
) -> Tuple[BITableData, str]:
    """Return (table, parser_name) — three-tier cascade: BI → generic heuristic → LLM mapping.

    The BI parser stays the primary path so known SEKI files keep their exact current
    behaviour. Its result is accepted only when it actually extracted data; a structurally
    successful but EMPTY parse (or a ValueError) falls through to the generic parser. When
    that also fails and an llm is available, the LLM structure-mapping parser (tier 3) gets
    a shot — it only maps structure; values are still extracted by code. As the last resort,
    any BI result we did get is returned even when empty (claims then come back Inconclusive
    instead of the whole request failing); only when every tier raised is the combined error
    surfaced.
    """
    bi_table = None
    bi_error: Optional[Exception] = None
    try:
        bi_table = parse_bi_table(excel_bytes, sheet_name)
        if bi_table.row_labels and bi_table._data:
            return bi_table, "bi"
    except ValueError as exc:
        bi_error = exc

    try:
        return parse_generic_table(excel_bytes, sheet_name), "generic"
    except ValueError as generic_error:
        llm_error: Optional[Exception] = None
        if llm is not None:
            try:
                return parse_table_with_llm(excel_bytes, sheet_name, llm), "llm"
            except ValueError as exc:
                llm_error = exc
                logger.warning("LLM structure-mapping parser failed: %s", exc)
        if bi_table is not None:
            return bi_table, "bi"
        raise ValueError(
            f"Tabel tidak dapat diparsing. Parser BI: {bi_error}. "
            f"Parser generik: {generic_error}."
            + (f" Parser LLM: {llm_error}." if llm_error is not None else "")
        ) from generic_error


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _deduplicate_facts(facts: List[ExtractedFact]) -> List[ExtractedFact]:
    """Remove duplicate (operation, periods) entries — keep first occurrence."""
    seen = set()
    unique = []
    for f in facts:
        key = (f.operation, tuple((p.metric_label, p.year, p.month, p.col_label) for p in f.periods))
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
    progress_cb: ProgressCb = None,
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
        progress_cb:    Optional per-stage progress callback (see ProgressCb above). None
                        disables reporting; the pipeline is otherwise identical.

    Returns:
        PairedVerificationResponse with per-fact verdicts.
    """
    def emit(stage: str, status: str, **extra) -> None:
        if progress_cb is not None:
            progress_cb({"type": "stage", "stage": stage, "status": status, **extra})

    # Step 1: Parse all Excel sources (BI layout → generic heuristics → LLM structure mapping)
    parsed_sources: List[_ExcelSource] = []
    excel_parsers: List[str] = []
    for i, (excel_bytes, sheet_name, filename) in enumerate(excel_sources, 1):
        logger.info("Parsing Excel sheet '%s' from '%s'", sheet_name, filename)
        emit(
            "excel", "running", current=i - 1, total=len(excel_sources),
            detail=f"{filename} / {sheet_name}",
        )
        table, parser_used = _parse_table_with_fallback(excel_bytes, sheet_name, llm=llm)
        logger.info(
            "Excel parsed via %s parser (%s axis): %d rows, unit='%s'",
            parser_used, table.axis_type, len(table.row_labels), table.unit,
        )
        parsed_sources.append(_ExcelSource(table=table, filename=filename, sheet=sheet_name))
        excel_parsers.append(parser_used)
    emit(
        "excel", "done",
        detail=f"{len(parsed_sources)} sumber · parser: {', '.join(excel_parsers)}",
    )

    # Per-source label groups with table title context (used by the LLM to understand
    # what generic rows like 'Total' represent in each table). Categorical sources also
    # advertise their attribute columns so the LLM can fill col_label with a real name.
    def _source_desc(src: _ExcelSource) -> str:
        desc = f"{src.table.title} / {src.filename}"
        if src.table.axis_type == "categorical" and src.table.col_labels:
            desc += " — kolom atribut (non-waktu): " + ", ".join(src.table.col_labels)
        return desc

    source_labels_for_extractor = [
        (_source_desc(src), src.table.row_labels) for src in parsed_sources
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

    def _on_chunk_progress(done: int, total: int) -> None:
        emit("extract", "running", current=done, total=total, detail=f"{total} bagian teks")

    raw_facts = await extract_structured_facts_async(
        narrative_text,
        all_row_labels,
        extraction_primary,
        fallback_llm=extraction_fallback,
        source_labels=source_labels_for_extractor,
        on_progress=_on_chunk_progress,
    )
    facts = _deduplicate_facts(raw_facts)
    logger.info("%d unique facts after deduplication (was %d)", len(facts), len(raw_facts))
    emit("extract", "done", detail=f"{len(facts)} klaim ditemukan")

    excel_filenames = [src.filename for src in parsed_sources]
    excel_sheets = [src.sheet for src in parsed_sources]
    excel_units = [src.table.unit for src in parsed_sources]

    if not facts:
        emit("compare", "done", detail="Tidak ada klaim untuk dibandingkan")
        return PairedVerificationResponse(
            pdf_filename=pdf_filename,
            excel_filenames=excel_filenames,
            excel_sheets=excel_sheets,
            excel_units=excel_units,
            excel_parsers=excel_parsers,
            total_facts=0,
            entailed_count=0,
            refuted_count=0,
            inconclusive_count=0,
            results=[],
        )

    # Step 3: Direct comparison (no SQL) — each fact is checked across all sources
    emit("compare", "running", detail=f"{len(facts)} klaim")
    results: List[FactVerificationResult] = [_evaluate_fact(fact, parsed_sources) for fact in facts]

    entailed = sum(1 for r in results if r.verdict == "Entailed")
    refuted = sum(1 for r in results if r.verdict == "Refuted")
    inconclusive = sum(1 for r in results if r.verdict == "Inconclusive")
    emit(
        "compare", "done",
        detail=f"{entailed} sesuai · {refuted} tidak sesuai · {inconclusive} tidak dapat dipastikan",
    )

    return PairedVerificationResponse(
        pdf_filename=pdf_filename,
        excel_filenames=excel_filenames,
        excel_sheets=excel_sheets,
        excel_units=excel_units,
        excel_parsers=excel_parsers,
        total_facts=len(results),
        entailed_count=entailed,
        refuted_count=refuted,
        inconclusive_count=inconclusive,
        results=results,
        table_suggestions=_build_table_suggestions(results),
    )
