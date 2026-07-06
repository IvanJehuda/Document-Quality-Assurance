"""Extracts structured (operation, periods, claimed value) facts from a PDF narrative text.

Unlike claim_extraction.py (which produces free-form natural-language claims for Text-to-SQL),
this module extracts machine-readable claims specifically for direct comparison against a known
BI statistical table. The LLM is given the table's row labels as anchors so it can pick the
closest matching metric name rather than inventing one.

Each claim is represented as an OPERATION over one or more (metric, year, month) data points
(a small, extensible DSL - see _SYSTEM_PROMPT for the full operation list) rather than a fixed
"claim_type" enum, so claims more complex than a single point-in-time value or YoY growth rate
(averages, sums, differences, ratios between two metrics, multi-month trends) are representable
without inventing a new schema shape per pattern. The LLM only ever identifies WHICH data points
a claim references and WHAT operation combines them - it never performs the calculation itself;
paired_verifier.py fetches the referenced points from Excel and computes the result in plain
Python. This mirrors the project-wide principle (see excel_parser_bi.py / typo_checker.py) that
numeric values and arithmetic never pass through an LLM, to avoid hallucination risk.

Only narrative text is targeted (paragraphs / bullet points in the report body).
PDF tables are intentionally excluded because their raw structure is handled separately.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger("fact-checker")

# Maps full month names (English and Indonesian) to 3-letter abbreviations.
# The LLM occasionally ignores the "3-letter only" instruction and outputs
# full names like "April" instead of "Apr".
_MONTH_NORM: dict = {
    "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
    "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
    "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec",
    "januari": "Jan", "februari": "Feb", "maret": "Mar",
    "mei": "May", "juni": "Jun", "juli": "Jul", "agustus": "Aug",
    "oktober": "Oct", "desember": "Dec",
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
}

_OPERATIONS = Literal[
    "value", "yoy_growth", "average", "sum", "diff", "ratio",
    "is_increasing", "is_decreasing", "is_stable",
]

# ---------------------------------------------------------------------------
# Internal LLM output schema
# ---------------------------------------------------------------------------

class _PeriodRef(BaseModel):
    metric_label: str = Field(
        ...,
        description=(
            "The metric name for this specific data point. If a label from the REFERENCE METRIC "
            "LIST clearly matches what the narrative is describing, copy it exactly. Otherwise use "
            "the metric name as it appears in the narrative text."
        ),
    )
    year: int = Field(..., description="The year (e.g. 2026) this data point is about.")
    month: str = Field(
        ...,
        description=(
            "3-letter English month abbreviation: Jan/Feb/Mar/Apr/May/Jun/"
            "Jul/Aug/Sep/Oct/Nov/Dec. Use Indonesian month name equivalents: "
            "Januari=Jan, Februari=Feb, Maret=Mar, April=Apr, Mei=May, Juni=Jun, "
            "Juli=Jul, Agustus=Aug, September=Sep, Oktober=Oct, November=Nov, Desember=Dec."
        ),
    )


class _ExtractedFact(BaseModel):
    operation: _OPERATIONS = Field(
        ..., description="Which operation this claim describes - see system prompt for definitions."
    )
    periods: List[_PeriodRef] = Field(
        ...,
        description=(
            "Every individual (metric, year, month) data point this claim references, listed in "
            "the order they appear/matter (chronological for average/sum/diff/is_increasing/"
            "is_decreasing/is_stable; numerator-then-denominator for ratio). "
            "'value' and 'yoy_growth' need exactly 1. 'diff' and 'ratio' need exactly 2. "
            "'average', 'sum', 'is_increasing', 'is_decreasing', 'is_stable' need 2 or more - "
            "list EVERY month in a stated range individually, never as a range description."
        ),
    )
    claimed_value_raw: Optional[str] = Field(
        None,
        description=(
            "The numeric value claimed in the narrative for this operation's result — copy verbatim "
            "(no Rp, no unit word). "
            "Examples: 'Rp10.355,1 triliun' → '10.355,1'  |  '9,7% (yoy)' → '9,7'  |  "
            "'-9,2% (yoy)' → '-9,2'. "
            "Leave null ONLY for is_increasing/is_decreasing/is_stable claims that state no explicit number."
        ),
    )
    unit: Optional[str] = Field(
        None,
        description=(
            "Unit of claimed_value_raw. Common values: 'triliun Rp', 'miliar Rp', 'persen_yoy', 'persen'. "
            "For non-Rupiah currencies, use the scale word and currency from the text "
            "(e.g. 'miliar dolar AS' → 'miliar USD', 'miliar USD', 'juta USD'). "
            "Leave null when claimed_value_raw is null."
        ),
    )
    context_quote: str = Field(
        ..., description="The verbatim sentence or phrase from the narrative where this claim appears."
    )


class _ExtractedFacts(BaseModel):
    """Wrapper object — always return as a JSON object with the 'facts' key, never as a bare array."""
    facts: List[_ExtractedFact] = Field(
        default_factory=list,
        description="Array of extracted facts. Must be a JSON array inside this object.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise fact extractor for Indonesian statistical monetary reports (Bank Indonesia M2 publications).

Your task: read the narrative text of the report and extract EVERY quantitative claim as a structured
"operation" over one or more (metric, year, month) data points. Do NOT compute anything yourself -
just identify which data points the claim refers to and what operation combines them; the value
comparison is done afterwards by code, never by you.

Available operations:
  value          : one data point stated as a level/absolute number. periods = [that one point].
  yoy_growth     : one data point's year-on-year growth %, e.g. "9,7% (yoy)". periods = [the current
                   point] - the prior-year same-month point is fetched automatically, do NOT include it.
  average        : the stated average of several months of the SAME metric.
                   periods = every individual month in the stated range, listed in order
                   (e.g. "rata-rata Januari-April 2026" → periods = [Jan, Feb, Mar, Apr] of 2026).
  sum            : the stated total of several months of the SAME metric. periods = every month, in order.
  diff           : the stated difference/selisih between exactly TWO points of the SAME metric.
                   periods = [earlier point, later point] (chronological order).
  ratio          : the stated ratio/percentage of one metric relative to ANOTHER metric (or point),
                   e.g. "kredit terhadap DPK sebesar 85%". periods = [numerator point, denominator
                   point] in the order named ("A terhadap/dibanding B" → A first, B second).
  is_increasing  : a claim that a metric rose/kept rising across several months, with no explicit
                   number (e.g. "kredit terus meningkat dari Januari hingga April"). periods = every
                   month in the range, chronological order. claimed_value_raw = null.
  is_decreasing  : same as is_increasing but for a claimed decline. claimed_value_raw = null.
  is_stable      : a claim that a metric stayed roughly flat/stable across several months.
                   claimed_value_raw = null.

For each claim, output a JSON object with these fields:
  operation        : one of the 9 values above
  periods          : array of {{metric_label, year, month}} - see each operation's rule above for how
                     many points and in what order
  claimed_value_raw: the number string EXACTLY as it appears in the text (no Rp, no unit word), or
                     null for is_increasing/is_decreasing/is_stable claims with no explicit number
  unit             : one of "triliun Rp", "miliar Rp", "persen_yoy", "persen", or null (matching
                     claimed_value_raw)
  context_quote    : verbatim sentence/phrase from the text where this claim appears

IMPORTANT RULES:
1. claimed_value_raw: copy the digit string verbatim — do NOT convert or interpret it.
   'Rp10.355,1 triliun' → '10.355,1'   |   '9,7% (yoy)' → '9,7'   |   '-9,2% (yoy)' → '-9,2'
2. 'yoy' means year-on-year → operation='yoy_growth', unit='persen_yoy'.
3. For each metric that has both an absolute value AND a growth rate in the same sentence, create
   TWO separate entries: one operation='value', one operation='yoy_growth'.
4. For metric_label in each period: use the closest match from the REFERENCE METRIC LIST if one
   clearly fits; otherwise use the exact metric name as it appears in the narrative text. Never skip
   a claim just because it has no reference list match.
5. average/sum/is_increasing/is_decreasing/is_stable: list EVERY month individually in periods —
   never summarize a range as just its start and end.
"""

_HUMAN_TEMPLATE = """\
REFERENCE METRIC LIST (row labels from the Excel source(s) below — use as metric_label when a clear \
match exists; otherwise use the metric name from the narrative):
{row_labels_block}

NARRATIVE TEXT:
{narrative_text}
"""

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


# ---------------------------------------------------------------------------
# Row-labels block builder
# ---------------------------------------------------------------------------

def _build_row_labels_block(
    row_labels: List[str],
    source_labels: Optional[List[Tuple[str, List[str]]]] = None,
) -> str:
    """Format the row labels for the extraction prompt.

    When source_labels is provided, each source is shown with a header containing
    the table title so the LLM knows what 'Total' represents in that table.
    """
    if not source_labels:
        return "\n".join(f"- {label}" for label in row_labels)

    parts: List[str] = []
    for source_desc, labels in source_labels:
        parts.append(f"[Tabel: {source_desc}]")
        for label in labels:
            parts.append(f"  - {label}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public dataclasses (re-exported for use in paired_verifier and schemas)
# ---------------------------------------------------------------------------

@dataclass
class PeriodPoint:
    """Plain-Python equivalent of _PeriodRef, decoupled from LLM schema."""
    metric_label: str
    year: int
    month: str


class ExtractedFact:
    """Plain-Python equivalent of _ExtractedFact, decoupled from LLM schema."""
    __slots__ = ("operation", "periods", "claimed_value", "unit", "context_quote", "page_number")

    def __init__(
        self,
        operation: str,
        periods: List[PeriodPoint],
        claimed_value: Optional[float],
        unit: Optional[str],
        context_quote: str,
        page_number: Optional[int] = None,
    ):
        self.operation = operation
        self.periods = periods
        self.claimed_value = claimed_value
        self.unit = unit
        self.context_quote = context_quote
        self.page_number = page_number

    @property
    def display_label(self) -> str:
        """Shared metric name, or 'A / B' for cross-metric operations (e.g. ratio)."""
        labels = list(dict.fromkeys(p.metric_label for p in self.periods))
        return " / ".join(labels)


# ---------------------------------------------------------------------------
# Deterministic Indonesian number parser
# ---------------------------------------------------------------------------

def _parse_indonesian_number(raw: str) -> Optional[float]:
    """Parse an Indonesian-format number string to float.

    Indonesian convention: dots are thousands separators, comma is the decimal point.
      '10.355,1'  → 10355.1
      '9,7'       → 9.7
      '8.516,0'   → 8516.0
      '-9,2'      → -9.2
      '2.396,5'   → 2396.5

    Returns None and logs a warning if the string cannot be parsed.
    """
    s = raw.strip()
    if not s:
        return None

    negative = s.startswith('-')
    if negative:
        s = s[1:].strip()

    # Strip any stray Rp prefix or unit words that the LLM may have included despite instructions
    s = re.sub(r'^Rp\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*(triliun|miliar|persen|%)[^\d]*$', '', s, flags=re.IGNORECASE)
    s = s.strip().rstrip('*').strip()

    try:
        if ',' in s:
            # e.g. "10.355,1" → integer part "10.355", decimal part "1"
            integer_part, decimal_part = s.rsplit(',', 1)
            integer_part = integer_part.replace('.', '')   # remove thousand separators
            value = float(f"{integer_part}.{decimal_part}")
        else:
            # No comma → no decimal; dots are thousand separators only
            value = float(s.replace('.', ''))
    except ValueError:
        logger.warning("Could not parse value_raw %r as Indonesian number", raw)
        return None

    return -value if negative else value


# ---------------------------------------------------------------------------
# Deterministic page-number lookup
# ---------------------------------------------------------------------------

_PAGE_MARKER_PATTERN = re.compile(r'\[== Halaman (\d+) ==\]')


def _find_page_number(context_quote: str, full_text: str) -> Optional[int]:
    """Find which page a context quote belongs to by searching the page-marked text.

    Slides a 6-word window across the context_quote and searches each window in
    full_text. This handles cases where the LLM's quote starts mid-sentence
    (so the first N words differ from what appears in the source text).
    The most recent [== Halaman N ==] marker before the match position is the page.
    Returns None when no window matches.

    Searches a whitespace-normalised copy of full_text so that LLM quotes
    (which join pypdf line breaks with spaces) still match the source text.
    """
    # Normalise to single spaces so that newlines between pypdf lines don't
    # prevent matching against LLM quotes that join those lines with a space.
    normalized = re.sub(r'\s+', ' ', full_text)

    words = context_quote.split()
    n = 6  # anchor length in words
    if len(words) < n:
        n = max(3, len(words))

    for start in range(min(len(words) - n + 1, 8)):  # slide up to 8 positions
        anchor = " ".join(words[start : start + n])
        if len(anchor) < 15:
            continue
        pos = normalized.find(anchor)
        if pos != -1:
            before = normalized[:pos]
            matches = _PAGE_MARKER_PATTERN.findall(before)
            return int(matches[-1]) if matches else None
    return None


# ---------------------------------------------------------------------------
# Narrative-only text filter
# ---------------------------------------------------------------------------

_PAGE_MARKER_LINE = re.compile(r'^\[== Halaman \d+ ==\]$')
_TABLE_PAGE_THRESHOLD = 0.70  # skip page entirely when ≥70% of classified lines are table rows
# Titles that identify a page as a statistical appendix, not narrative
_APPENDIX_TITLE_RE = re.compile(r'^(Lampiran|Tabel\s+[IVX]+\.)', re.IGNORECASE)
# Lines that start with * or ** — could be a footnote/disclaimer marker (e.g. "*Angka sementara")
# OR a markdown bullet point from the vision LLM (e.g. "*   M2 tumbuh sebesar 9,7% (yoy) ...").
# Footnotes in these reports are pure prose with no dated figures; real narrative bullets always
# cite a number, so requiring a digit after the stars tells them apart.
_FOOTNOTE_RE = re.compile(r'^\*+')
_LEADING_STARS_RE = re.compile(r'^\*+\s*')
_HAS_DIGIT_RE = re.compile(r'\d')


def _classify_line(line: str) -> str:
    """Return 'narrative', 'table', or 'skip' for a stripped non-empty, non-marker line."""
    tokens = line.split()
    if len(tokens) < 6:
        return "skip"
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\d.,%()\-]+", t))
    return "table" if numeric / len(tokens) > 0.55 else "narrative"


def _filter_narrative(full_text: str) -> str:
    """Return only narrative prose, skipping appendix pages and non-narrative lines.

    Page-level rules (applied first — drops entire page):
    1. Pages whose first content line matches an appendix title pattern (e.g. "Lampiran N.")
       are dropped entirely — they are statistical table appendices, not narrative.
    2. Pages where ≥70% of classifiable lines are table rows are dropped.

    Line-level rules (within kept pages):
    3. Lines starting with * or ** are dropped UNLESS the text after the stars both reads as a
       full sentence and cites a number — a real narrative bullet (e.g. "*   M2 tumbuh sebesar
       9,7% (yoy) ...") is kept; a footnote/disclaimer (e.g. "*Angka sementara" or a long
       acronym-explainer with no dated figure) is dropped either way.
    4. Short lines (< 6 tokens) and numeric-heavy lines (> 55% numbers) are dropped.
    """
    page_blocks = re.split(r'(?=\[== Halaman \d+ ==\])', full_text)

    result_parts: List[str] = []

    for block in page_blocks:
        if not block.strip():
            continue

        raw_lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not raw_lines:
            continue

        marker = raw_lines[0] if _PAGE_MARKER_LINE.match(raw_lines[0]) else None
        content_lines = raw_lines[1:] if marker else raw_lines

        # Rule 1: skip appendix pages by their title
        first_content = next((l for l in content_lines if len(l.split()) >= 3), "")
        if _APPENDIX_TITLE_RE.match(first_content):
            logger.debug("Skipping appendix %s", marker or "page")
            continue

        narrative: List[str] = []
        table_count = 0

        for line in content_lines:
            if _FOOTNOTE_RE.match(line):
                de_starred = _LEADING_STARS_RE.sub('', line)
                # Rule 3: keep only if it's a full sentence AND cites a number
                if _classify_line(de_starred) != "narrative" or not _HAS_DIGIT_RE.search(de_starred):
                    continue
                narrative.append(line)
                continue
            cls = _classify_line(line)
            if cls == "narrative":
                narrative.append(line)
            elif cls == "table":
                table_count += 1

        # Rule 2: drop pages dominated by table rows
        total_classified = len(narrative) + table_count
        if total_classified > 0:
            if table_count / total_classified >= _TABLE_PAGE_THRESHOLD:
                logger.debug("Skipping %s — %.0f%% table content", marker or "page", table_count / total_classified * 100)
                continue

        if not narrative:
            continue

        if marker:
            result_parts.append(marker)
        result_parts.extend(narrative)

    return "\n".join(result_parts)


def _split_into_page_chunks(filtered_text: str, max_chars: int) -> List[str]:
    """Split page-marker-annotated text into chunks ≤ max_chars, keeping page blocks together."""
    page_blocks = re.split(r'(?=\[== Halaman \d+ ==\])', filtered_text)
    page_blocks = [b for b in page_blocks if b.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for block in page_blocks:
        if current_len + len(block) > max_chars and current:
            chunks.append("".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += len(block)

    if current:
        chunks.append("".join(current))

    return chunks if chunks else [filtered_text]


# ---------------------------------------------------------------------------
# Raw-facts → ExtractedFact finalization (shared by sync and async entry points)
# ---------------------------------------------------------------------------

def _finalize_facts(raw_facts: List[_ExtractedFact], filtered_text: str) -> List[ExtractedFact]:
    """Normalize months, parse claimed values, and resolve page numbers for raw LLM output.

    Skips (with a warning) any fact with an unrecognised month or an unparseable/missing
    claimed_value_raw, except for is_increasing/is_decreasing/is_stable where a null
    claimed_value_raw is expected (pure direction claims with no explicit number).
    """
    facts: List[ExtractedFact] = []
    for f in raw_facts:
        periods: List[PeriodPoint] = []
        valid = True
        for p in f.periods:
            month = _MONTH_NORM.get(p.month.lower().strip())
            if month is None:
                logger.warning("Unrecognised month %r in extracted period, skipping fact", p.month)
                valid = False
                break
            periods.append(PeriodPoint(metric_label=p.metric_label, year=p.year, month=month))
        if not valid or not periods:
            continue

        if f.operation in ("is_increasing", "is_decreasing", "is_stable"):
            claimed_value = None
        else:
            if not f.claimed_value_raw:
                logger.warning("Missing claimed_value_raw for operation=%r, skipping fact", f.operation)
                continue
            claimed_value = _parse_indonesian_number(f.claimed_value_raw)
            if claimed_value is None:
                logger.warning("Skipping fact with unparseable claimed_value_raw=%r", f.claimed_value_raw)
                continue

        page_num = _find_page_number(f.context_quote, filtered_text)
        facts.append(ExtractedFact(
            operation=f.operation,
            periods=periods,
            claimed_value=claimed_value,
            unit=f.unit,
            context_quote=f.context_quote,
            page_number=page_num,
        ))

    logger.info("Finalized %d structured facts from %d raw LLM facts", len(facts), len(raw_facts))
    return facts


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _invoke_extraction(
    chain,
    payload: dict,
    fallback_llm: Optional[BaseChatModel],
) -> Optional[_ExtractedFacts]:
    """Invoke the extraction chain with one optional fallback LLM on failure.

    Some providers (notably Groq) occasionally reject valid structured-output responses
    with a 400 'Failed to call a function' error when the model outputs a bare JSON array
    instead of the expected wrapper object. If that happens and a fallback_llm is provided,
    we retry once using the fallback.
    """
    try:
        return chain.invoke(payload)
    except Exception as primary_err:
        if fallback_llm is None:
            logger.exception("Structured fact extraction failed (no fallback available)")
            return None
        logger.warning("Primary LLM extraction failed (%s), retrying with fallback LLM", primary_err)
        try:
            fallback_chain = _EXTRACTION_PROMPT | fallback_llm.with_structured_output(_ExtractedFacts)
            return fallback_chain.invoke(payload)
        except Exception:
            logger.exception("Fallback LLM extraction also failed")
            return None


async def _ainvoke_extraction(
    chain,
    payload: dict,
    fallback_llm: Optional[BaseChatModel],
) -> Optional[_ExtractedFacts]:
    """Async version of _invoke_extraction — uses ainvoke for non-blocking LLM calls."""
    try:
        return await chain.ainvoke(payload)
    except Exception as primary_err:
        if fallback_llm is None:
            logger.exception("Structured fact extraction failed (no fallback available)")
            return None
        logger.warning("Primary LLM extraction failed (%s), retrying with fallback LLM", primary_err)
        try:
            fallback_chain = _EXTRACTION_PROMPT | fallback_llm.with_structured_output(_ExtractedFacts)
            return await fallback_chain.ainvoke(payload)
        except Exception:
            logger.exception("Fallback LLM extraction also failed")
            return None


def extract_structured_facts(
    narrative_text: str,
    row_labels: List[str],
    llm: BaseChatModel,
    max_chars_per_chunk: int = 6_000,
    fallback_llm: Optional[BaseChatModel] = None,
    source_labels: Optional[List[Tuple[str, List[str]]]] = None,
) -> List[ExtractedFact]:
    """Extract (operation, periods, claimed value) facts from PDF narrative text.

    The narrative is split into page-aligned chunks of ≤ max_chars_per_chunk so
    that long documents are fully covered — the LLM is called once per chunk and
    results are merged. This avoids silently skipping later pages.

    Args:
        narrative_text:     Full text of the PDF (pypdf or vision output).
        row_labels:         Combined metric names from all Excel sources.
        llm:                Primary chat model with structured-output support.
        max_chars_per_chunk: Max chars per LLM call (default 8 000).
        fallback_llm:       Optional secondary model tried if the primary fails.
        source_labels:      Optional list of (table_title, row_labels) per source.
                            When provided, the LLM sees each source's title so it
                            can map generic 'Total' rows to the right aggregate metric.

    Returns:
        List of ExtractedFact objects (may be empty if nothing extractable found).
    """
    chain = _EXTRACTION_PROMPT | llm.with_structured_output(_ExtractedFacts)

    row_labels_block = _build_row_labels_block(row_labels, source_labels)
    filtered_text = _filter_narrative(narrative_text)
    logger.info("Narrative filter: %d → %d chars", len(narrative_text), len(filtered_text))

    chunks = _split_into_page_chunks(filtered_text, max_chars_per_chunk)
    logger.info("Processing %d chunk(s) (max %d chars each)", len(chunks), max_chars_per_chunk)

    all_raw_facts: List[_ExtractedFact] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("Chunk %d/%d: %d chars", i, len(chunks), len(chunk))
        payload = {"row_labels_block": row_labels_block, "narrative_text": chunk}
        result = _invoke_extraction(chain, payload, fallback_llm)
        if result is not None:
            all_raw_facts.extend(result.facts)

    return _finalize_facts(all_raw_facts, filtered_text)


async def extract_structured_facts_async(
    narrative_text: str,
    row_labels: List[str],
    llm: BaseChatModel,
    max_chars_per_chunk: int = 6_000,
    fallback_llm: Optional[BaseChatModel] = None,
    source_labels: Optional[List[Tuple[str, List[str]]]] = None,
) -> List[ExtractedFact]:
    """Parallel version of extract_structured_facts — all chunks are processed concurrently.

    Splits the narrative into page-aligned chunks then fires one LLM call per chunk
    simultaneously via asyncio.gather, reducing wall-clock time from O(N) sequential
    to roughly O(1) (network round-trip for a single call).

    Args:
        source_labels: Optional list of (table_title, row_labels) per source.
                       When provided, the LLM sees each source's title so it
                       can map generic 'Total' rows to the right aggregate metric.
    """
    chain = _EXTRACTION_PROMPT | llm.with_structured_output(_ExtractedFacts)

    row_labels_block = _build_row_labels_block(row_labels, source_labels)
    filtered_text = _filter_narrative(narrative_text)
    logger.info("Narrative filter: %d → %d chars", len(narrative_text), len(filtered_text))

    chunks = _split_into_page_chunks(filtered_text, max_chars_per_chunk)
    logger.info(
        "Processing %d chunk(s) in parallel (max %d chars each)", len(chunks), max_chars_per_chunk
    )

    async def _process_chunk(i: int, chunk: str) -> List[_ExtractedFact]:
        logger.info("Chunk %d/%d: %d chars", i, len(chunks), len(chunk))
        payload = {"row_labels_block": row_labels_block, "narrative_text": chunk}
        result = await _ainvoke_extraction(chain, payload, fallback_llm)
        return result.facts if result is not None else []

    chunk_results = await asyncio.gather(
        *[_process_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)]
    )
    all_raw_facts: List[_ExtractedFact] = [f for chunk_facts in chunk_results for f in chunk_facts]

    return _finalize_facts(all_raw_facts, filtered_text)
