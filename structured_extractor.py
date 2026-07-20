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
from typing import Callable, List, Literal, Optional, Tuple

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
            "the metric name as it appears in the narrative text. For non-time-series tables this "
            "is the ROW name (e.g. the item name in an item list)."
        ),
    )
    year: Optional[int] = Field(
        None,
        description=(
            "The year (e.g. 2026) this data point is about. REQUIRED for every time-series claim; "
            "leave null ONLY for attribute claims that use col_label instead."
        ),
    )
    month: Optional[str] = Field(
        None,
        description=(
            "3-letter English month abbreviation: Jan/Feb/Mar/Apr/May/Jun/"
            "Jul/Aug/Sep/Oct/Nov/Dec. Use Indonesian month name equivalents: "
            "Januari=Jan, Februari=Feb, Maret=Mar, April=Apr, Mei=May, Juni=Jun, "
            "Juli=Jul, Agustus=Aug, September=Sep, Oktober=Oct, November=Nov, Desember=Dec. "
            "REQUIRED for every time-series claim; leave null ONLY for attribute claims "
            "that use col_label instead."
        ),
    )
    col_label: Optional[str] = Field(
        None,
        description=(
            "ONLY for claims against a NON-time-series table (item lists, inventories, budgets): "
            "the attribute/column name this data point refers to (e.g. 'Harga', 'Stok'). Copy the "
            "column name from the source's column list when it clearly matches. Leave null for "
            "time-series claims — never fill both col_label and year/month."
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
    anchor_quote: str = Field(
        ...,
        description=(
            "SHORT verbatim snippet — 3 to 8 consecutive words copied character-for-character "
            "from the narrative — containing this claim's number (for trend claims with no "
            "number: the key directional phrase). Examples: 'tercatat sebesar Rp10.355,1 triliun', "
            "'tumbuh 9,7% (yoy)'. NEVER copy the whole sentence — the surrounding sentence is "
            "recovered from the source text by code."
        ),
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

TWO KINDS OF REFERENCE TABLES:
  - Time-series tables (e.g. BI monetary statistics): every data point is identified by
    (metric_label, year, month). ALWAYS fill year and month; leave col_label null.
  - Non-time-series tables (item lists, inventories, budgets — the source's column list shows
    attribute names like 'Harga' or 'Stok' instead of time periods): for claims about such data
    (e.g. "harga Laptop X adalah Rp7.500.000"), set metric_label to the ROW name (the item),
    set col_label to the COLUMN/attribute name, and leave year and month null. Typical
    operations here are value, diff, and ratio; yoy_growth and trend operations do not apply.

For each claim, output a JSON object with these fields:
  operation        : one of the 9 values above
  periods          : array of {{metric_label, year, month, col_label}} - see each operation's rule
                     above for how many points and in what order
  claimed_value_raw: the number string EXACTLY as it appears in the text (no Rp, no unit word), or
                     null for is_increasing/is_decreasing/is_stable claims with no explicit number
  unit             : one of "triliun Rp", "miliar Rp", "persen_yoy", "persen", or null (matching
                     claimed_value_raw)
  anchor_quote     : SHORT verbatim snippet (3-8 consecutive words, copied character-for-character
                     from the text) containing this claim's number — NOT the whole sentence. For
                     trend claims with no number, quote the key directional phrase instead.

IMPORTANT RULES:
1. claimed_value_raw: copy the digit string verbatim — do NOT convert or interpret it.
   'Rp10.355,1 triliun' → '10.355,1'   |   '9,7% (yoy)' → '9,7'   |   '-9,2% (yoy)' → '-9,2'
2. 'yoy' means year-on-year → operation='yoy_growth', unit='persen_yoy'.
3. For each metric that has both an absolute value AND a growth rate in the same sentence, create
   TWO separate entries: one operation='value', one operation='yoy_growth' — each with its own
   short anchor_quote covering just its own number (e.g. 'sebesar Rp10.355,1 triliun' vs
   'tumbuh 9,7% (yoy)'), never the shared full sentence.
4. For metric_label in each period: use the closest match from the REFERENCE METRIC LIST if one
   clearly fits; otherwise use the exact metric name as it appears in the narrative text. Never skip
   a claim just because it has no reference list match.
5. average/sum/is_increasing/is_decreasing/is_stable: list EVERY month individually in periods —
   never summarize a range as just its start and end.
6. col_label is ONLY for non-time-series tables. If a claim mentions a time period, always use
   year+month and leave col_label null — never fill both.
7. anchor_quote must be an EXACT substring of the narrative (same spelling, punctuation and
   digits) so code can locate it afterwards — never paraphrase, translate, or re-punctuate it.
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
    """Plain-Python equivalent of _PeriodRef, decoupled from LLM schema.

    Exactly one of the two column-axis forms is populated:
      temporal    : year + month set, col_label None (BI time-series claims — unchanged).
      categorical : col_label set, year/month None (attribute claims, e.g. 'Harga' of an item).
    """
    metric_label: str
    year: Optional[int] = None
    month: Optional[str] = None
    col_label: Optional[str] = None


class ExtractedFact:
    """Plain-Python equivalent of _ExtractedFact, decoupled from LLM schema.

    context_quote is the full source sentence recovered deterministically around the
    LLM's short anchor_quote (or the anchor verbatim when it could not be located) —
    the LLM never emits whole sentences, to keep extraction output-token cost down.
    """
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
      '(49,8)'    → -49.8   (accounting-style negative, the BI table convention)

    Returns None and logs a warning if the string cannot be parsed.
    """
    def unwrap_parens(tok: str, neg: bool) -> Tuple[str, bool]:
        # Parentheses wrapping the WHOLE token come in two kinds, told apart by '%':
        #   '(49,8)'  — accounting-style negative, the BI TABLE convention → -49,8.
        #   '(5,5%)'  — prose annotation ('Posisi GWM ... Januari 2020 (5,5%)'), a
        #               POSITIVE value; table cells never carry '%' inside the cell.
        # Partial wraps like '9,7 (yoy)' never reach here, so no sign flip there either.
        if len(tok) > 2 and tok.startswith('(') and tok.endswith(')'):
            inner = tok[1:-1].strip()
            if '%' not in inner:
                neg = True
            if inner.startswith('-'):  # '(-5,5%)' / '(-49,8)': one negation either way
                inner = inner[1:].strip()
                neg = True
            return inner, neg
        return tok, neg

    s = raw.strip()
    if not s:
        return None

    s, negative = unwrap_parens(s, False)
    if s.startswith('-'):
        negative = True
        s = s[1:].strip()

    # Strip any stray Rp prefix or unit words that the LLM may have included despite instructions
    s = re.sub(r'^Rp\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*(triliun|miliar|persen|%)[^\d]*$', '', s, flags=re.IGNORECASE)
    s = s.strip().rstrip('*').strip()
    # Unwrap again: in '(49,8) triliun' the parentheses only surround the number, so they
    # are still intact after the unit suffix was stripped off.
    s, negative = unwrap_parens(s, negative)
    # Remove any spaces still sitting inside the number. Indonesian numerals never contain a
    # space, so an interior space is a PDF-extraction artifact (e.g. "2 1,3" -> "21,3", "8 ,9"
    # -> "8,9"); left in, it made float() raise and the whole fact was dropped.
    s = re.sub(r'\s+', '', s)

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
# Deterministic anchor location: page number + sentence expansion
# ---------------------------------------------------------------------------

_PAGE_MARKER_PATTERN = re.compile(r'\[== Halaman (\d+) ==\]')

# A sentence boundary is [.!?] NOT preceded by a single-letter token — that guard keeps
# Indonesian abbreviations ("s.d.", "a.l.") and list numbering ("1.") from splitting a
# sentence. Digit-internal dots ("10.355,1") never match because no whitespace follows.
# "==]" treats the end of a page marker as a boundary too.
_SENT_START_RE = re.compile(r"(?:(?<!\b\w)[.!?]|==\])\s+(?=[\"'“(\[*]*\s*[A-Z0-9])")
_SENT_END_RE = re.compile(r"(?<!\b\w)[.!?](?=\s|$)")


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces (pypdf line breaks → spaces)."""
    return re.sub(r'\s+', ' ', text)


def _locate_anchor(anchor: str, normalized: str) -> Optional[Tuple[int, int]]:
    """Find a verbatim anchor inside whitespace-normalized text.

    Tries an exact substring match first (the common case for short anchors), then
    falls back to sliding a 6-word window across the anchor so longer quotes whose
    first words differ from the source (mid-sentence starts, LLM prefix drift) still
    match. Returns the (start, end) span of whatever was matched, or None.
    """
    anchor_norm = _normalize_ws(anchor.strip())
    if len(anchor_norm) >= 10:
        pos = normalized.find(anchor_norm)
        if pos != -1:
            return pos, pos + len(anchor_norm)

    words = anchor_norm.split()
    n = 6  # window length in words
    if len(words) < n:
        n = max(3, len(words))

    for start in range(min(len(words) - n + 1, 8)):  # slide up to 8 positions
        window = " ".join(words[start : start + n])
        if len(window) < 15:
            continue
        pos = normalized.find(window)
        if pos != -1:
            return pos, pos + len(window)
    return None


def _page_at(normalized: str, pos: int) -> Optional[int]:
    """Page of the most recent [== Halaman N ==] marker before pos, or None."""
    matches = _PAGE_MARKER_PATTERN.findall(normalized, 0, pos)
    return int(matches[-1]) if matches else None


def _find_page_number(context_quote: str, full_text: str) -> Optional[int]:
    """Find which page a quote belongs to by searching the page-marked text."""
    normalized = _normalize_ws(full_text)
    span = _locate_anchor(context_quote, normalized)
    return _page_at(normalized, span[0]) if span else None


def _expand_anchor_to_sentence(
    normalized: str, start: int, end: int, max_len: int = 600
) -> Optional[str]:
    """Expand a located anchor span to the full source sentence containing it.

    The LLM only emits a short anchor (to keep extraction output-token cost down —
    measured latency is proportional to output tokens); the display sentence is
    recovered here from the source text instead, which is also immune to LLM
    misquoting. Returns None when the expansion is empty or implausibly long
    (runaway text with no sentence punctuation) — the caller then falls back to
    showing the anchor itself.
    """
    sent_start = 0
    for m in _SENT_START_RE.finditer(normalized):
        if m.end() > start:
            break
        sent_start = m.end()

    # The anchor may itself end with the sentence's final punctuation.
    search_from = end - 1 if end > start and normalized[end - 1] in '.!?' else end
    end_match = _SENT_END_RE.search(normalized, search_from)
    next_marker = normalized.find('[== Halaman', end)

    candidates = [len(normalized)]
    if end_match:
        candidates.append(end_match.end())
    if next_marker != -1:
        candidates.append(next_marker)  # page break ends a sentence even without punctuation
    sent_end = min(candidates)

    sentence = _PAGE_MARKER_PATTERN.sub('', normalized[sent_start:sent_end]).strip()
    if not sentence or len(sentence) > max_len:
        return None
    return sentence


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
    """Normalize months, parse claimed values, and resolve pages + quotes for raw LLM output.

    The LLM only emits a short anchor_quote per fact; the full display sentence
    (public context_quote) and the page number are both recovered here by locating
    that anchor in the source text. When the anchor cannot be found, the anchor
    itself becomes the quote and the page stays None.

    Skips (with a warning) any fact with an unrecognised month or an unparseable/missing
    claimed_value_raw, except for is_increasing/is_decreasing/is_stable where a null
    claimed_value_raw is expected (pure direction claims with no explicit number).
    """
    normalized = _normalize_ws(filtered_text)
    facts: List[ExtractedFact] = []
    for f in raw_facts:
        periods: List[PeriodPoint] = []
        valid = True
        for p in f.periods:
            raw_month = getattr(p, "month", None)
            month = _MONTH_NORM.get(raw_month.lower().strip()) if raw_month else None
            year = getattr(p, "year", None)
            col_label = (getattr(p, "col_label", None) or "").strip() or None
            # Temporal wins when both forms are present: a valid (year, month) means the claim
            # is time-series and a stray col_label from the LLM must not reroute it to a
            # categorical source.
            if year is not None and month is not None:
                periods.append(PeriodPoint(metric_label=p.metric_label, year=year, month=month))
            elif col_label is not None:
                periods.append(PeriodPoint(metric_label=p.metric_label, col_label=col_label))
            else:
                logger.warning(
                    "Period without a usable (year, month) or col_label "
                    "(month=%r, year=%r), skipping fact", raw_month, year,
                )
                valid = False
                break
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

        span = _locate_anchor(f.anchor_quote, normalized)
        if span is not None:
            page_num = _page_at(normalized, span[0])
            quote = _expand_anchor_to_sentence(normalized, span[0], span[1]) or f.anchor_quote
        else:
            page_num = None
            quote = f.anchor_quote
        facts.append(ExtractedFact(
            operation=f.operation,
            periods=periods,
            claimed_value=claimed_value,
            unit=f.unit,
            context_quote=quote,
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
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[ExtractedFact]:
    """Parallel version of extract_structured_facts — all chunks are processed concurrently.

    Splits the narrative into page-aligned chunks then fires one LLM call per chunk
    simultaneously via asyncio.gather, reducing wall-clock time from O(N) sequential
    to roughly O(1) (network round-trip for a single call).

    Args:
        source_labels: Optional list of (table_title, row_labels) per source.
                       When provided, the LLM sees each source's title so it
                       can map generic 'Total' rows to the right aggregate metric.
        on_progress:   Optional callback invoked as (completed_chunks, total_chunks) —
                       once with (0, total) as soon as the chunk count is known, then
                       after each chunk's LLM call returns. This step dominates the
                       pipeline's wall-clock time, so it is the only one worth
                       reporting at sub-step granularity. Called on the event loop
                       thread; keep it non-blocking (it must not do I/O or await).
    """
    chain = _EXTRACTION_PROMPT | llm.with_structured_output(_ExtractedFacts)

    row_labels_block = _build_row_labels_block(row_labels, source_labels)
    filtered_text = _filter_narrative(narrative_text)
    logger.info("Narrative filter: %d → %d chars", len(narrative_text), len(filtered_text))

    chunks = _split_into_page_chunks(filtered_text, max_chars_per_chunk)
    logger.info(
        "Processing %d chunk(s) in parallel (max %d chars each)", len(chunks), max_chars_per_chunk
    )

    completed = 0
    if on_progress is not None:
        on_progress(0, len(chunks))

    async def _process_chunk(i: int, chunk: str) -> List[_ExtractedFact]:
        nonlocal completed
        logger.info("Chunk %d/%d: %d chars", i, len(chunks), len(chunk))
        payload = {"row_labels_block": row_labels_block, "narrative_text": chunk}
        result = await _ainvoke_extraction(chain, payload, fallback_llm)
        # Chunks finish out of order; report how many are DONE, not which one, so the
        # count never appears to go backwards.
        completed += 1
        if on_progress is not None:
            on_progress(completed, len(chunks))
        return result.facts if result is not None else []

    chunk_results = await asyncio.gather(
        *[_process_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)]
    )
    all_raw_facts: List[_ExtractedFact] = [f for chunk_facts in chunk_results for f in chunk_facts]

    return _finalize_facts(all_raw_facts, filtered_text)
