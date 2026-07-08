"""Indonesian spelling/grammar checker for PDF narrative text.

Hybrid design: a free, deterministic dictionary/rule pass handles clear-cut cases; only
genuinely ambiguous candidates are escalated to a single batched LLM call. Mirrors the
confidence-scored heuristic + LLM-escalation pattern used in excel_ingestion.py, with one
important difference discovered empirically while building this module: word-similarity
ratio CANNOT reliably distinguish a real typo from a valid domain term missing from the
general dictionary (e.g. 'inflasi' scores as close to a bogus suggestion as 'resiko' does
to its real correction 'risiko'). So there is no ratio-based auto-flag tier here — every
word the dictionary doesn't recognise (and that isn't a curated tidak-baku entry, a
reduplication, or a likely proper noun) is escalated to the LLM rather than guessed.

Detection tiers, in order, per token:
  1. Reduplication without a hyphen ("rumah rumah") -> grammar issue, fully deterministic.
  2. Exact match in BAKU_MAP (curated tidak-baku -> baku corrections) -> tidak_baku issue,
     fully deterministic.
  3. Standalone "di"/"ke" followed by a word where the JOINED form is a valid dictionary
     word -> ambiguous (could be a passive-voice prefix or a preposition), escalated to LLM.
  4. Capitalised/ALLCAPS unknown words are skipped entirely (likely proper nouns - flagging
     every place/person/brand name in a long report would swamp the LLM call for no benefit).
  5. Any other word not found in the id_ID dictionary -> escalated to LLM.

Escalation is deduplicated by unique word (case-insensitive): if the same unknown word
appears many times in a document, it is judged once and the verdict is applied to every
occurrence, keeping the single batched LLM call bounded by vocabulary size, not word count.
With no LLM available, ambiguous candidates are dropped rather than guessed (conservative
default - no flagging without either deterministic confidence or LLM confirmation).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import phunspell
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pdf_extraction import PAGE_MARKER_RE
from schemas import TypoCheckResponse, TypoIssue

logger = logging.getLogger("fact-checker")

_SP = phunspell.Phunspell("id_ID")

# Only letters and internal hyphens - numbers, punctuation and page markers are never tokens.
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

# How much surrounding text (chars, each side) to show the LLM for an escalated candidate.
_CONTEXT_RADIUS = 60

# Curated common Indonesian tidak-baku (non-standard) -> baku (standard) word corrections.
# Exact-match only (case-insensitive) - no fuzzy matching, so this tier is always safe to
# auto-flag without an LLM call.
BAKU_MAP: Dict[str, str] = {
    "resiko": "risiko",
    "aktifitas": "aktivitas",
    "aktifitasnya": "aktivitasnya",
    "jaman": "zaman",
    "ijin": "izin",
    "kwalitas": "kualitas",
    "kwantitas": "kuantitas",
    "analisa": "analisis",
    "sistim": "sistem",
    "konsekwensi": "konsekuensi",
    "frekwensi": "frekuensi",
    "aktip": "aktif",
    "negatip": "negatif",
    "positip": "positif",
    "efektip": "efektif",
    "subyek": "subjek",
    "obyek": "objek",
    "komplit": "komplet",
    "nasehat": "nasihat",
    "hakekat": "hakikat",
    "praktek": "praktik",
    "karir": "karier",
    "standarisasi": "standardisasi",
    "sekedar": "sekadar",
    "terlanjur": "telanjur",
    "nafas": "napas",
    "trampil": "terampil",
    "jadual": "jadwal",
    "hirarki": "hierarki",
    "menejemen": "manajemen",
}


# ---------------------------------------------------------------------------
# Page-marker stripping with offset -> page mapping
# ---------------------------------------------------------------------------

def _strip_page_markers(text: str) -> Tuple[str, List[Tuple[int, int, Optional[int]]]]:
    """Remove [== Halaman N ==] markers, returning the clean text plus a page-range map.

    The map is a list of (start, end, page_number) spans over the CLEAN text's offsets,
    so a char offset found by the word tokenizer can be attributed back to its source page.
    """
    blocks = re.split(r"(?=\[== Halaman \d+ ==\])", text)
    clean_parts: List[str] = []
    page_ranges: List[Tuple[int, int, Optional[int]]] = []
    offset = 0
    for block in blocks:
        if not block:
            continue
        m = PAGE_MARKER_RE.match(block)
        page_num = int(m.group(1)) if m else None
        content = PAGE_MARKER_RE.sub("", block)
        if not content:
            continue
        clean_parts.append(content)
        page_ranges.append((offset, offset + len(content), page_num))
        offset += len(content)
    return "".join(clean_parts), page_ranges


def _page_for_offset(offset: int, page_ranges: List[Tuple[int, int, Optional[int]]]) -> Optional[int]:
    for start, end, page in page_ranges:
        if start <= offset < end:
            return page
    return page_ranges[-1][2] if page_ranges else None


# ---------------------------------------------------------------------------
# Candidate collection (deduplicated by word, escalated to the LLM)
# ---------------------------------------------------------------------------

@dataclass
class _Occurrence:
    start: int
    end: int
    text: str


@dataclass
class _Candidate:
    index: int
    kind: Literal["unknown_word", "di_ke_prefix"]
    display_text: str
    context: str
    suggestion_hint: Optional[str]
    occurrences: List[_Occurrence] = field(default_factory=list)


_MONTHS = frozenset({
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
    "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "jun", "jul", "ags", "agu", "agt", "sep", "sept",
    "okt", "nov", "des",
})


def _joins_to_real_word(tokens, i: int, clean_text: str) -> bool:
    """True when the token at index i is a fragment pypdf split off with a stray
    space, i.e. gluing it with one or two directly-adjacent neighbours (exactly
    one space between each) reforms a real dictionary word or a month name.

    Handles 2-part splits ("k redit"->kredit, "M ei"->Mei) and 3-part splits
    ("di terbi tkan"->diterbitkan). Any join that lands on a real word means the
    fragment is an extraction artifact, not a misspelling.
    """
    def glued(a, b) -> bool:  # b directly follows a with exactly one space
        return clean_text[a.end():b.start()] == " "

    n = len(tokens)
    left = i - 1 >= 0 and glued(tokens[i - 1], tokens[i])
    right = i + 1 < n and glued(tokens[i], tokens[i + 1])
    combos = []
    if left:
        combos.append((i - 1, i))
        if i - 2 >= 0 and glued(tokens[i - 2], tokens[i - 1]):
            combos.append((i - 2, i - 1, i))
    if right:
        combos.append((i, i + 1))
        if i + 2 < n and glued(tokens[i + 1], tokens[i + 2]):
            combos.append((i, i + 1, i + 2))
    if left and right:  # fragment in the MIDDLE of a 3-part split ("di terbi tkan")
        combos.append((i - 1, i, i + 1))

    for combo in combos:
        joined = "".join(tokens[j].group() for j in combo).lower()
        if _SP.lookup(joined) or joined in _MONTHS:
            return True
    return False


def _collect_candidates_and_deterministic_issues(
    clean_text: str,
    page_ranges: List[Tuple[int, int, Optional[int]]],
) -> Tuple[List[TypoIssue], List[_Candidate]]:
    tokens = list(_WORD_RE.finditer(clean_text))
    issues: List[TypoIssue] = []
    candidates_by_key: Dict[str, _Candidate] = {}
    skip_next = False

    for i, m in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        word = m.group()
        start, end = m.start(), m.end()
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        between = clean_text[end:nxt.start()] if nxt else ""
        adjacent = nxt is not None and between.strip() == ""

        # Tier 1: reduplication without a hyphen. Case-insensitive on the FIRST token only
        # (sentence-initial capitalisation shouldn't hide "Rumah rumah"), but the second token
        # must be genuinely lowercase - real reduplication never re-capitalises the repeat, so
        # requiring this excludes false positives like an ALL-CAPS section heading immediately
        # followed by a new Title-case paragraph starting with the same word (e.g. a "PERKEMBANGAN
        # KREDIT" heading directly above a "Kredit yang disalurkan..." paragraph).
        if (
            adjacent
            and nxt.group().islower()
            and nxt.group() == word.lower()
            and word.isalpha()
            and len(word) >= 3
        ):
            issues.append(TypoIssue(
                word=f"{word} {nxt.group()}",
                start=start,
                end=nxt.end(),
                category="grammar",
                suggestion=f"{word}-{nxt.group().lower()}",
                explanation=f"Reduplikasi sebaiknya ditulis dengan tanda hubung: '{word}-{nxt.group().lower()}'.",
                page_number=_page_for_offset(start, page_ranges),
            ))
            skip_next = True
            continue

        # Tier 2: curated tidak-baku -> baku exact match
        baku = BAKU_MAP.get(word.lower())
        if baku is not None:
            issues.append(TypoIssue(
                word=word,
                start=start,
                end=end,
                category="tidak_baku",
                suggestion=baku,
                explanation=f"'{word}' adalah bentuk tidak baku; bentuk baku: '{baku}'.",
                page_number=_page_for_offset(start, page_ranges),
            ))
            continue

        # Tier 3: standalone di-/ke- where the joined form is a real word -> ambiguous
        if adjacent and word.lower() in ("di", "ke") and _SP.lookup((word + nxt.group()).lower()):
            key = f"di_ke:{nxt.group().lower()}"
            display = f"{word} {nxt.group()}"
            ctx_start = max(0, start - _CONTEXT_RADIUS)
            ctx_end = min(len(clean_text), nxt.end() + _CONTEXT_RADIUS)
            cand = candidates_by_key.get(key)
            if cand is None:
                cand = _Candidate(
                    index=len(candidates_by_key),
                    kind="di_ke_prefix",
                    display_text=display,
                    context=clean_text[ctx_start:ctx_end],
                    suggestion_hint=(word + nxt.group()).lower(),
                )
                candidates_by_key[key] = cand
            cand.occurrences.append(_Occurrence(start=start, end=nxt.end(), text=display))
            skip_next = True
            continue

        # Tier 4: likely proper noun (capitalised/ALLCAPS) -> skip entirely
        if not word.islower():
            continue

        # Tier 5: unknown to the dictionary -> always escalate, never guess from ratio alone.
        # Single-letter tokens are skipped outright - Indonesian has no standalone one-letter
        # words, so these are always artifacts (e.g. an English possessive/contraction like
        # "Banker's" splitting into "Banker" + "s" since the word regex has no apostrophe
        # handling, or an abbreviation fragment like "k/" leaking in from tabular content) and
        # would otherwise burn an escalation slot on noise instead of real candidates.
        if len(word) < 2:
            continue
        if _SP.lookup(word):
            continue
        # A 2-letter lowercase token with no vowel ("lz" and other stray glyphs, or
        # unit codes like "kg"/"km") is never a real Indonesian misspelling to flag.
        if len(word) == 2 and word.isalpha() and not any(v in word for v in "aiueo"):
            continue
        # Extraction artifact, not a typo: pypdf split a word with a stray space
        # ("k redit"->kredit, "Tagi han"->Tagihan, "M ei"->Mei, "di terbi tkan"->
        # diterbitkan). If gluing this fragment with adjacent neighbours reforms a
        # real dictionary word or month, skip it rather than flag the fragment.
        if _joins_to_real_word(tokens, i, clean_text):
            continue
        key = f"word:{word.lower()}"
        ctx_start = max(0, start - _CONTEXT_RADIUS)
        ctx_end = min(len(clean_text), end + _CONTEXT_RADIUS)
        cand = candidates_by_key.get(key)
        if cand is None:
            suggestions = list(_SP.suggest(word))[:3]
            cand = _Candidate(
                index=len(candidates_by_key),
                kind="unknown_word",
                display_text=word,
                context=clean_text[ctx_start:ctx_end],
                suggestion_hint=suggestions[0] if suggestions else None,
            )
            candidates_by_key[key] = cand
        cand.occurrences.append(_Occurrence(start=start, end=end, text=word))

    return issues, list(candidates_by_key.values())


# ---------------------------------------------------------------------------
# LLM escalation (batched, one call for the whole document)
# ---------------------------------------------------------------------------

_TYPO_SYSTEM_PROMPT = """You are checking Indonesian-language text (Bank Indonesia statistical reports)
for genuine spelling and grammar issues. You are given a list of AMBIGUOUS candidates that a
dictionary pass could not resolve on its own - your job is to make the final call for each one.

Each candidate has a "kind":

- "unknown_word": a word not found in the general Indonesian dictionary. This does NOT mean it
  is a typo - it is very often a valid economic/statistical term, proper noun, abbreviation, or
  loanword the dictionary simply doesn't contain (e.g. "inflasi", "yoy", "moneter" are all
  perfectly valid despite sometimes tripping a dictionary lookup). Only mark is_issue=true if the
  word is genuinely misspelled - judge from the surrounding context. When it IS a typo, category
  should usually be "ejaan" and suggestion should be the corrected spelling.

- "di_ke_prefix": the text currently shows "di X" or "ke X" written as TWO SEPARATE words, where
  joining them ("diX"/"keX") also happens to be a valid dictionary word. Decide which is
  grammatically correct here:
    * If "X" is being used as a VERB taking the passive prefix "di-"/"ke-" (e.g. "di makan" should
      be "dimakan", meaning "is eaten") -> is_issue=true, category="grammar", suggestion=the
      JOINED form.
    * If "di"/"ke" is a PREPOSITION before a place/noun (e.g. "di rumah" = "at home" is correct
      as written) -> is_issue=false.

Respond with one verdict per candidate index. Never invent a candidate_index that wasn't given."""

_TYPO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _TYPO_SYSTEM_PROMPT),
    ("human", "Candidates:\n{items_block}\n\nGive one verdict per candidate."),
])


class IndexedTypoVerdict(BaseModel):
    candidate_index: int = Field(..., description="Index of the candidate this verdict is for.")
    is_issue: bool = Field(..., description="True if this candidate is a genuine spelling/grammar issue.")
    category: Literal["ejaan", "tidak_baku", "grammar"] = Field(
        ..., description="Issue category. Ignored when is_issue is false."
    )
    suggestion: str = Field(..., description="Corrected form. Ignored when is_issue is false.")
    explanation: str = Field(..., description="Brief explanation in Indonesian of the verdict.")


class BatchTypoVerdicts(BaseModel):
    verdicts: List[IndexedTypoVerdict] = Field(default_factory=list)


def build_typo_verdict_chain(llm: BaseChatModel):
    return _TYPO_PROMPT | llm.with_structured_output(BatchTypoVerdicts)


def _escalate_to_llm(candidates: List[_Candidate], llm: BaseChatModel) -> Dict[int, IndexedTypoVerdict]:
    chain = build_typo_verdict_chain(llm)
    items_block = "\n".join(
        f"{c.index}. kind={c.kind}, text={c.display_text!r}, context=\"...{c.context}...\", "
        f"kamus_suggestion={c.suggestion_hint!r}"
        for c in candidates
    )
    try:
        response: BatchTypoVerdicts = chain.invoke({"items_block": items_block})
    except Exception:
        logger.exception("Typo LLM escalation failed; ambiguous candidates will be dropped")
        return {}
    return {v.candidate_index: v for v in response.verdicts}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_typos(text: str, llm: Optional[BaseChatModel] = None) -> TypoCheckResponse:
    """Check Indonesian narrative text for spelling/tidak-baku/grammar issues.

    Args:
        text: Narrative text, optionally containing [== Halaman N ==] page markers
              (e.g. from pdf_extraction.extract_narrative_text()).
        llm:  Chat model used for the single batched escalation call. If None, ambiguous
              candidates (unknown words, di-/ke- prefix cases) are dropped rather than guessed.

    Returns:
        TypoCheckResponse with all detected issues, sorted by position in the text.
    """
    clean_text, page_ranges = _strip_page_markers(text)
    issues, candidates = _collect_candidates_and_deterministic_issues(clean_text, page_ranges)

    if candidates:
        verdicts = _escalate_to_llm(candidates, llm) if llm is not None else {}
        for cand in candidates:
            verdict = verdicts.get(cand.index)
            if verdict is None or not verdict.is_issue:
                continue
            for occ in cand.occurrences:
                issues.append(TypoIssue(
                    word=occ.text,
                    start=occ.start,
                    end=occ.end,
                    category=verdict.category,
                    suggestion=verdict.suggestion,
                    explanation=verdict.explanation,
                    page_number=_page_for_offset(occ.start, page_ranges),
                ))

    issues.sort(key=lambda x: x.start)
    ejaan_count = sum(1 for x in issues if x.category == "ejaan")
    tidak_baku_count = sum(1 for x in issues if x.category == "tidak_baku")
    grammar_count = sum(1 for x in issues if x.category == "grammar")
    total = len(issues)
    summary = (
        f"Ditemukan {total} isu: {ejaan_count} ejaan, {tidak_baku_count} kata tidak baku, "
        f"{grammar_count} tata bahasa."
        if total else "Tidak ditemukan isu ejaan atau tata bahasa."
    )

    return TypoCheckResponse(
        total_issues=total,
        ejaan_count=ejaan_count,
        tidak_baku_count=tidak_baku_count,
        grammar_count=grammar_count,
        summary=summary,
        issues=issues,
    )
