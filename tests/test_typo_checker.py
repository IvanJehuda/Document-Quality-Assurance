from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda

from typo_checker import (
    BatchTypoVerdicts,
    IndexedTypoVerdict,
    _collect_candidates_and_deterministic_issues,
    _strip_page_markers,
    check_typos,
)


def _llm_returning(verdicts):
    """A fake llm whose `.with_structured_output(...)` returns the given verdict list."""
    structured = RunnableLambda(lambda _prompt_value: BatchTypoVerdicts(verdicts=verdicts))
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured)
    return llm


# ---------------------------------------------------------------------------
# Deterministic tiers - no LLM involved
# ---------------------------------------------------------------------------

def test_baku_map_hit_is_flagged_without_an_llm():
    resp = check_typos("Resiko aktifitas ekonomi tetap terjaga.", llm=None)

    categories = {i.word.lower(): i.category for i in resp.issues}
    assert categories["resiko"] == "tidak_baku"
    assert categories["aktifitas"] == "tidak_baku"
    assert resp.tidak_baku_count == 2


def test_reduplication_without_hyphen_is_flagged():
    resp = check_typos("Rumah rumah di kompleks itu megah.", llm=None)

    assert resp.total_issues == 1
    issue = resp.issues[0]
    assert issue.category == "grammar"
    assert issue.suggestion == "Rumah-rumah"


def test_all_caps_heading_followed_by_titlecase_paragraph_is_not_reduplication():
    # Regression test: found via a real BI report where an ALL-CAPS section heading
    # ("PERKEMBANGAN KREDIT") sits directly above a new paragraph starting with the same
    # word in Title-case ("Kredit yang disalurkan..."). This must not be flagged as
    # reduplication - real reduplication never re-capitalises the repeated word.
    resp = check_typos("PERKEMBANGAN KREDIT\nKredit yang disalurkan oleh perbankan tumbuh positif.", llm=None)

    assert resp.total_issues == 0


def test_proper_noun_is_skipped_without_escalating_to_llm():
    llm = _llm_returning([])
    llm.with_structured_output = Mock(side_effect=AssertionError("should not be called"))

    resp = check_typos("Saya tinggal di Surabaya sejak lama.", llm=llm)

    assert resp.total_issues == 0


def test_clean_text_has_no_issues():
    resp = check_typos("Perekonomian tumbuh dengan baik pada triwulan ini.", llm=None)
    assert resp.total_issues == 0
    assert resp.summary == "Tidak ditemukan isu ejaan atau tata bahasa."


# ---------------------------------------------------------------------------
# Ambiguous cases - always escalated, never auto-flagged by suggestion ratio
# ---------------------------------------------------------------------------

def test_unknown_word_is_dropped_when_no_llm_available():
    # "asalasalan" is not in the dictionary and has no BAKU_MAP entry - with no llm it must be
    # dropped rather than guessed, per the conservative-default design.
    resp = check_typos("Kata asalasalan ini aneh sekali.", llm=None)
    assert resp.total_issues == 0


def test_unknown_word_resolved_as_valid_jargon_by_llm_produces_no_issue():
    # This is the key regression test for the false-positive risk found while building this
    # module: word-similarity ratio alone cannot tell "inflasi" (valid jargon) apart from a
    # real typo like "resiko" - so the LLM (not a ratio threshold) must make this call.
    llm = _llm_returning([
        IndexedTypoVerdict(candidate_index=0, is_issue=False, category="ejaan", suggestion="", explanation="istilah valid"),
    ])

    resp = check_typos("Inflasi tercatat rendah pada bulan ini.", llm=llm)

    assert resp.total_issues == 0


def test_unknown_word_confirmed_as_typo_by_llm_produces_ejaan_issue():
    llm = _llm_returning([
        IndexedTypoVerdict(candidate_index=0, is_issue=True, category="ejaan", suggestion="asal-asalan", explanation="typo"),
    ])

    resp = check_typos("Kata asalasalan ini aneh sekali.", llm=llm)

    assert resp.total_issues == 1
    assert resp.issues[0].category == "ejaan"
    assert resp.issues[0].suggestion == "asal-asalan"


def test_di_prefix_candidate_confirmed_as_grammar_issue_by_llm():
    llm = _llm_returning([
        IndexedTypoVerdict(candidate_index=0, is_issue=True, category="grammar", suggestion="dibaca", explanation="prefiks pasif"),
    ])

    resp = check_typos("Buku itu di baca oleh banyak orang.", llm=llm)

    assert resp.total_issues == 1
    issue = resp.issues[0]
    assert issue.category == "grammar"
    assert issue.word == "di baca"
    assert issue.suggestion == "dibaca"


def test_di_prefix_candidate_confirmed_as_correct_preposition_produces_no_issue():
    llm = _llm_returning([
        IndexedTypoVerdict(candidate_index=0, is_issue=False, category="grammar", suggestion="", explanation="preposisi benar"),
    ])

    resp = check_typos("Dia tinggal di sana.", llm=llm)

    assert resp.total_issues == 0


def test_ambiguous_candidates_dropped_when_llm_call_raises():
    # The real failure point is the .invoke() call (a network round-trip), not
    # with_structured_output() (which only configures the model) - mirrors how
    # excel_ingestion.py's equivalent escalation call can fail.
    failing_chain = RunnableLambda(lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    llm = Mock()
    llm.with_structured_output = Mock(return_value=failing_chain)

    resp = check_typos("Kata asalasalan ini aneh sekali.", llm=llm)

    assert resp.total_issues == 0


# ---------------------------------------------------------------------------
# Deduplication - a repeated unknown word is escalated once, verdict applied to every occurrence
# ---------------------------------------------------------------------------

def test_repeated_unknown_word_is_escalated_only_once():
    # Mid-sentence placement on every occurrence - a word right after a sentence-ending period
    # is sentence-initial and gets skipped as a likely proper noun (see the module docstring's
    # tier 4), so this deliberately keeps "asalasalan" away from that position.
    clean_text, page_ranges = _strip_page_markers(
        "Kata itu asalasalan sekali. Katanya asalasalan lagi. Dan asalasalan lagi ketiga kalinya."
    )
    _issues, candidates = _collect_candidates_and_deterministic_issues(clean_text, page_ranges)

    unknown_word_candidates = [c for c in candidates if c.kind == "unknown_word"]
    assert len(unknown_word_candidates) == 1
    assert len(unknown_word_candidates[0].occurrences) == 3


def test_repeated_unknown_word_confirmed_as_typo_flags_every_occurrence():
    llm = _llm_returning([
        IndexedTypoVerdict(candidate_index=0, is_issue=True, category="ejaan", suggestion="asal-asalan", explanation="typo"),
    ])

    resp = check_typos("Kata itu asalasalan sekali. Katanya asalasalan lagi.", llm=llm)

    assert resp.total_issues == 2
    assert all(i.suggestion == "asal-asalan" for i in resp.issues)


# ---------------------------------------------------------------------------
# Page attribution
# ---------------------------------------------------------------------------

def test_issue_page_number_matches_the_page_marker_it_appears_under():
    text = (
        "[== Halaman 1 ==]\nSemua baik-baik saja di sini.\n\n"
        "[== Halaman 2 ==]\nResiko aktifitas ekonomi tetap terjaga."
    )

    resp = check_typos(text, llm=None)

    assert resp.total_issues == 2
    assert all(i.page_number == 2 for i in resp.issues)


def test_page_markers_are_stripped_and_not_treated_as_words():
    # "Halaman" itself must never show up as a flagged/considered token.
    text = "[== Halaman 1 ==]\nTeks yang bersih dan baku."
    resp = check_typos(text, llm=None)
    assert resp.total_issues == 0
