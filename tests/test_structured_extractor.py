import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from structured_extractor import (
    ExtractedFact,
    PeriodPoint,
    _filter_narrative,
    _finalize_facts,
    _find_page_number,
    _parse_indonesian_number,
    _split_into_page_chunks,
    extract_structured_facts,
    extract_structured_facts_async,
)

NARRATIVE = (
    "[== Halaman 1 ==]\n"
    "Uang beredar M2 pada April 2026 tercatat sebesar Rp10.355,1 triliun.\n"
)
ROW_LABELS = ["Uang Beredar (M2)"]


def _fake_period(**overrides):
    base = dict(metric_label="Uang Beredar (M2)", year=2026, month="Apr")
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_fact(**overrides):
    base = dict(
        operation="value",
        periods=[_fake_period()],
        claimed_value_raw="10.355,1",
        unit="triliun Rp",
        context_quote="Uang beredar M2 pada April 2026 tercatat sebesar Rp10.355,1 triliun.",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _llm_with_facts(facts):
    """A fake llm whose `.with_structured_output(...)` always returns the given facts."""
    structured = RunnableLambda(lambda _prompt_value: SimpleNamespace(facts=facts))
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured)
    return llm


def _llm_raising(message):
    def _raise(_prompt_value):
        raise RuntimeError(message)

    structured = RunnableLambda(_raise)
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured)
    return llm


# ---------------------------------------------------------------------------
# _parse_indonesian_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.355,1", 10355.1),
        ("9,7", 9.7),
        ("8.516,0", 8516.0),
        ("-9,2", -9.2),
        ("2.396,5", 2396.5),
        ("100", 100.0),
        ("1.234", 1234.0),
    ],
)
def test_parse_indonesian_number_valid_formats(raw, expected):
    assert _parse_indonesian_number(raw) == pytest.approx(expected)


def test_parse_indonesian_number_strips_rp_prefix_and_unit_suffix():
    assert _parse_indonesian_number("Rp10.355,1 triliun") == pytest.approx(10355.1)


def test_parse_indonesian_number_returns_none_for_unparseable_string():
    assert _parse_indonesian_number("abc") is None


def test_parse_indonesian_number_returns_none_for_empty_string():
    assert _parse_indonesian_number("   ") is None


# ---------------------------------------------------------------------------
# _find_page_number
# ---------------------------------------------------------------------------

def test_find_page_number_returns_page_of_matching_quote():
    full_text = (
        "[== Halaman 1 ==]\n"
        "Ini adalah teks halaman pertama yang cukup panjang untuk dicocokkan.\n"
        "[== Halaman 2 ==]\n"
        "Uang beredar M2 pada April 2026 tercatat sebesar Rp10.355,1 triliun.\n"
    )

    assert _find_page_number("Uang beredar M2 pada April 2026", full_text) == 2


def test_find_page_number_returns_none_when_no_match():
    full_text = "[== Halaman 1 ==]\nSome unrelated narrative text here for testing.\n"

    assert _find_page_number("Nonexistent quote that never appears anywhere", full_text) is None


# ---------------------------------------------------------------------------
# _filter_narrative
# ---------------------------------------------------------------------------

def test_filter_narrative_keeps_prose_drops_footnotes_and_appendix_pages():
    full_text = (
        "[== Halaman 1 ==]\n"
        "Uang beredar M2 pada April 2026 tumbuh signifikan dibandingkan bulan sebelumnya.\n"
        "* Catatan kaki yang seharusnya diabaikan sepenuhnya dari hasil ekstraksi.\n"
        "[== Halaman 2 ==]\n"
        "Lampiran 1. Tabel Statistik Uang Beredar\n"
        "1.234,5 2.345,6 3.456,7 4.567,8 5.678,9 6.789,0\n"
    )

    result = _filter_narrative(full_text)

    assert "tumbuh signifikan" in result
    assert "Catatan kaki" not in result
    assert "Lampiran" not in result
    assert "1.234,5" not in result


def test_filter_narrative_drops_page_dominated_by_numeric_table_rows():
    full_text = (
        "[== Halaman 3 ==]\n"
        "Data Bulanan Ringkasan Ekonomi\n"
        "1.000,0 2.000,0 3.000,0 4.000,0 5.000,0 6.000,0\n"
        "1.100,0 2.100,0 3.100,0 4.100,0 5.100,0 6.100,0\n"
        "1.200,0 2.200,0 3.200,0 4.200,0 5.200,0 6.200,0\n"
    )

    assert _filter_narrative(full_text) == ""


# ---------------------------------------------------------------------------
# _split_into_page_chunks
# ---------------------------------------------------------------------------

def test_split_into_page_chunks_keeps_pages_together_when_under_max_chars():
    text = "[== Halaman 1 ==]\nAAA\n[== Halaman 2 ==]\nBBB\n[== Halaman 3 ==]\nCCC\n"

    chunks = _split_into_page_chunks(text, max_chars=100)

    assert chunks == [text]


def test_split_into_page_chunks_splits_when_exceeding_max_chars():
    page1 = "[== Halaman 1 ==]\n" + "A" * 50 + "\n"
    page2 = "[== Halaman 2 ==]\n" + "B" * 50 + "\n"

    chunks = _split_into_page_chunks(page1 + page2, max_chars=60)

    assert chunks == [page1, page2]


# ---------------------------------------------------------------------------
# ExtractedFact.display_label
# ---------------------------------------------------------------------------

def test_display_label_returns_single_label_when_all_periods_share_metric():
    fact = ExtractedFact(
        operation="average",
        periods=[
            PeriodPoint(metric_label="Penyaluran Kredit", year=2026, month="Jan"),
            PeriodPoint(metric_label="Penyaluran Kredit", year=2026, month="Feb"),
        ],
        claimed_value=100.0, unit="triliun Rp", context_quote="q",
    )
    assert fact.display_label == "Penyaluran Kredit"


def test_display_label_joins_distinct_labels_for_cross_metric_operation():
    fact = ExtractedFact(
        operation="ratio",
        periods=[
            PeriodPoint(metric_label="Kredit", year=2026, month="Apr"),
            PeriodPoint(metric_label="DPK", year=2026, month="Apr"),
        ],
        claimed_value=85.0, unit="persen", context_quote="q",
    )
    assert fact.display_label == "Kredit / DPK"


# ---------------------------------------------------------------------------
# _finalize_facts
# ---------------------------------------------------------------------------

def test_finalize_facts_parses_value_operation():
    facts = _finalize_facts([_fake_fact()], NARRATIVE)

    assert len(facts) == 1
    assert facts[0].operation == "value"
    assert facts[0].claimed_value == pytest.approx(10355.1)
    assert facts[0].periods[0].month == "Apr"
    assert facts[0].page_number == 1


def test_finalize_facts_skips_unparseable_claimed_value():
    facts = _finalize_facts([_fake_fact(claimed_value_raw="not-a-number")], NARRATIVE)
    assert facts == []


def test_finalize_facts_skips_missing_claimed_value_for_value_operation():
    facts = _finalize_facts([_fake_fact(claimed_value_raw=None)], NARRATIVE)
    assert facts == []


def test_finalize_facts_skips_unrecognized_month():
    facts = _finalize_facts([_fake_fact(periods=[_fake_period(month="Xyz")])], NARRATIVE)
    assert facts == []


def test_finalize_facts_normalizes_full_indonesian_month_name():
    facts = _finalize_facts([_fake_fact(periods=[_fake_period(month="April")])], NARRATIVE)
    assert facts[0].periods[0].month == "Apr"


def test_finalize_facts_average_keeps_every_period_and_claimed_value():
    raw = _fake_fact(
        operation="average",
        periods=[
            _fake_period(month="Jan"), _fake_period(month="Feb"),
            _fake_period(month="Mar"), _fake_period(month="Apr"),
        ],
        claimed_value_raw="105,5",
    )
    facts = _finalize_facts([raw], NARRATIVE)

    assert len(facts) == 1
    assert facts[0].operation == "average"
    assert len(facts[0].periods) == 4
    assert facts[0].claimed_value == pytest.approx(105.5)


def test_finalize_facts_sum_keeps_every_period():
    raw = _fake_fact(
        operation="sum",
        periods=[_fake_period(month="Jan"), _fake_period(month="Feb")],
        claimed_value_raw="200",
    )
    facts = _finalize_facts([raw], NARRATIVE)

    assert facts[0].operation == "sum"
    assert len(facts[0].periods) == 2


def test_finalize_facts_diff_keeps_two_chronological_periods():
    raw = _fake_fact(
        operation="diff",
        periods=[_fake_period(month="Jan"), _fake_period(month="Apr")],
        claimed_value_raw="50",
    )
    facts = _finalize_facts([raw], NARRATIVE)

    assert facts[0].operation == "diff"
    assert [p.month for p in facts[0].periods] == ["Jan", "Apr"]


def test_finalize_facts_ratio_keeps_two_periods_with_distinct_metrics():
    raw = _fake_fact(
        operation="ratio",
        periods=[
            _fake_period(metric_label="Kredit", month="Apr"),
            _fake_period(metric_label="DPK", month="Apr"),
        ],
        claimed_value_raw="85",
        unit="persen",
    )
    facts = _finalize_facts([raw], NARRATIVE)

    assert facts[0].operation == "ratio"
    assert [p.metric_label for p in facts[0].periods] == ["Kredit", "DPK"]


@pytest.mark.parametrize("operation", ["is_increasing", "is_decreasing", "is_stable"])
def test_finalize_facts_trend_operations_allow_null_claimed_value(operation):
    raw = _fake_fact(
        operation=operation,
        periods=[_fake_period(month="Jan"), _fake_period(month="Feb"), _fake_period(month="Mar")],
        claimed_value_raw=None,
        unit=None,
    )
    facts = _finalize_facts([raw], NARRATIVE)

    assert len(facts) == 1
    assert facts[0].operation == operation
    assert facts[0].claimed_value is None
    assert len(facts[0].periods) == 3


# ---------------------------------------------------------------------------
# extract_structured_facts / extract_structured_facts_async
# ---------------------------------------------------------------------------

def test_extract_structured_facts_parses_value_and_resolves_page_number():
    llm = _llm_with_facts([_fake_fact()])

    facts = extract_structured_facts(NARRATIVE, ROW_LABELS, llm)

    assert len(facts) == 1
    assert facts[0].claimed_value == pytest.approx(10355.1)
    assert facts[0].periods[0].month == "Apr"
    assert facts[0].page_number == 1


def test_extract_structured_facts_skips_unparseable_value():
    llm = _llm_with_facts([_fake_fact(claimed_value_raw="not-a-number")])

    assert extract_structured_facts(NARRATIVE, ROW_LABELS, llm) == []


def test_extract_structured_facts_skips_unrecognized_month():
    llm = _llm_with_facts([_fake_fact(periods=[_fake_period(month="Xyz")])])

    assert extract_structured_facts(NARRATIVE, ROW_LABELS, llm) == []


def test_extract_structured_facts_normalizes_full_indonesian_month_name():
    llm = _llm_with_facts([_fake_fact(periods=[_fake_period(month="April")])])

    facts = extract_structured_facts(NARRATIVE, ROW_LABELS, llm)

    assert facts[0].periods[0].month == "Apr"


def test_extract_structured_facts_falls_back_to_secondary_llm_on_primary_failure():
    primary = _llm_raising("primary exploded")
    fallback = _llm_with_facts([_fake_fact()])

    facts = extract_structured_facts(NARRATIVE, ROW_LABELS, primary, fallback_llm=fallback)

    assert len(facts) == 1


def test_extract_structured_facts_returns_empty_when_primary_fails_without_fallback():
    primary = _llm_raising("primary exploded")

    assert extract_structured_facts(NARRATIVE, ROW_LABELS, primary) == []


def test_extract_structured_facts_async_produces_same_result_as_sync():
    llm = _llm_with_facts([_fake_fact()])

    facts = asyncio.run(extract_structured_facts_async(NARRATIVE, ROW_LABELS, llm))

    assert len(facts) == 1
    assert facts[0].claimed_value == pytest.approx(10355.1)
    assert facts[0].page_number == 1
