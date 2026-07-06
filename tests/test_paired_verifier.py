import asyncio
from unittest.mock import Mock, patch

import pytest

from excel_parser_bi import BITableData
from paired_verifier import (
    _ExcelSource,
    _deduplicate_facts,
    _evaluate_fact,
    _unit_factor,
    verify_paired,
)
from structured_extractor import ExtractedFact, PeriodPoint


def _make_table(title="Uang Beredar (M2)", unit="triliun Rp", data=None):
    """data: {(row_label, year, month): value}"""
    table = BITableData(title=title, unit=unit, row_labels=[])
    for (label, year, month), value in (data or {}).items():
        if label not in table.row_labels:
            table.row_labels.append(label)
        table._data[(label, year, month)] = value
    return table


def _make_source(table, filename="TABEL1_1.xls", sheet="I.1"):
    return _ExcelSource(table=table, filename=filename, sheet=sheet)


def _make_period(**overrides):
    base = dict(metric_label="Total", year=2026, month="Apr")
    base.update(overrides)
    return PeriodPoint(**base)


def _make_fact(**overrides):
    periods = overrides.pop("periods", None) or [_make_period()]
    base = dict(
        operation="value",
        periods=periods,
        claimed_value=10355.1,
        unit="triliun Rp",
        context_quote="quote",
        page_number=1,
    )
    base.update(overrides)
    return ExtractedFact(**base)


# ---------------------------------------------------------------------------
# _unit_factor
# ---------------------------------------------------------------------------

def test_unit_factor_returns_none_for_unknown_pair():
    assert _unit_factor("foo", "bar") is None


def test_unit_factor_is_case_and_whitespace_insensitive():
    assert _unit_factor(" Triliun Rp ", "Miliar Rp") == 1000.0


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="value"
# ---------------------------------------------------------------------------

def test_evaluate_value_entailed_within_tolerance():
    table = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    fact = _make_fact()

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.periods[0].metric_label == "Total"
    assert result.delta == 0.0


def test_evaluate_value_refuted_outside_tolerance():
    table = _make_table(data={("Total", 2026, "Apr"): 10000.0})
    fact = _make_fact(claimed_value=10355.1)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Refuted"
    assert result.delta == pytest.approx(355.1)


def test_evaluate_value_converts_units_between_pdf_and_excel():
    table = _make_table(unit="miliar Rp", data={("Total", 2026, "Apr"): 10355100.0})
    fact = _make_fact(unit="triliun Rp", claimed_value=10355.1)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(10355.1)


def test_evaluate_value_inconclusive_when_metric_not_found():
    table = _make_table(data={("Other", 2026, "Apr"): 1.0})
    fact = _make_fact(periods=[_make_period(metric_label="Nonexistent Metric")])

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "Nonexistent Metric" in result.reasoning


def test_evaluate_value_inconclusive_when_units_incompatible():
    table = _make_table(unit="juta USD", data={("Total", 2026, "Apr"): 100.0})
    fact = _make_fact(unit="triliun Rp")

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "not supported" in result.reasoning


def test_evaluate_value_uses_first_matching_source_in_order():
    table_without_match = _make_table(data={})
    table_with_match = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    fact = _make_fact()

    result = _evaluate_fact(
        fact,
        [_make_source(table_without_match, filename="a.xls"), _make_source(table_with_match, filename="b.xls")],
    )

    assert result.verdict == "Entailed"
    assert result.matched_excel_source == "b.xls / I.1"


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="yoy_growth"
# ---------------------------------------------------------------------------

def test_evaluate_yoy_growth_entailed():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0, ("Total", 2025, "Apr"): 100.0})
    fact = _make_fact(operation="yoy_growth", claimed_value=10.0, unit="persen_yoy")

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(10.0)
    assert len(result.periods) == 2


def test_evaluate_yoy_growth_inconclusive_when_prior_year_missing():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0})
    fact = _make_fact(operation="yoy_growth", claimed_value=10.0)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "yoy denominator" in result.reasoning


def test_evaluate_yoy_growth_inconclusive_when_prior_year_zero():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0, ("Total", 2025, "Apr"): 0.0})
    fact = _make_fact(operation="yoy_growth", claimed_value=10.0)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "undefined" in result.reasoning


def test_evaluate_yoy_growth_inconclusive_when_metric_not_found_in_any_source():
    table = _make_table(data={})
    fact = _make_fact(
        operation="yoy_growth", periods=[_make_period(metric_label="Nonexistent Metric")], claimed_value=10.0
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "Nonexistent Metric" in result.reasoning


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="average" / "sum"
# ---------------------------------------------------------------------------

def _range_periods(months, metric="Total", year=2026):
    return [_make_period(metric_label=metric, year=year, month=m) for m in months]


def test_evaluate_average_entailed():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 105.0,
        ("Total", 2026, "Mar"): 98.0, ("Total", 2026, "Apr"): 110.0,
    })
    fact = _make_fact(operation="average", periods=_range_periods(["Jan", "Feb", "Mar", "Apr"]), claimed_value=103.25)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(103.25)
    assert len(result.periods) == 4


def test_evaluate_average_refuted():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 105.0,
        ("Total", 2026, "Mar"): 98.0, ("Total", 2026, "Apr"): 110.0,
    })
    fact = _make_fact(operation="average", periods=_range_periods(["Jan", "Feb", "Mar", "Apr"]), claimed_value=110.0)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Refuted"


def test_evaluate_average_inconclusive_when_one_month_missing():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 105.0, ("Total", 2026, "Mar"): 98.0,
    })
    fact = _make_fact(operation="average", periods=_range_periods(["Jan", "Feb", "Mar", "Apr"]), claimed_value=103.25)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "Total Apr 2026" in result.reasoning


def test_evaluate_sum_entailed():
    table = _make_table(data={("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 100.0})
    fact = _make_fact(operation="sum", periods=_range_periods(["Jan", "Feb"]), claimed_value=200.0)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="diff"
# ---------------------------------------------------------------------------

def test_evaluate_diff_entailed_later_minus_earlier():
    table = _make_table(data={("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Apr"): 150.0})
    fact = _make_fact(operation="diff", periods=_range_periods(["Jan", "Apr"]), claimed_value=50.0)

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="ratio"
# ---------------------------------------------------------------------------

def test_evaluate_ratio_entailed_as_percentage():
    table = _make_table(data={("Kredit", 2026, "Apr"): 850.0, ("DPK", 2026, "Apr"): 1000.0})
    fact = _make_fact(
        operation="ratio",
        periods=[_make_period(metric_label="Kredit"), _make_period(metric_label="DPK")],
        claimed_value=85.0,
        unit="persen",
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.computed_value == pytest.approx(85.0)


def test_evaluate_ratio_inconclusive_when_denominator_zero():
    table = _make_table(data={("Kredit", 2026, "Apr"): 850.0, ("DPK", 2026, "Apr"): 0.0})
    fact = _make_fact(
        operation="ratio",
        periods=[_make_period(metric_label="Kredit"), _make_period(metric_label="DPK")],
        claimed_value=85.0,
        unit="persen",
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "tidak terdefinisi" in result.reasoning


# ---------------------------------------------------------------------------
# _evaluate_fact — operation="is_increasing" / "is_decreasing" / "is_stable"
# ---------------------------------------------------------------------------

def test_evaluate_is_increasing_entailed_when_strictly_increasing():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 105.0, ("Total", 2026, "Mar"): 110.0,
    })
    fact = _make_fact(
        operation="is_increasing", periods=_range_periods(["Jan", "Feb", "Mar"]),
        claimed_value=None, unit=None,
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.claimed_value is None


def test_evaluate_is_decreasing_refuted_when_actually_increasing():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 105.0, ("Total", 2026, "Mar"): 110.0,
    })
    fact = _make_fact(
        operation="is_decreasing", periods=_range_periods(["Jan", "Feb", "Mar"]),
        claimed_value=None, unit=None,
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Refuted"


def test_evaluate_is_stable_entailed_when_within_tolerance():
    table = _make_table(data={
        ("Total", 2026, "Jan"): 100.0, ("Total", 2026, "Feb"): 100.02, ("Total", 2026, "Mar"): 100.0,
    })
    fact = _make_fact(
        operation="is_stable", periods=_range_periods(["Jan", "Feb", "Mar"]),
        claimed_value=None, unit=None,
    )

    result = _evaluate_fact(fact, [_make_source(table)])

    assert result.verdict == "Entailed"


# ---------------------------------------------------------------------------
# _deduplicate_facts
# ---------------------------------------------------------------------------

def test_deduplicate_facts_keeps_first_occurrence_for_same_key():
    f1 = _make_fact(context_quote="first")
    f2 = _make_fact(context_quote="second")  # same (operation, periods)
    f3 = _make_fact(periods=[_make_period(month="May")], context_quote="third")

    result = _deduplicate_facts([f1, f2, f3])

    assert len(result) == 2
    assert result[0].context_quote == "first"
    assert result[1].periods[0].month == "May"


# ---------------------------------------------------------------------------
# verify_paired (end-to-end orchestration, deps mocked)
# ---------------------------------------------------------------------------

# Note: PDF extraction / vision fallback is no longer verify_paired's concern - it now takes
# already-extracted narrative_text (see pdf_extraction.extract_narrative_text, tested in
# tests/test_pdf_extraction.py) so that fact-verification and typo_checker.check_typos can share
# one extraction pass instead of each re-running it.

@patch("paired_verifier.extract_structured_facts_async")
@patch("paired_verifier.parse_bi_table")
def test_verify_paired_end_to_end_produces_entailed_verdict(mock_parse, mock_extract_facts):
    table = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    mock_parse.return_value = table
    mock_extract_facts.return_value = [_make_fact()]

    response = asyncio.run(
        verify_paired(
            narrative_text="[== Halaman 1 ==]\n" + "x" * 250,
            excel_sources=[(b"xls-bytes", "I.1", "TABEL1_1.xls")],
            llm=Mock(),
        )
    )

    assert response.total_facts == 1
    assert response.entailed_count == 1
    assert response.results[0].verdict == "Entailed"
    assert response.excel_filenames == ["TABEL1_1.xls"]
    mock_parse.assert_called_once_with(b"xls-bytes", "I.1")


@patch("paired_verifier.extract_structured_facts_async")
@patch("paired_verifier.parse_bi_table")
def test_verify_paired_returns_no_facts_when_narrative_has_nothing_extractable(mock_parse, mock_extract_facts):
    mock_parse.return_value = _make_table(data={})
    mock_extract_facts.return_value = []

    response = asyncio.run(
        verify_paired(
            narrative_text="[== Halaman 1 ==]\n" + "y" * 250,
            excel_sources=[(b"xls-bytes", "I.1", "TABEL1_1.xls")],
            llm=Mock(),
        )
    )

    assert response.total_facts == 0
    assert response.results == []
