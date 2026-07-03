import asyncio
from unittest.mock import Mock, patch

import pytest

from excel_parser_bi import BITableData
from paired_verifier import (
    _ExcelSource,
    _compare_absolute,
    _compare_growth_yoy,
    _deduplicate_facts,
    _unit_factor,
    verify_paired,
)
from structured_extractor import ExtractedFact


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


def _make_fact(**overrides):
    base = dict(
        metric_label="Total",
        year=2026,
        month="Apr",
        value=10355.1,
        unit="triliun Rp",
        claim_type="absolute",
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
# _compare_absolute
# ---------------------------------------------------------------------------

def test_compare_absolute_entailed_within_tolerance():
    table = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    fact = _make_fact()

    result = _compare_absolute(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.matched_excel_label == "Total"
    assert result.delta == 0.0


def test_compare_absolute_refuted_outside_tolerance():
    table = _make_table(data={("Total", 2026, "Apr"): 10000.0})
    fact = _make_fact(value=10355.1)

    result = _compare_absolute(fact, [_make_source(table)])

    assert result.verdict == "Refuted"
    assert result.delta == pytest.approx(355.1)


def test_compare_absolute_converts_units_between_pdf_and_excel():
    table = _make_table(unit="miliar Rp", data={("Total", 2026, "Apr"): 10355100.0})
    fact = _make_fact(unit="triliun Rp", value=10355.1)

    result = _compare_absolute(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.excel_value == pytest.approx(10355.1)


def test_compare_absolute_inconclusive_when_metric_not_found():
    table = _make_table(data={("Other", 2026, "Apr"): 1.0})
    fact = _make_fact(metric_label="Nonexistent Metric")

    result = _compare_absolute(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "No data found" in result.reasoning


def test_compare_absolute_inconclusive_when_units_incompatible():
    table = _make_table(unit="juta USD", data={("Total", 2026, "Apr"): 100.0})
    fact = _make_fact(unit="triliun Rp")

    result = _compare_absolute(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "not supported" in result.reasoning


def test_compare_absolute_uses_first_matching_source_in_order():
    table_without_match = _make_table(data={})
    table_with_match = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    fact = _make_fact()

    result = _compare_absolute(
        fact,
        [_make_source(table_without_match, filename="a.xls"), _make_source(table_with_match, filename="b.xls")],
    )

    assert result.verdict == "Entailed"
    assert result.matched_excel_source == "b.xls / I.1"


# ---------------------------------------------------------------------------
# _compare_growth_yoy
# ---------------------------------------------------------------------------

def test_compare_growth_yoy_entailed():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0, ("Total", 2025, "Apr"): 100.0})
    fact = _make_fact(claim_type="growth_yoy", value=10.0, unit="persen_yoy")

    result = _compare_growth_yoy(fact, [_make_source(table)])

    assert result.verdict == "Entailed"
    assert result.excel_value == pytest.approx(10.0)


def test_compare_growth_yoy_inconclusive_when_prior_year_missing():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0})
    fact = _make_fact(claim_type="growth_yoy", value=10.0)

    result = _compare_growth_yoy(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "yoy denominator" in result.reasoning


def test_compare_growth_yoy_inconclusive_when_prior_year_zero():
    table = _make_table(data={("Total", 2026, "Apr"): 110.0, ("Total", 2025, "Apr"): 0.0})
    fact = _make_fact(claim_type="growth_yoy", value=10.0)

    result = _compare_growth_yoy(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "undefined" in result.reasoning


def test_compare_growth_yoy_inconclusive_when_metric_not_found_in_any_source():
    table = _make_table(data={})
    fact = _make_fact(claim_type="growth_yoy", metric_label="Nonexistent Metric", value=10.0)

    result = _compare_growth_yoy(fact, [_make_source(table)])

    assert result.verdict == "Inconclusive"
    assert "yoy numerator" in result.reasoning


# ---------------------------------------------------------------------------
# _deduplicate_facts
# ---------------------------------------------------------------------------

def test_deduplicate_facts_keeps_first_occurrence_for_same_key():
    f1 = _make_fact(context_quote="first")
    f2 = _make_fact(context_quote="second")  # same (metric, year, month, claim_type)
    f3 = _make_fact(month="May", context_quote="third")

    result = _deduplicate_facts([f1, f2, f3])

    assert len(result) == 2
    assert result[0].context_quote == "first"
    assert result[1].month == "May"


# ---------------------------------------------------------------------------
# verify_paired (end-to-end orchestration, deps mocked)
# ---------------------------------------------------------------------------

@patch("paired_verifier.extract_structured_facts_async")
@patch("paired_verifier.extract_text_from_pdf")
@patch("paired_verifier.parse_bi_table")
def test_verify_paired_end_to_end_produces_entailed_verdict(mock_parse, mock_extract_text, mock_extract_facts):
    table = _make_table(data={("Total", 2026, "Apr"): 10355.1})
    mock_parse.return_value = table
    mock_extract_text.return_value = "[== Halaman 1 ==]\n" + "x" * 250  # long enough to skip vision fallback
    mock_extract_facts.return_value = [_make_fact()]

    response = asyncio.run(
        verify_paired(
            pdf_bytes=b"%PDF-1.4 fake",
            excel_sources=[(b"xls-bytes", "I.1", "TABEL1_1.xls")],
            llm=Mock(),
        )
    )

    assert response.total_facts == 1
    assert response.entailed_count == 1
    assert response.results[0].verdict == "Entailed"
    assert response.excel_filenames == ["TABEL1_1.xls"]
    mock_parse.assert_called_once_with(b"xls-bytes", "I.1")


@patch("paired_verifier.extract_text_from_pdf_vision_async")
@patch("paired_verifier.extract_structured_facts_async")
@patch("paired_verifier.extract_text_from_pdf")
@patch("paired_verifier.parse_bi_table")
def test_verify_paired_falls_back_to_vision_when_pdf_text_too_short(
    mock_parse, mock_extract_text, mock_extract_facts, mock_vision
):
    mock_parse.return_value = _make_table(data={})
    mock_extract_text.return_value = "too short"
    mock_vision.return_value = "[== Halaman 1 ==]\n" + "y" * 250
    mock_extract_facts.return_value = []

    vision_llm = Mock()
    response = asyncio.run(
        verify_paired(
            pdf_bytes=b"%PDF-1.4 fake",
            excel_sources=[(b"xls-bytes", "I.1", "TABEL1_1.xls")],
            llm=Mock(),
            vision_llm=vision_llm,
        )
    )

    mock_vision.assert_called_once_with(b"%PDF-1.4 fake", vision_llm)
    assert response.total_facts == 0
    assert response.results == []


@patch("paired_verifier.extract_text_from_pdf_vision_async")
@patch("paired_verifier.extract_structured_facts_async")
@patch("paired_verifier.extract_text_from_pdf")
@patch("paired_verifier.parse_bi_table")
def test_verify_paired_skips_vision_fallback_when_no_vision_llm_provided(
    mock_parse, mock_extract_text, mock_extract_facts, mock_vision
):
    mock_parse.return_value = _make_table(data={})
    mock_extract_text.return_value = "too short"
    mock_extract_facts.return_value = []

    response = asyncio.run(
        verify_paired(
            pdf_bytes=b"%PDF-1.4 fake",
            excel_sources=[(b"xls-bytes", "I.1", "TABEL1_1.xls")],
            llm=Mock(),
            vision_llm=None,
        )
    )

    mock_vision.assert_not_called()
    assert response.total_facts == 0
