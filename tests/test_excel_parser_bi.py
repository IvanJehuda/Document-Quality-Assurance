import io

import pytest
from openpyxl import Workbook

from excel_parser_bi import MONTH_ABBREVS, parse_bi_table


def _save(wb) -> bytes:
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_bi_workbook_bytes(sheet_name="I.1", title="Uang Beredar (M2)", unit="(Miliar Rp)"):
    """Two-year, three-month-each BI table: title/unit/blank/year-row/month-row/2 data rows.

    2023 block occupies cols D-F (Jan-Mar), 2024 block occupies cols G-I (Jan-Mar),
    matching the real TABEL1_1.xls layout (col0=row#, col1=indent, col2=label, col3+=values).
    """
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append([title])
    ws.append([unit])
    ws.append([])
    ws.append([None, None, None, 2023, None, None, 2024])
    ws.append([None, None, None, "Jan", "Feb", "Mar", "Jan", "Feb", "Mar"])
    ws.append([1, None, "Total", 100.0, 110.0, 120.0, 200.0, 210.0, 220.0])
    ws.append([2, None, "M1", 50.0, 55.0, 60.0, 90.0, 95.0, 100.0])
    return _save(wb)


def test_parse_bi_table_reads_title_unit_and_row_labels():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    assert result.title == "Uang Beredar (M2)"
    assert result.unit == "Miliar Rp"
    assert result.row_labels == ["Total", "M1"]


def test_lookup_returns_exact_value_for_each_year_month():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    assert result.lookup("Total", 2023, "Jan") == 100.0
    assert result.lookup("Total", 2024, "Mar") == 220.0
    assert result.lookup("M1", 2023, "Feb") == 55.0


def test_lookup_returns_none_for_missing_label_or_period():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    assert result.lookup("Nonexistent", 2023, "Jan") is None
    assert result.lookup("Total", 2023, "Dec") is None


def test_lookup_fuzzy_matches_via_substring():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    matched_label, value = result.lookup_fuzzy("Tota", 2023, "Jan")

    assert matched_label == "Total"
    assert value == 100.0


def test_lookup_fuzzy_falls_back_to_total_row_when_query_matches_table_subject():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    # "Uang Beredar" doesn't match any row label directly, but it does share
    # significant words with the table title "Uang Beredar (M2)".
    matched_label, value = result.lookup_fuzzy("Uang Beredar", 2023, "Jan")

    assert matched_label == "Total"
    assert value == 100.0


def test_lookup_fuzzy_returns_none_when_nothing_matches():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    matched_label, value = result.lookup_fuzzy("Completely Unrelated Metric", 2023, "Jan")

    assert matched_label is None
    assert value is None


def test_available_periods_returns_all_periods_for_matched_label():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    periods = result.available_periods("Total")

    assert sorted(periods) == sorted(
        [(2023, "Jan"), (2023, "Feb"), (2023, "Mar"), (2024, "Jan"), (2024, "Feb"), (2024, "Mar")]
    )


def test_available_periods_returns_empty_list_when_label_unresolvable():
    result = parse_bi_table(_build_bi_workbook_bytes(), "I.1")

    assert result.available_periods("Nonexistent Metric") == []


def test_year_anchor_at_december_column_still_maps_all_months_correctly():
    # BI tables sometimes anchor the year label at the December cell instead of January -
    # the column-period mapper must back-propagate to find January regardless.
    wb = Workbook()
    ws = wb.active
    ws.title = "I.1"
    ws.append(["Title"])
    ws.append(["(Miliar Rp)"])
    ws.append([])
    year_row = [None, None, None] + [None] * 11 + [2023]
    month_row = [None, None, None] + MONTH_ABBREVS
    values = [float(100 + i) for i in range(12)]
    ws.append(year_row)
    ws.append(month_row)
    ws.append([1, None, "Total"] + values)

    result = parse_bi_table(_save(wb), "I.1")

    assert result.lookup("Total", 2023, "Jan") == 100.0
    assert result.lookup("Total", 2023, "Dec") == 111.0


def test_parse_bi_table_raises_when_sheet_not_found():
    data = _build_bi_workbook_bytes(sheet_name="I.1")

    with pytest.raises(ValueError, match="not found"):
        parse_bi_table(data, "WrongSheet")


def test_parse_bi_table_raises_when_no_year_header_present():
    wb = Workbook()
    ws = wb.active
    ws.title = "I.1"
    ws.append(["Title"])
    ws.append(["(Miliar Rp)"])
    for _ in range(8):
        ws.append(["no year header here"])

    with pytest.raises(ValueError, match="auto-detect year/month header rows"):
        parse_bi_table(_save(wb), "I.1")


def test_parse_bi_table_raises_on_unrecognized_file_format():
    with pytest.raises(ValueError, match="Unrecognized file format"):
        parse_bi_table(b"not an excel file at all", "I.1")
