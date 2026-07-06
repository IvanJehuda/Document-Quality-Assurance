from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from schemas import PairedVerificationResponse, TypoCheckResponse, TypoIssue

client = TestClient(main.app)


def _fact_response(**overrides):
    base = dict(
        pdf_filename="report.pdf",
        excel_filenames=["TABEL1_1.xls"],
        excel_sheets=["I.1"],
        excel_units=["triliun Rp"],
        total_facts=0,
        entailed_count=0,
        refuted_count=0,
        inconclusive_count=0,
        results=[],
    )
    base.update(overrides)
    return PairedVerificationResponse(**base)


@patch("main.check_typos")
@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_verify_paired_endpoint_merges_fact_and_typo_results(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired, mock_check_typos
):
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nInflasi tumbuh 9,7% (yoy)."
    mock_verify_paired.return_value = _fact_response(total_facts=1, entailed_count=1)
    mock_check_typos.return_value = TypoCheckResponse(
        total_issues=1,
        ejaan_count=0,
        tidak_baku_count=1,
        grammar_count=0,
        summary="Ditemukan 1 isu: 0 ejaan, 1 kata tidak baku, 0 tata bahasa.",
        issues=[TypoIssue(
            word="Resiko", start=0, end=6, category="tidak_baku",
            suggestion="risiko", explanation="bentuk tidak baku", page_number=1,
        )],
    )

    response = client.post(
        "/api/verify-paired",
        data={"sheet_names": "I.1"},
        files=[
            ("pdf_file", ("report.pdf", b"%PDF-1.4 fake", "application/pdf")),
            ("excel_file", ("TABEL1_1.xls", b"xls-bytes", "application/vnd.ms-excel")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_facts"] == 1
    assert body["typo_check"]["total_issues"] == 1
    assert body["typo_check"]["issues"][0]["suggestion"] == "risiko"

    # Both checks must share the one extracted narrative_text rather than re-extracting.
    mock_verify_paired.assert_called_once()
    assert mock_verify_paired.call_args.kwargs["narrative_text"] == mock_extract_narrative_text.return_value
    mock_check_typos.assert_called_once()
    assert mock_check_typos.call_args.args[0] == mock_extract_narrative_text.return_value


@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_verify_paired_endpoint_returns_400_when_pipeline_fails(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired
):
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nsome text"
    mock_verify_paired.side_effect = Exception("boom")

    response = client.post(
        "/api/verify-paired",
        files=[
            ("pdf_file", ("report.pdf", b"%PDF-1.4 fake", "application/pdf")),
            ("excel_file", ("TABEL1_1.xls", b"xls-bytes", "application/vnd.ms-excel")),
        ],
    )

    assert response.status_code == 400
