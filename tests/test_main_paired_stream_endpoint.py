"""Tests for /api/verify-paired-stream — the NDJSON progress-streaming variant.

The stage events themselves come from verify_paired/check_typos, which are mocked here; what
these tests pin down is the STREAM contract the frontend depends on: progress lines arrive
before the result, the result payload matches the non-streaming endpoint, and a pipeline
failure is delivered in-band (the 200 and its headers are long gone by the time it happens).
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from schemas import (
    FactVerificationResult,
    PairedVerificationResponse,
    PeriodResult,
    TableSuggestion,
    TypoCheckResponse,
    TypoIssue,
)

client = TestClient(main.app)

_FILES = [
    ("pdf_file", ("report.pdf", b"%PDF-1.4 fake", "application/pdf")),
    ("excel_file", ("TABEL1_1.xls", b"xls-bytes", "application/vnd.ms-excel")),
]


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


def _typo_response():
    return TypoCheckResponse(
        total_issues=0,
        ejaan_count=0,
        tidak_baku_count=0,
        grammar_count=0,
        summary="Tidak ditemukan isu ejaan atau tata bahasa.",
        issues=[],
    )


def _read_events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@patch("main.check_typos")
@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_stream_emits_stage_events_then_result(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired, mock_check_typos
):
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nInflasi tumbuh 9,7% (yoy)."
    mock_verify_paired.return_value = _fact_response(total_facts=1, entailed_count=1)
    mock_check_typos.return_value = _typo_response()

    response = client.post("/api/verify-paired-stream", data={"sheet_names": "I.1"}, files=_FILES)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = _read_events(response)

    # The result is terminal: nothing may follow it, or the frontend would render early.
    assert events[-1]["type"] == "result"
    assert all(e["type"] == "stage" for e in events[:-1])

    # Progress must be reported while work happens, not batched at the end.
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert "pdf" in stages
    assert "typo" in stages

    # The payload has to match what /api/verify-paired returns, typo_check merged in.
    data = events[-1]["data"]
    assert data["total_facts"] == 1
    assert data["typo_check"]["total_issues"] == 0


@patch("main.check_typos")
@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_stream_reports_pdf_stage_done_before_result(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired, mock_check_typos
):
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nInflasi tumbuh 9,7% (yoy)."
    mock_verify_paired.return_value = _fact_response()
    mock_check_typos.return_value = _typo_response()

    response = client.post("/api/verify-paired-stream", files=_FILES)
    events = _read_events(response)

    pdf_events = [e for e in events if e.get("stage") == "pdf"]
    assert [e["status"] for e in pdf_events] == ["running", "done"]
    # The 'done' detail is what the user reads as confirmation the PDF was understood.
    assert "1 halaman" in pdf_events[-1]["detail"]


@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_stream_delivers_pipeline_failure_in_band_not_as_http_error(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired
):
    """A mid-stream failure cannot become a 400 — headers are already flushed."""
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nsome text"
    mock_verify_paired.side_effect = Exception("boom")

    response = client.post("/api/verify-paired-stream", files=_FILES)

    assert response.status_code == 200
    events = _read_events(response)
    assert events[-1]["type"] == "error"
    assert "boom" in events[-1]["detail"]
    assert not any(e["type"] == "result" for e in events)


def _rich_fact_response():
    """A response exercising the shapes the UI reads: nested models, None-valued Optionals,
    floats, and the default-factory list fields."""
    return _fact_response(
        total_facts=2,
        entailed_count=1,
        refuted_count=0,
        inconclusive_count=1,
        excel_parsers=["bi"],
        results=[
            FactVerificationResult(
                operation="value",
                metric_label="Uang Beredar (M2)",
                matched_excel_source="TABEL1_1.xls / I.1",
                periods=[PeriodResult(
                    metric_label="Uang Beredar (M2)", year=2026, month="Apr", excel_value=10355.1,
                )],
                claimed_value=10355.1,
                claimed_unit="triliun Rp",
                computed_value=10355.1,
                computed_unit="triliun Rp",
                delta=0.0,
                verdict="Entailed",
                reasoning="PDF: 10355.1 triliun Rp | Excel: 10355.1 triliun Rp",
                context_quote="M2 tercatat sebesar Rp10.355,1 triliun.",
                page_number=1,
            ),
            FactVerificationResult(
                operation="yoy_growth",
                metric_label="Kredit",
                matched_excel_source=None,
                periods=[],
                claimed_value=8.2,
                claimed_unit="persen_yoy",
                computed_value=None,
                computed_unit="persen_yoy",
                delta=None,
                verdict="Inconclusive",
                reasoning="Data tidak ditemukan.",
                context_quote="Kredit tumbuh 8,2% (yoy).",
                page_number=None,
            ),
        ],
        table_suggestions=[TableSuggestion(table="Posisi Kredit (SEKI 1.5)", metrics=["Kredit"])],
    )


def _rich_typo_response():
    return TypoCheckResponse(
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


@patch("main.check_typos")
@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_stream_result_payload_is_identical_to_the_non_streaming_endpoint(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired, mock_check_typos
):
    """The UI moved from /api/verify-paired to the stream and reuses the same renderer.

    The two endpoints serialize by different routes — FastAPI's response_model versus an
    explicit model_dump(mode="json") — so pin them together: any divergence (a dropped
    Optional, a stringified float, a missing default-factory list) would break the UI
    silently while every other test still passed.
    """
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nInflasi tumbuh 9,7% (yoy)."
    mock_verify_paired.return_value = _rich_fact_response()
    mock_check_typos.return_value = _rich_typo_response()

    plain = client.post("/api/verify-paired", files=_FILES).json()
    streamed = [e for e in _read_events(client.post("/api/verify-paired-stream", files=_FILES))
                if e["type"] == "result"][0]["data"]

    assert streamed == plain


@patch("main.check_typos")
@patch("main.verify_paired")
@patch("main.extract_narrative_text")
@patch("main.get_vision_llm")
def test_stream_shares_one_narrative_extraction_between_both_checks(
    mock_get_vision_llm, mock_extract_narrative_text, mock_verify_paired, mock_check_typos
):
    """The vision fallback is an LLM call; re-extracting per consumer would double its cost."""
    mock_get_vision_llm.side_effect = RuntimeError("no key configured")
    mock_extract_narrative_text.return_value = "[== Halaman 1 ==]\nInflasi tumbuh 9,7% (yoy)."
    mock_verify_paired.return_value = _fact_response()
    mock_check_typos.return_value = _typo_response()

    client.post("/api/verify-paired-stream", files=_FILES)

    mock_extract_narrative_text.assert_called_once()
    assert mock_verify_paired.call_args.kwargs["narrative_text"] == mock_extract_narrative_text.return_value
    assert mock_check_typos.call_args.args[0] == mock_extract_narrative_text.return_value
