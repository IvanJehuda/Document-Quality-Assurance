import asyncio
from unittest.mock import Mock, patch

from pdf_extraction import extract_narrative_text, extract_text_from_pdf


def _reader_with_pages(texts):
    pages = []
    for text in texts:
        page = Mock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = Mock()
    reader.pages = pages
    return reader


@patch("pdf_extraction.PdfReader")
def test_extract_text_from_pdf_prefixes_each_page_with_a_marker(mock_pdf_reader):
    mock_pdf_reader.return_value = _reader_with_pages(["Page one text.", "Page two text."])

    result = extract_text_from_pdf(b"fake-pdf-bytes")

    assert result == "[== Halaman 1 ==]\nPage one text.\n\n[== Halaman 2 ==]\nPage two text."


@patch("pdf_extraction.PdfReader")
def test_extract_text_from_pdf_skips_blank_pages(mock_pdf_reader):
    mock_pdf_reader.return_value = _reader_with_pages(["Real content.", "", "   "])

    result = extract_text_from_pdf(b"fake-pdf-bytes")

    assert result == "[== Halaman 1 ==]\nReal content."


@patch("pdf_extraction.PdfReader")
def test_extract_text_from_pdf_returns_empty_string_when_no_text_found(mock_pdf_reader):
    # No raise here — paired_verifier.py relies on this returning a short/empty string so it
    # can trigger the vision-OCR fallback rather than catching an exception.
    mock_pdf_reader.return_value = _reader_with_pages(["", "   ", None])

    assert extract_text_from_pdf(b"fake-pdf-bytes") == ""


@patch("pdf_extraction.PdfReader")
def test_extract_text_from_pdf_returns_empty_string_when_no_pages_at_all(mock_pdf_reader):
    mock_pdf_reader.return_value = _reader_with_pages([])

    assert extract_text_from_pdf(b"fake-pdf-bytes") == ""


# ---------------------------------------------------------------------------
# extract_narrative_text (pypdf, falling back to vision when too short)
# ---------------------------------------------------------------------------

@patch("pdf_extraction.extract_text_from_pdf")
def test_extract_narrative_text_skips_vision_when_pypdf_text_is_long_enough(mock_extract_text):
    mock_extract_text.return_value = "[== Halaman 1 ==]\n" + "x" * 250

    result = asyncio.run(extract_narrative_text(b"%PDF-1.4 fake", vision_llm=Mock()))

    assert result == mock_extract_text.return_value


@patch("pdf_extraction.extract_text_from_pdf_vision_async")
@patch("pdf_extraction.extract_text_from_pdf")
def test_extract_narrative_text_falls_back_to_vision_when_pypdf_text_too_short(mock_extract_text, mock_vision):
    mock_extract_text.return_value = "too short"
    mock_vision.return_value = "[== Halaman 1 ==]\n" + "y" * 250

    vision_llm = Mock()
    result = asyncio.run(extract_narrative_text(b"%PDF-1.4 fake", vision_llm=vision_llm))

    mock_vision.assert_called_once_with(b"%PDF-1.4 fake", vision_llm)
    assert result == mock_vision.return_value


@patch("pdf_extraction.extract_text_from_pdf_vision_async")
@patch("pdf_extraction.extract_text_from_pdf")
def test_extract_narrative_text_skips_vision_fallback_when_no_vision_llm_provided(mock_extract_text, mock_vision):
    mock_extract_text.return_value = "too short"

    result = asyncio.run(extract_narrative_text(b"%PDF-1.4 fake", vision_llm=None))

    mock_vision.assert_not_called()
    assert result == "too short"
