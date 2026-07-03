from unittest.mock import Mock, patch

from pdf_extraction import extract_text_from_pdf


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
