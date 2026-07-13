import asyncio
from unittest.mock import Mock, patch

from pdf_extraction import (
    extract_narrative_text,
    extract_text_from_pdf,
    extract_text_from_pdf_vision_async,
)


@patch("pdf_extraction._extract_pages_raw")
def test_extract_text_from_pdf_prefixes_each_page_with_a_marker(mock_pages):
    mock_pages.return_value = ["Page one text.", "Page two text."]

    result = extract_text_from_pdf(b"fake-pdf-bytes")

    assert result == "[== Halaman 1 ==]\nPage one text.\n\n[== Halaman 2 ==]\nPage two text."


@patch("pdf_extraction._extract_pages_raw")
def test_extract_text_from_pdf_skips_blank_pages(mock_pages):
    mock_pages.return_value = ["Real content.", "", "   "]

    result = extract_text_from_pdf(b"fake-pdf-bytes")

    assert result == "[== Halaman 1 ==]\nReal content."


@patch("pdf_extraction._extract_pages_raw")
def test_extract_text_from_pdf_returns_empty_string_when_no_text_found(mock_pages):
    # No raise here — paired_verifier.py relies on this returning a short/empty string so it
    # can trigger the vision-OCR fallback rather than catching an exception.
    mock_pages.return_value = ["", "   ", ""]

    assert extract_text_from_pdf(b"fake-pdf-bytes") == ""


@patch("pdf_extraction._extract_pages_raw")
def test_extract_text_from_pdf_returns_empty_string_when_no_pages_at_all(mock_pages):
    mock_pages.return_value = []

    assert extract_text_from_pdf(b"fake-pdf-bytes") == ""


# ---------------------------------------------------------------------------
# extract_narrative_text (pypdf, falling back to vision when too short)
# ---------------------------------------------------------------------------

@patch("pdf_extraction._count_total_pages")
@patch("pdf_extraction.extract_text_from_pdf")
def test_extract_narrative_text_skips_vision_when_pypdf_text_is_long_enough(mock_extract_text, mock_count_pages):
    mock_extract_text.return_value = "[== Halaman 1 ==]\n" + "x" * 250
    mock_count_pages.return_value = 1

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


@patch("pdf_extraction._count_total_pages")
@patch("pdf_extraction.extract_text_from_pdf_vision_async")
@patch("pdf_extraction.extract_text_from_pdf")
def test_extract_narrative_text_falls_back_to_vision_when_most_pages_are_blank(
    mock_extract_text, mock_vision, mock_count_pages
):
    # Chart-only PDF: pypdf clears MIN_USEFUL_CHARS in aggregate from just 2 of 10 pages
    # (stray chart-axis numbers), but 8 of 10 pages produced no text at all.
    mock_extract_text.return_value = (
        "[== Halaman 1 ==]\n" + "a" * 150 + "\n\n[== Halaman 5 ==]\n" + "b" * 150
    )
    mock_count_pages.return_value = 10
    mock_vision.return_value = "[== Halaman 1 ==]\nReal narrative text."

    vision_llm = Mock()
    result = asyncio.run(extract_narrative_text(b"%PDF-1.4 fake", vision_llm=vision_llm))

    mock_vision.assert_called_once_with(b"%PDF-1.4 fake", vision_llm)
    assert result == mock_vision.return_value


# ---------------------------------------------------------------------------
# extract_text_from_pdf_vision_async — page batching
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, content):
        self.content = content


def _count_images(message) -> int:
    return sum(1 for part in message.content if part.get("type") == "image_url")


class _FakeGoogleLLM:
    """Vision LLM stub whose class name contains 'Google' so it takes the Gemini batch path.

    Echoes one [== Halaman j ==] marker per image it receives, numbered 1..k regardless of the
    real page numbers, to prove the code renumbers markers by position (not by the model's N).
    """
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages[0])
        k = _count_images(messages[0])
        return _Resp("\n".join(f"[== Halaman {j} ==]\nNarasi gambar {j}." for j in range(1, k + 1)))


class _FakeLocalLLM:
    """Vision LLM stub with no 'Google'/'Groq' in its name → stays one page per call."""
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages[0])
        return _Resp("teks tanpa penanda")  # no marker → code must prepend it


def test_vision_async_batches_gemini_pages_and_renumbers_markers():
    b64 = [f"img{i}" for i in range(5)]  # 5 pages
    llm = _FakeGoogleLLM()
    with patch("pdf_extraction._render_pages_to_b64", return_value=b64):
        result = asyncio.run(extract_text_from_pdf_vision_async(b"pdf", llm, dpi=100))

    # 5 pages at 3/call → 2 calls (batches [1,2,3] and [4,5]).
    assert len(llm.calls) == 2
    assert _count_images(llm.calls[0]) == 3
    assert _count_images(llm.calls[1]) == 2
    # Every real page marker is present exactly once, renumbered by position even though the
    # model always numbered its output 1..k within each batch.
    for n in range(1, 6):
        assert result.count(f"[== Halaman {n} ==]") == 1
    # Second batch's text was attributed to real pages 4 and 5, not 1 and 2.
    assert "[== Halaman 4 ==]\nNarasi gambar 1." in result
    assert "[== Halaman 5 ==]\nNarasi gambar 2." in result


def test_vision_async_keeps_one_page_per_call_for_non_gemini_and_adds_markers():
    b64 = [f"img{i}" for i in range(3)]
    llm = _FakeLocalLLM()
    with patch("pdf_extraction._render_pages_to_b64", return_value=b64):
        result = asyncio.run(extract_text_from_pdf_vision_async(b"pdf", llm, dpi=100))

    assert len(llm.calls) == 3
    for call in llm.calls:
        assert _count_images(call) == 1
    for n in range(1, 4):
        assert f"[== Halaman {n} ==]\nteks tanpa penanda" in result


def test_vision_pages_per_call_env_override(monkeypatch):
    monkeypatch.setattr("pdf_extraction._VISION_PAGES_PER_CALL_ENV", 2)
    b64 = [f"img{i}" for i in range(5)]
    llm = _FakeLocalLLM()
    with patch("pdf_extraction._render_pages_to_b64", return_value=b64):
        asyncio.run(extract_text_from_pdf_vision_async(b"pdf", llm, dpi=100))

    # Env forces 2 pages/call even for the local provider → 3 calls for 5 pages.
    assert len(llm.calls) == 3
