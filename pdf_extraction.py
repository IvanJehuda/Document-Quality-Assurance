"""Text extraction from PDFs.

Two strategies:
  1. extract_text_from_pdf      — fast path: pypdf text-layer extraction (no LLM, no cost).
                                   Works for digital PDFs with a real text layer.
  2. extract_text_from_pdf_vision — fallback: render pages to images via pypdfium2, then ask
                                   a vision-capable LLM to read out the narrative text only.
                                   Used when the PDF has text stored as vector paths (outlined
                                   fonts) so pypdf returns empty or near-empty output.

extract_narrative_text() combines both: tries pypdf first, falls back to the async vision
path automatically when the extracted text is below a minimum useful length. Callers that need
the narrative text for more than one downstream step (e.g. paired_verifier.py's fact-checking
AND typo_checker.py's spelling/grammar check) should call this once and reuse the result, rather
than re-extracting per consumer — the vision fallback is an LLM call and re-running it doubles
cost and rate-limit exposure for no benefit.
"""

import asyncio
import base64
import io
import logging
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pypdf import PdfReader

logger = logging.getLogger("fact-checker")

# pypdf result shorter than this is treated as "extraction failed" and triggers the vision fallback.
MIN_USEFUL_CHARS = 200

# Marks the start of each page's text in extract_text_from_pdf's/vision's output.
# Shared across modules that need to strip markers or attribute a char offset to a page.
PAGE_MARKER_RE = re.compile(r'\[== Halaman (\d+) ==\]')

_VISION_PROMPT = """\
Halaman-halaman berikut adalah laporan statistik Bank Indonesia.

Tugasmu: ekstrak HANYA teks narasi dari setiap halaman.

Yang DIAMBIL:
- Paragraf teks bahasa Indonesia (kalimat lengkap)
- Bullet point yang berisi pernyataan naratif
- Angka yang muncul di dalam kalimat narasi \
(contoh: "M2 tumbuh 10,8% (yoy) mencapai Rp10.415,9 triliun")

Yang DIABAIKAN sepenuhnya:
- Grafik dan chart (sumbu, legenda, label titik data)
- Tabel data (baris/kolom berisi angka-angka)
- Judul halaman, header, footer, nomor halaman
- Keterangan/footnote di bawah tabel
- Nama departemen, tanggal publikasi

FORMAT OUTPUT:
- Awali setiap halaman dengan baris penanda: [== Halaman N ==]  (N = nomor urut, mulai dari 1)
- Lalu tuliskan teks narasi halaman tersebut, paragraf dipisah baris kosong
- Jika halaman tidak mengandung narasi, tetap tulis penanda halamannya lalu kosongkan isinya
"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract and concatenate text from every page of a PDF via pypdf.

    Each page is prefixed with a [== Halaman N ==] marker so downstream
    components (structured extractor) can attribute facts to specific pages.
    Returns the concatenated text (may be empty/short for text-as-paths PDFs).
    Lets pypdf's own parsing errors propagate for corrupt/non-PDF input.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[== Halaman {i} ==]\n{text}")
    return "\n\n".join(parts)


def extract_text_from_pdf_vision(
    file_bytes: bytes,
    vision_llm: BaseChatModel,
    dpi: int = 100,
) -> str:
    """Extract narrative text from a PDF by rendering pages as images and using a vision LLM.

    Intended as a fallback for PDFs where pypdf returns little or no text (text stored as
    vector paths / outlined fonts). Because the source is a digital PDF, rendering at 100 DPI
    produces clean, noise-free images — sufficient for accurate text recognition.

    Args:
        file_bytes:  Raw PDF bytes.
        vision_llm:  A vision-capable LangChain chat model (use llm_provider.get_vision_llm()).
        dpi:         Render resolution. 100 DPI balances token cost vs. readability.

    Returns:
        Narrative text extracted from the PDF pages, or empty string on failure.
    """
    try:
        import pypdfium2 as pdfium  # installed as pdfplumber dependency
    except ImportError:
        raise RuntimeError(
            "pypdfium2 is required for vision-based PDF extraction. "
            "Install it with: pip install pypdfium2"
        )

    # Render all pages to PNG bytes
    logger.info("Rendering PDF pages at %d DPI for vision extraction", dpi)
    doc = pdfium.PdfDocument(file_bytes)
    scale = dpi / 72  # PDF coordinate space is 72 pt/inch

    image_parts = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    logger.info("Rendered %d pages, sending to vision LLM", len(image_parts))

    message = HumanMessage(content=image_parts + [{"type": "text", "text": _VISION_PROMPT}])

    try:
        response = vision_llm.invoke([message])
        text = response.content if isinstance(response.content, str) else ""
        logger.info("Vision extraction returned %d chars", len(text))
        return text
    except Exception:
        logger.exception("Vision-based PDF extraction failed")
        return ""


# Complex narrative-only prompt — works well for Gemini which reliably follows nuanced instructions.
_VISION_PROMPT_SINGLE_PAGE = """\
Halaman ini adalah bagian dari laporan statistik Bank Indonesia.

Tugasmu: salin SEMUA kalimat dan paragraf narasi dari halaman ini secara verbatim dan lengkap.
Jangan ringkas, jangan potong, jangan parafrase.

Yang WAJIB disertakan:
- Semua paragraf dan kalimat (bahasa Indonesia maupun Inggris)
- Semua angka, persentase, dan nilai moneter yang muncul DALAM kalimat
- Bullet point dan pernyataan singkat
- Sub-judul section (seperti "PERKEMBANGAN M2", "FAKTOR-FAKTOR...")

Yang HARUS DILEWATI sepenuhnya — jangan tulis sama sekali:
- Isi tabel data (baris berisi deretan angka, baik dengan karakter | maupun tidak)
- JANGAN gunakan format tabel markdown (|col|col|) dalam output apapun
- Label sumbu grafik, legenda chart, judul grafik
- Nomor halaman, nama departemen, tanggal publikasi

PENTING:
- Jika ada kalimat yang MENJELASKAN atau MERUJUK ke tabel/grafik, kalimat itu TETAP disertakan
- Jangan gunakan format | pipe | untuk apapun dalam output
- Jika halaman hanya berisi grafik atau tabel tanpa kalimat narasi, balas kosong
"""

# Simple extraction prompt for models that struggle with complex selective-inclusion instructions
# (e.g. Groq llama-4-scout).  Extract everything verbatim; tabular rows are filtered in
# _strip_tabular_content() deterministically instead of relying on the model to skip them.
_VISION_PROMPT_SIMPLE = """\
Read this page from a Bank Indonesia report and copy all visible text exactly as it appears.

Include paragraphs, headings, sentences, numbers, bullet points, and footnotes.
Do not summarize, rephrase, or skip anything. Copy the text directly.
If the page has no text at all (only images or blank areas), output nothing.
"""


def _get_page_prompt(vision_llm: BaseChatModel) -> str:
    """Select extraction prompt based on the vision LLM class."""
    if "Groq" in type(vision_llm).__name__:
        return _VISION_PROMPT_SIMPLE
    return _VISION_PROMPT_SINGLE_PAGE


# Indonesian and English month abbreviations used in BI statistical tables.
_MONTH_ABBREVS = frozenset({
    "jan", "feb", "mar", "apr", "mei", "may", "jun", "jul",
    "ags", "agu", "aug", "sep", "okt", "oct", "nov", "des", "dec",
})

# A token that is purely numeric (integer, decimal with comma/period, optional %).
_NUMERIC_RE = re.compile(r'^-?[\d.,]+%?$')


def _strip_tabular_content(text: str) -> str:
    """Remove tabular data rows from vision LLM output.

    Catches both:
    - Markdown table rows (lines starting with |)
    - Non-pipe tabular rows: lines where ≥ 65% of tokens are numeric or month abbreviations
      (e.g. "6,2  Mar  6,1  Apr  5,2  Mei  4,9  Jun  6,4  Jul" from BI time-series lampiran).
    """
    lines = text.split('\n')
    result = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        # Markdown table rows and separator rows
        if stripped.startswith('|'):
            continue
        # Non-pipe tabular data rows: predominantly numbers + month abbreviations
        tokens = stripped.split()
        if len(tokens) >= 8:
            tabular_count = sum(
                1 for t in tokens
                if bool(_NUMERIC_RE.match(t)) or t.lower() in _MONTH_ABBREVS
            )
            if tabular_count / len(tokens) >= 0.65:
                continue
        is_blank = not stripped
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    return '\n'.join(result)


def _render_pages_to_b64(file_bytes: bytes, dpi: int) -> list[str]:
    """Render every PDF page to a base64-encoded PNG string (CPU-bound, call in a thread)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError(
            "pypdfium2 is required for vision-based PDF extraction. "
            "Install it with: pip install pypdfium2"
        )
    doc = pdfium.PdfDocument(file_bytes)
    scale = dpi / 72
    result = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


async def extract_text_from_pdf_vision_async(
    file_bytes: bytes,
    vision_llm: BaseChatModel,
    dpi: int = 150,
) -> str:
    """Async, per-page vision extraction — each page is a separate parallel LLM call.

    Fires one ainvoke per page and gathers them concurrently, giving O(1 page) wall-clock
    time regardless of document length.  The extraction prompt is chosen per-provider:
    Gemini gets a selective narrative-only prompt (it follows it reliably); Groq gets a
    simple "copy everything" prompt and tabular rows are stripped deterministically in
    _strip_tabular_content() instead.

    Both free-tier providers throttle per request, not just per token: Groq's account-wide
    TPM budget is easily blown by a handful of image-heavy calls, and Gemini's free tier caps
    gemini-2.5-flash at 5 requests/minute — a 9-page document sent "in parallel" instantly
    exceeds both. A semaphore caps concurrency per provider and failed calls are retried with
    backoff parsed from the provider's own suggested wait time when a rate-limit error is hit.
    """
    logger.info("Rendering PDF pages at %d DPI for vision extraction (async per-page)", dpi)
    b64_pages = await asyncio.to_thread(_render_pages_to_b64, file_bytes, dpi)
    logger.info("Rendered %d pages, sending to vision LLM in parallel", len(b64_pages))

    prompt = _get_page_prompt(vision_llm)
    provider_name = type(vision_llm).__name__
    is_groq = "Groq" in provider_name
    is_gemini = "Google" in provider_name
    # Groq: ~7-10k tokens/page against a 30k TPM account-wide budget -> serialize (semaphore=1).
    # Gemini free tier: hard cap of 5 requests/minute for gemini-2.5-flash -> cap at 4 concurrent
    # to leave headroom. Other providers (e.g. local Ollama) are assumed unthrottled.
    if is_groq:
        semaphore = asyncio.Semaphore(1)
        max_retries = 6
    elif is_gemini:
        semaphore = asyncio.Semaphore(4)
        max_retries = 6
    else:
        semaphore = asyncio.Semaphore(len(b64_pages) + 1)
        max_retries = 1

    # Groq says "Please try again in 9.08s"; Gemini says "Please retry in 51.95s".
    _RETRY_AFTER_RE = re.compile(r'[Pp]lease (?:try again|retry) in (\d+\.?\d*)s')

    def _parse_retry_after(err: str, fallback: float) -> float:
        """Extract the provider's suggested wait time from a rate-limit error message."""
        m = _RETRY_AFTER_RE.search(err)
        return float(m.group(1)) + 1.0 if m else fallback

    def _is_rate_limit(exc: Exception) -> bool:
        cls = type(exc).__name__
        if cls == "RateLimitError":
            return True
        err = str(exc).lower()
        return any(kw in err for kw in ("rate", "429", "too many", "quota"))

    async def _extract_page(page_num: int, b64: str) -> str:
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        retry_wait = 0.0
        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(retry_wait)
            async with semaphore:
                try:
                    msg = HumanMessage(content=[image_part, {"type": "text", "text": prompt}])
                    response = await vision_llm.ainvoke([msg])
                    text = response.content if isinstance(response.content, str) else ""
                    text = _strip_tabular_content(text)
                    return (
                        f"[== Halaman {page_num} ==]\n{text}"
                        if text.strip()
                        else f"[== Halaman {page_num} ==]"
                    )
                except Exception as exc:
                    if _is_rate_limit(exc) and attempt < max_retries - 1:
                        retry_wait = _parse_retry_after(str(exc), fallback=(attempt + 1) * 10)
                        logger.warning(
                            "Rate limit on page %d (attempt %d/%d), retrying in %.1fs",
                            page_num, attempt + 1, max_retries, retry_wait,
                        )
                    else:
                        logger.exception("Vision extraction failed for page %d", page_num)
                        return f"[== Halaman {page_num} ==]"
        logger.error("Vision extraction gave up on page %d after %d retries", page_num, max_retries)
        return f"[== Halaman {page_num} ==]"

    page_texts = await asyncio.gather(
        *[_extract_page(i + 1, b64) for i, b64 in enumerate(b64_pages)]
    )
    combined = "\n\n".join(t for t in page_texts if t.strip())
    logger.info("Async vision extraction returned %d chars total", len(combined))
    return combined


async def extract_narrative_text(file_bytes: bytes, vision_llm: Optional[BaseChatModel] = None) -> str:
    """Extract a PDF's narrative text, falling back to vision extraction when pypdf's output is too short.

    Single entry point for any caller that needs the full narrative text (with page markers) —
    call this once and share the result across multiple downstream consumers instead of extracting
    per-consumer, since the vision fallback is an LLM call with its own cost and rate limits.
    """
    text = extract_text_from_pdf(file_bytes)
    content_chars = len(PAGE_MARKER_RE.sub('', text).strip())
    if content_chars < MIN_USEFUL_CHARS:
        if vision_llm is None:
            logger.warning(
                "PDF content too short (%d chars excl. markers) and no vision_llm provided — "
                "pass vision_llm=get_vision_llm() to enable the fallback.",
                content_chars,
            )
        else:
            logger.info(
                "PDF content too short (%d chars excl. markers), falling back to vision extraction",
                content_chars,
            )
            text = await extract_text_from_pdf_vision_async(file_bytes, vision_llm)
    return text
