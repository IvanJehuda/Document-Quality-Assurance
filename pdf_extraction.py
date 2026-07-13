"""Text extraction from PDFs.

Two strategies:
  1. extract_text_from_pdf      — fast path: text-layer extraction via pypdfium2 (no LLM, no
                                   cost). Works for digital PDFs with a real text layer.
  2. extract_text_from_pdf_vision — fallback: render pages to images via pypdfium2, then ask
                                   a vision-capable LLM to read out the narrative text only.
                                   Used when the PDF has text stored as vector paths (outlined
                                   fonts) so the text layer is empty or near-empty.

extract_narrative_text() combines both: tries the text layer first, falls back to the async
vision path automatically when the extracted text is below a minimum useful length. Callers that
need the narrative text for more than one downstream step (e.g. paired_verifier.py's fact-checking
AND typo_checker.py's spelling/grammar check) should call this once and reuse the result, rather
than re-extracting per consumer — the vision fallback is an LLM call and re-running it doubles
cost and rate-limit exposure for no benefit.

Text-layer extraction uses pypdfium2 (PDFium, Chromium's PDF engine), NOT pypdf: pypdf inserts
phantom spaces at text-run boundaries in some PDFs — splitting words ("bershih"→"bers hih") and
numbers ("21,3%"→"2 1,3%"). Those splits both flooded the typo checker with bogus fragments and
made structured_extractor drop facts whose value string ("2 1,3") could not be parsed. PDFium
reconstructs word/number boundaries reliably on the same files.
"""

import asyncio
import base64
import io
import logging
import os
import re
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger("fact-checker")

# Text-layer result shorter than this is treated as "extraction failed" and triggers the vision fallback.
MIN_USEFUL_CHARS = 200

# How many PDF pages to send per vision LLM call. Batching cuts the number of requests, which
# is what matters when the provider throttles by requests-per-minute rather than by tokens:
# Gemini's free tier caps gemini-2.5-flash at ~5 requests/minute, so 10 per-page calls need
# several rate-limit windows (~2 min), whereas 3 pages/call fits 10 pages into ~4 calls (one
# window). 0/unset = per-provider auto: Gemini batches; Groq (throttled by tokens-per-minute,
# where more images per call blows the budget faster) and local/other providers stay 1/call.
_VISION_PAGES_PER_CALL_ENV = int(os.getenv("VISION_PAGES_PER_CALL", "0"))

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


def _extract_pages_raw(file_bytes: bytes) -> List[str]:
    """Return the raw text of each page via pypdfium2 (PDFium's text layer).

    Separated out as the single seam over the PDF engine so extract_text_from_pdf stays pure
    string-shaping (and is trivially testable). PDFium is used instead of pypdf because pypdf
    over-inserts spaces at text-run boundaries (see module docstring). Lets PDFium's own errors
    propagate for corrupt/non-PDF input.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(file_bytes)
    try:
        pages: List[str] = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            pages.append((textpage.get_text_range() or "").replace("\r\n", "\n"))
            textpage.close()
            page.close()
        return pages
    finally:
        doc.close()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract and concatenate the text layer of every page of a PDF (via pypdfium2).

    Each page is prefixed with a [== Halaman N ==] marker so downstream
    components (structured extractor) can attribute facts to specific pages.
    Returns the concatenated text (may be empty/short for text-as-paths PDFs).
    """
    parts = []
    for i, text in enumerate(_extract_pages_raw(file_bytes), start=1):
        if text and text.strip():
            parts.append(f"[== Halaman {i} ==]\n{text}")
    return "\n\n".join(parts)


def _count_total_pages(file_bytes: bytes) -> int:
    """Total page count, used to detect PDFs where most pages have no text layer at all.

    A document can clear MIN_USEFUL_CHARS in aggregate while still being almost entirely
    image/chart pages with no extractable text (e.g. a handful of stray chart-axis numbers
    on 2 of 10 pages) - that case needs the vision fallback just as much as a uniformly
    short document does, but a pure char-count check over the whole document can't see it.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(file_bytes)
    try:
        return len(doc)
    finally:
        doc.close()


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
        import pypdfium2 as pdfium
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


# Appended when several pages are sent in one call, so the model separates each page's text
# with its own marker. Images are handed over each preceded by their [== Halaman N ==] marker.
_MULTIPAGE_SUFFIX_ID = """\

CATATAN MULTI-HALAMAN:
- Kamu menerima BEBERAPA halaman sekaligus; setiap gambar didahului baris penanda [== Halaman N ==].
- Untuk SETIAP halaman, tulis ulang baris penanda [== Halaman N ==] (dengan nomor N yang sama persis),
  lalu teks narasi halaman itu di bawahnya.
- Proses semua halaman sesuai urutan; jangan lewati satu halaman pun.
"""

_MULTIPAGE_SUFFIX_EN = """\

MULTI-PAGE NOTE:
- You are given SEVERAL pages at once; each image is preceded by a [== Halaman N ==] marker line.
- For EACH page, repeat its [== Halaman N ==] marker line (same number), then that page's text below it.
- Process every page in order; do not skip any page.
"""


def _get_page_prompt(vision_llm: BaseChatModel) -> str:
    """Select extraction prompt based on the vision LLM class."""
    if "Groq" in type(vision_llm).__name__:
        return _VISION_PROMPT_SIMPLE
    return _VISION_PROMPT_SINGLE_PAGE


def _get_batch_prompt(vision_llm: BaseChatModel, multi: bool) -> str:
    """Extraction prompt, with the multi-page marker instruction appended when batching."""
    base = _get_page_prompt(vision_llm)
    if not multi:
        return base
    suffix = _MULTIPAGE_SUFFIX_EN if "Groq" in type(vision_llm).__name__ else _MULTIPAGE_SUFFIX_ID
    return base + suffix


def _renumber_markers(text: str, page_nums: List[int]) -> str:
    """Attach the correct [== Halaman N ==] markers to a (possibly multi-page) vision response.

    Images are always sent in page order, so page attribution follows marker order, not the
    number the model happened to print. When the model emitted exactly one marker per expected
    page, each marker is rewritten to the expected page number by position. Otherwise the
    segmentation is unreliable: the batch's text is attributed to its first page (so no text is
    lost) and the remaining pages are appended as empty markers to keep the page set complete.
    """
    markers = list(PAGE_MARKER_RE.finditer(text))
    if len(markers) == len(page_nums):
        out: List[str] = []
        last = 0
        for m, n in zip(markers, page_nums):
            out.append(text[last:m.start()])
            out.append(f"[== Halaman {n} ==]")
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    body = PAGE_MARKER_RE.sub("", text).strip()
    if not body:
        return "\n".join(f"[== Halaman {n} ==]" for n in page_nums)
    if len(page_nums) > 1:
        logger.warning(
            "Vision batch %s returned %d marker(s) (expected %d); attributing text to page %d",
            page_nums, len(markers), len(page_nums), page_nums[0],
        )
    extra = "".join(f"\n[== Halaman {n} ==]" for n in page_nums[1:])
    return f"[== Halaman {page_nums[0]} ==]\n{body}{extra}"


# Indonesian and English month abbreviations used in BI statistical tables.
_MONTH_ABBREVS = frozenset({
    "jan", "feb", "mar", "apr", "mei", "may", "jun", "jul",
    "ags", "agu", "aug", "sep", "okt", "oct", "nov", "des", "dec",
})

# A token that is purely numeric (integer, decimal with comma/period, optional %).
_NUMERIC_RE = re.compile(r'^-?[\d.,]+%?$')

# A run of 4+ spaces between two non-space characters: the inter-column gap that
# justified PDF tables leave behind. Real prose never has interior 4+ space gaps.
_JUSTIFIED_GAP_RE = re.compile(r'\S {4,}\S')


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
        # Justified table rows: the big inter-column gaps (4+ spaces) that pypdf
        # reproduces never occur in real prose, and they come with words split
        # apart ("Tagi han", "Akti va", "El ektroni k") that would otherwise flood
        # the typo checker with bogus candidates. Checked on the stripped line so
        # trailing/leading whitespace on a genuine sentence doesn't trip it.
        if _JUSTIFIED_GAP_RE.search(stripped):
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


def _plan_pages_per_call(is_groq: bool, is_gemini: bool) -> int:
    """How many pages to batch per vision call for this provider (env override wins).

    Batching only helps a request-per-minute cap (Gemini free tier), so only Gemini batches by
    default. Groq is throttled by tokens-per-minute — more images per call blows that budget
    faster, not slower — and local/other providers already parallelize freely, so both stay 1.
    """
    if _VISION_PAGES_PER_CALL_ENV > 0:
        return _VISION_PAGES_PER_CALL_ENV
    return 3 if is_gemini else 1


async def extract_text_from_pdf_vision_async(
    file_bytes: bytes,
    vision_llm: BaseChatModel,
    dpi: int = 150,
) -> str:
    """Async vision extraction — pages are batched into as few LLM calls as the provider allows.

    Pages are grouped into batches (see _plan_pages_per_call) and each batch is one ainvoke;
    batches are gathered concurrently. Fewer, larger calls beat many small ones under a
    requests-per-minute cap (Gemini free tier: 5 rpm), while token-throttled Groq and local
    providers keep one page per call. The extraction prompt is chosen per-provider: Gemini gets
    a selective narrative-only prompt (it follows it reliably); Groq gets a simple "copy
    everything" prompt and tabular rows are stripped deterministically in _strip_tabular_content.

    Both free-tier providers throttle at the request level: Groq's account-wide TPM budget is
    easily blown by image-heavy calls, and Gemini's free tier caps gemini-2.5-flash at 5
    requests/minute. A semaphore caps concurrency per provider and failed calls are retried with
    backoff parsed from the provider's own suggested wait time when a rate-limit error is hit.
    """
    logger.info("Rendering PDF pages at %d DPI for vision extraction (async, batched)", dpi)
    b64_pages = await asyncio.to_thread(_render_pages_to_b64, file_bytes, dpi)
    n_pages = len(b64_pages)

    provider_name = type(vision_llm).__name__
    is_groq = "Groq" in provider_name
    is_gemini = "Google" in provider_name

    pages_per_call = _plan_pages_per_call(is_groq, is_gemini)
    prompt = _get_batch_prompt(vision_llm, multi=pages_per_call > 1)

    # Batches of 0-based page indices, in order.
    batches = [
        list(range(i, min(i + pages_per_call, n_pages)))
        for i in range(0, n_pages, pages_per_call)
    ]
    logger.info(
        "Rendered %d page(s); sending as %d vision call(s) (%d page(s)/call)",
        n_pages, len(batches), pages_per_call,
    )

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
        semaphore = asyncio.Semaphore(len(batches) + 1)
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

    def _empty_markers(page_nums: List[int]) -> str:
        return "\n".join(f"[== Halaman {n} ==]" for n in page_nums)

    async def _extract_batch(idxs: List[int]) -> str:
        page_nums = [i + 1 for i in idxs]
        # Single page: just the image (unchanged behaviour). Multi page: precede each image with
        # its marker so the model can echo them back and keep the pages separable.
        content: list = []
        if len(idxs) == 1:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_pages[idxs[0]]}"}}
            )
        else:
            for i in idxs:
                content.append({"type": "text", "text": f"[== Halaman {i + 1} ==]"})
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_pages[i]}"}}
                )
        content.append({"type": "text", "text": prompt})

        retry_wait = 0.0
        for attempt in range(max_retries):
            if attempt > 0:
                await asyncio.sleep(retry_wait)
            async with semaphore:
                try:
                    response = await vision_llm.ainvoke([HumanMessage(content=content)])
                    text = response.content if isinstance(response.content, str) else ""
                    text = _strip_tabular_content(text)
                    return _renumber_markers(text, page_nums)
                except Exception as exc:
                    if _is_rate_limit(exc) and attempt < max_retries - 1:
                        retry_wait = _parse_retry_after(str(exc), fallback=(attempt + 1) * 10)
                        logger.warning(
                            "Rate limit on pages %s (attempt %d/%d), retrying in %.1fs",
                            page_nums, attempt + 1, max_retries, retry_wait,
                        )
                    else:
                        logger.exception("Vision extraction failed for pages %s", page_nums)
                        return _empty_markers(page_nums)
        logger.error("Vision extraction gave up on pages %s after %d retries", page_nums, max_retries)
        return _empty_markers(page_nums)

    batch_texts = await asyncio.gather(*[_extract_batch(idxs) for idxs in batches])
    combined = "\n\n".join(t for t in batch_texts if t.strip())
    logger.info("Async vision extraction returned %d chars total", len(combined))
    return combined


async def extract_narrative_text(file_bytes: bytes, vision_llm: Optional[BaseChatModel] = None) -> str:
    """Extract a PDF's narrative text, falling back to vision extraction when the text layer is too short.

    Single entry point for any caller that needs the full narrative text (with page markers) —
    call this once and share the result across multiple downstream consumers instead of extracting
    per-consumer, since the vision fallback is an LLM call with its own cost and rate limits.
    """
    text = extract_text_from_pdf(file_bytes)
    content_chars = len(PAGE_MARKER_RE.sub('', text).strip())
    too_short = content_chars < MIN_USEFUL_CHARS

    # A document can clear MIN_USEFUL_CHARS in aggregate while still being almost entirely
    # image/chart pages - only checked when the char count alone looks fine, so fake/short
    # bytes in the too-short branch never reach a second real PDF parse.
    mostly_blank_pages = False
    if not too_short:
        pages_with_text = len(PAGE_MARKER_RE.findall(text))
        total_pages = _count_total_pages(file_bytes)
        mostly_blank_pages = total_pages > 0 and pages_with_text / total_pages < 0.5

    if too_short or mostly_blank_pages:
        if vision_llm is None:
            logger.warning(
                "PDF content too short or mostly blank pages (%d chars excl. markers) and no "
                "vision_llm provided — pass vision_llm=get_vision_llm() to enable the fallback.",
                content_chars,
            )
        else:
            logger.info(
                "PDF content too short or mostly blank pages (%d chars excl. markers), "
                "falling back to vision extraction",
                content_chars,
            )
            text = await extract_text_from_pdf_vision_async(file_bytes, vision_llm)
    else:
        # Statistical-table rows still leak into the text layer; strip them so the typo checker
        # and structured extractor only see prose. The vision path already runs this per-page
        # (see extract_text_from_pdf_vision_async); the text-layer path needs it too.
        text = _strip_tabular_content(text)
    return text
