# Fact-Checker

Verifies natural-language claims about statistical data. The project has two independent
verification pipelines that share one FastAPI app.

## Mode 1: Text-to-SQL + LLM-as-a-Judge

Verifies claims against a SQLite database, using an LLM to translate each claim into SQL (rather
than a fixed query template) and a second LLM call to judge the claim against whatever the query
actually returns.

Two kinds of ground truth feed the same database:
- **Clean relational tables** (`indikator_ekonomi`, `neraca_dagang`) - seeded dummy macroeconomic
  data with a normal SQL schema.
- **Ingested Excel sources** (`excel_facts`) - real-world statistical tables are rarely a clean
  schema: headers can start at any row/column, column order varies file to file, and totals are
  embedded as aggregate cells. `excel_ingestion.py` reads `.xlsx` files with `openpyxl` (so cell
  *style* - bold, fill, borders, merges - is available as a structural signal, not just values),
  detects header/data/aggregate regions with style+keyword+position heuristics, and normalizes
  every sheet into one long/tidy table (`source_file, sheet, row_label, col_label, value,
  is_aggregate, source_cell_ref`) regardless of the original layout.

### How it works

1. **Claim extraction** (document/PDF endpoints only) - one LLM call pulls atomic, checkable
   claims out of free-form text.
2. **Text-to-SQL** - one batched LLM call writes a SQL `SELECT` per claim against the live DB
   schema. To avoid the model guessing a plausible-but-wrong free-text value (e.g. the wrong
   `source_file` for a given `row_label`), the prompt is also given every small-cardinality text
   column's distinct values *and* which values of different columns actually co-occur together.
   Claims that compare multiple periods/categories are detected and retried if the first query
   doesn't return enough data to support the comparison.
3. **LLM-as-a-Judge** - a second batched LLM call compares each claim against its retrieved data
   and returns `Entailed`, `Refuted`, or `Inconclusive` with reasoning.

Every SQL query runs through a read-only SQLite connection (`mode=ro`), so the verification path
can never write to the database no matter what the model generates.

## Mode 2: Paired PDF + Excel verification

Verifies every quantitative claim in a PDF report's narrative text directly against the
authoritative values in a companion BI-format Excel file. No SQL is generated, and the LLM never
touches a number:

1. **Excel parsing** (`excel_parser_bi.py`) - fully deterministic. Uses `openpyxl` + cell style
   heuristics to locate headers, data rows, unit label, and column periods.
2. **PDF text extraction** (`pdf_extraction.py`) - `pypdf` first, falling back to Gemini vision
   OCR if the extracted text is too sparse.
3. **Structured fact extraction** (`structured_extractor.py`) - the *only* LLM step. The model
   extracts `(metric_label, year, month, value_raw, unit, claim_type, context_quote)` tuples from
   the narrative, anchored to the Excel's own row labels. It may only copy `value_raw` verbatim
   from the source text - all numeric parsing happens afterward in Python
   (`_parse_indonesian_number`), so a hallucinated or misformatted number can never reach the
   comparison step.
4. **Comparison and verdict** (`paired_verifier.py`) - fully deterministic fuzzy label lookup,
   unit conversion, and delta/YoY checks against the Excel. No LLM judges the result.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and fill in the provider you want to use:
   - `LLM_PROVIDER=ollama` - runs fully locally, no API key (requires Ollama running with the
     configured model pulled).
   - `LLM_PROVIDER=groq` or `LLM_PROVIDER=google` - set the matching `*_API_KEY`.
2. Seed the dummy SQL tables:
   ```bash
   python setup_db.py
   ```
3. (Optional) Generate and ingest the sample stylized Excel fixtures, to populate `excel_facts`:
   ```bash
   python create_sample_excel.py
   python excel_ingestion.py sample_data/penjualan_2024.xlsx sample_data/penjualan_cabang.xlsx
   ```
4. Start the API:
   ```bash
   uvicorn main:app --reload
   ```

A small web UI is served at `http://127.0.0.1:8000/` (claim/document verification, Excel upload,
and a raw table browser). Interactive API docs: `http://127.0.0.1:8000/docs`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/verify-claim` | POST | Verify a single claim. |
| `/api/verify-document` | POST | Extract every checkable claim from a block of text, verify each, return a per-claim report plus an aggregate summary. |
| `/api/verify-document-pdf` | POST | Same as above, but the document is an uploaded digital PDF (falls back to Gemini vision OCR if extracted text is too sparse). |
| `/api/upload-excel-source` | POST | Upload a `.xlsx` file to ingest as a new data source into `excel_facts`. Trusted-uploader endpoint, no auth - meant for a developer/small team, not arbitrary public users. |
| `/api/verify-paired` | POST | Mode 2. Upload a PDF report plus its companion BI-format Excel file; verifies every quantitative claim in the PDF narrative against the Excel's authoritative values. |
| `/api/tables` | GET | List every table in the database. |
| `/api/tables/{table_name}` | GET | Paginated raw rows for one table (`?limit=&offset=`). |
| `/health` | GET | Liveness check. |

### Example: verify a single claim

```bash
curl -X POST http://127.0.0.1:8000/api/verify-claim \
  -H "Content-Type: application/json" \
  -d "{\"claim\": \"Inflasi pada Q1 2023 lebih tinggi dibandingkan Q4 2023\"}"
```

```json
{
  "status": "Entailed",
  "sql_query_used": "SELECT ...",
  "reasoning": "..."
}
```

### Example: verify a whole document

```bash
curl -X POST http://127.0.0.1:8000/api/verify-document \
  -H "Content-Type: application/json" \
  -d "{\"document\": \"Inflasi pada Q1 2023 lebih tinggi dibandingkan Q4 2023. Suku bunga acuan naik sepanjang 2023 dan 2024.\"}"
```

```json
{
  "total_claims": 2,
  "entailed_count": 1,
  "refuted_count": 1,
  "inconclusive_count": 0,
  "error_count": 0,
  "summary": "2 claim(s) extracted: 1 entailed, 1 refuted, 0 inconclusive, 0 could not be verified.",
  "results": [
    {"claim": "...", "status": "Entailed", "sql_query_used": "SELECT ...", "reasoning": "..."},
    {"claim": "...", "status": "Refuted", "sql_query_used": "SELECT ...", "reasoning": "..."}
  ]
}
```

### Example: verify a PDF document

```bash
curl -X POST http://127.0.0.1:8000/api/verify-document-pdf \
  -F "file=@laporan.pdf;type=application/pdf"
```

Response shape is identical to `/api/verify-document`.

### Example: verify a paired PDF + Excel report

```bash
curl -X POST http://127.0.0.1:8000/api/verify-paired \
  -F "pdf_file=@laporan.pdf;type=application/pdf" \
  -F "excel_file=@TABEL1_1.xls" \
  -F "sheet_names=I.1"
```

`excel_file` accepts multiple files (repeat the `-F "excel_file=@..."` flag); `sheet_names` is a
comma-separated list, one sheet name per file, and is reused for any remaining files if shorter.

## Tests

```bash
pytest
```

All LLM calls are mocked, so the suite needs no real API keys or network access.
