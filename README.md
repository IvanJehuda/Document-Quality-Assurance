# Fact-Checker

[![CI](https://github.com/DStaDQAD/Document-Quality-Assurance/actions/workflows/ci.yml/badge.svg)](https://github.com/DStaDQAD/Document-Quality-Assurance/actions/workflows/ci.yml)

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
authoritative values in a companion Excel file. No SQL is generated, and the LLM never
touches a number:

1. **Excel parsing** - a three-tier cascade. The BI wide-table parser (`excel_parser_bi.py`,
   `openpyxl`/`xlrd` + cell-style heuristics for headers, data rows, unit label, and column
   periods) is tried first; when it fails or extracts nothing, a generic heuristic parser
   (`table_parser_generic.py`) takes over, handling any table with a single detectable header
   row: time-series tables with combined period headers ("Apr 2026", "2026M04", real date
   cells) **and** non-time-series tables such as item lists or budgets
   ("Nama Barang | Harga | Stok"). Tiers 1-2 are fully deterministic. When both fail, tier 3
   (`table_parser_llm.py`) shows the LLM a coordinate snapshot of the grid and asks for a
   structure **spec** (header row, label column, each column's period/attribute meaning) - the
   spec is validated against the grid and all values are then extracted by code, so the LLM
   still never touches a number. Validated specs are cached per layout fingerprint, so repeat
   uploads of the same layout parse without another LLM call, and the UI flags LLM-mapped
   sources for review. All parsers emit the same two-axis `TableData` container
   (`table_model.py`): rows × (periods **or** attribute columns) → value.
2. **PDF text extraction** (`pdf_extraction.py`) - `pypdf` first, falling back to Gemini vision
   OCR if the extracted text is too sparse. The extracted narrative text is shared between fact
   verification and the typo/grammar check below, so the vision fallback only ever runs once.
3. **Structured fact extraction** (`structured_extractor.py`) - the *only* LLM step in fact
   verification. Claims are represented as an **operation** (`value`, `yoy_growth`, `average`,
   `sum`, `diff`, `ratio`, `is_increasing`, `is_decreasing`, `is_stable`) over one or more
   `(metric_label, year, month, value_raw, unit)` data points, anchored to the Excel's own row
   labels. Against a non-time-series source, a data point carries a `col_label` (the attribute
   column, e.g. "Harga") instead of year/month; time-only operations (`yoy_growth`, trends) are
   gated to temporal sources. Claims more complex than a single point-in-time value (multi-period averages,
   cross-metric ratios, trend statements) are supported without a schema change per pattern. The
   model may only copy `value_raw` verbatim from the source text - all numeric parsing happens
   afterward in Python (`_parse_indonesian_number`), so a hallucinated or misformatted number can
   never reach the comparison step.
4. **Comparison and verdict** (`paired_verifier.py`) - fully deterministic. Every data point is a
   direct dict lookup into the parsed Excel tables (fuzzy label matching, unit conversion); the
   operation itself (average, sum, diff, ratio, monotonic-trend check, YoY growth) is computed in
   plain Python, never by an LLM, to avoid hallucination risk in the comparison step.
5. **Typo/grammar check** (`typo_checker.py`) - runs on the same extracted narrative text,
   independent of fact verification. A free, deterministic dictionary/rule pass (`phunspell`
   id_ID + a curated tidak-baku word list + reduplication detection) handles clear-cut cases;
   only genuinely ambiguous words are escalated to a single batched LLM call, deduplicated by
   unique word so the call is bounded by vocabulary size rather than document length. Returned
   as `typo_check` alongside the fact-verification results.

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

A small web UI is served at `http://127.0.0.1:8000/` with two tabs: **Verifikasi Berpasangan**
(Mode 2 - upload a PDF + Excel, see fact-check and typo/grammar results) and **Lihat Data** (a
raw browser over every table in the SQLite database, including any `.xlsx` sources ingested via
`/api/upload-excel-source`). Mode 1's single-claim/document verification has no UI form - it is
reachable only via the API endpoints below. Interactive API docs: `http://127.0.0.1:8000/docs`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/verify-claim` | POST | Verify a single claim. |
| `/api/verify-document` | POST | Extract every checkable claim from a block of text, verify each, return a per-claim report plus an aggregate summary. |
| `/api/verify-document-pdf` | POST | Same as above, but the document is an uploaded digital PDF (falls back to Gemini vision OCR if extracted text is too sparse). |
| `/api/upload-excel-source` | POST | Upload a `.xlsx` file to ingest as a new data source into `excel_facts`. Trusted-uploader endpoint, no auth - meant for a developer/small team, not arbitrary public users. |
| `/api/verify-paired` | POST | Mode 2. Upload a PDF report plus its companion BI-format Excel file; verifies every quantitative claim in the PDF narrative against the Excel's authoritative values, plus an Indonesian typo/grammar check on the same narrative text. |
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

```json
{
  "pdf_filename": "laporan.pdf",
  "excel_filenames": ["TABEL1_1.xls"],
  "excel_sheets": ["I.1"],
  "excel_units": ["triliun Rp"],
  "total_facts": 2,
  "entailed_count": 1,
  "refuted_count": 1,
  "inconclusive_count": 0,
  "results": [
    {
      "operation": "value",
      "metric_label": "Uang Kartal",
      "matched_excel_source": "TABEL1_1.xls / I.1",
      "periods": [{"metric_label": "Uang Kartal", "year": 2024, "month": "Maret", "excel_value": 123.4}],
      "claimed_value": 123.4, "claimed_unit": "triliun Rp",
      "computed_value": 123.4, "computed_unit": "triliun Rp",
      "delta": 0.0, "verdict": "Entailed",
      "reasoning": "...", "context_quote": "...", "page_number": 1
    },
    {
      "operation": "yoy_growth",
      "metric_label": "M2",
      "matched_excel_source": "TABEL1_1.xls / I.1",
      "periods": [
        {"metric_label": "M2", "year": 2024, "month": "Maret", "excel_value": 8900.0},
        {"metric_label": "M2", "year": 2023, "month": "Maret", "excel_value": 8500.0}
      ],
      "claimed_value": 6.0, "claimed_unit": "persen_yoy",
      "computed_value": 4.7, "computed_unit": "persen_yoy",
      "delta": 1.3, "verdict": "Refuted",
      "reasoning": "...", "context_quote": "...", "page_number": 2
    }
  ],
  "typo_check": {
    "total_issues": 1,
    "ejaan_count": 1, "tidak_baku_count": 0, "grammar_count": 0,
    "summary": "1 issue found: 1 spelling.",
    "issues": [
      {"word": "inflsi", "start": 120, "end": 126, "category": "ejaan",
       "suggestion": "inflasi", "explanation": "...", "page_number": 1}
    ]
  }
}
```

## Tests

```bash
pytest
```

All LLM calls are mocked, so the suite needs no real API keys or network access.

Accuracy (not just plumbing) is checked separately by the eval harness in `eval/` - see
`eval/README.md`. CI runs the deterministic Layer-1 comparison-engine eval
(`python -m eval.run_comparison_eval --fail-under 1.0`) on every push to `main`, gating on
100% verdict accuracy against the fixed cases in `eval/cases/`.
