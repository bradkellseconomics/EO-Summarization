# Daily Scheduled EO Reviewer (Scraper)

This component scrapes the White House Presidential Actions pages, caches page text, and optionally summarizes each item with OpenAI. It appends new rows to a CSV, deduping by URL.

## Quick Start

- Install deps: `pip install -r requirements.txt`
- Optional: create `.env` and set `OPENAI_API_KEY=...` to enable summaries
- One-page test: `python eo_summarization_tool.py --single-page -v --no-summarize`

## CLI Usage (flexible)

- Basic: `python eo_summarization_tool.py --single-page`
- With summaries: `python eo_summarization_tool.py --summarize --single-page`
- Useful options:
  - `--output-dir` default `whitehouse_texts`
  - `--csv-file` default `whitehouse_presidential_actions.csv`
  - `--start-page` default `1`
  - `--max-pages` limit number of pages
  - `--delay` default `1.0` seconds
  - `--lookback-days` default `15` (set `0` to disable cutoff)
  - `--relevance-topic` optional label (e.g., "Immigration")

Notes:
- Topic-aware CSVs use the pattern: `whitehouse_presidential_actions - <Topic>.csv`.
- The tool stops scraping a page once it hits the first known URL but keeps earlier new items.

## Single-topic, Fixed CSV Runner

- Script: `run_scraper.py` (edit constants for topic, CSV, and lookback).
- Batch launcher: `run_scraper.bat` (safe to commit)
- Local override: `run_scraper.local.bat` (gitignored) for machine-specific paths.

Run:
- Double-click `run_scraper.bat` or run it from a terminal.

Interpreter resolution order in `run_scraper.bat`:
1) `%PYTHON_EXE%` env var if set
2) `.venv\Scripts\python.exe` if present
3) `py -3` if available
4) `python` on PATH

## Windows Task Scheduler

- Program/script: `run_scraper.bat`
- Start in: leave blank (the batch changes to its own folder)
- If Python isn’t on PATH, either:
  - Set a user/system env var `PYTHON_EXE` to your interpreter, or
  - Create a `.venv` and `pip install -r requirements.txt`

## Output

- Text cache: `whitehouse_texts/` (created automatically)
- CSV: `whitehouse_presidential_actions.csv` or topic-suffixed CSV
- Columns: `date, title, url, types, text_file, description, relevance_topic, relevant_eo, full_summary`

## Troubleshooting

- No rows added: increase scope (set `LOOKBACK_DAYS=None` or `--lookback-days 0`), set `MAX_PAGES=1` for a quick check, and enable `-v`.
- “Python was not found”: set `PYTHON_EXE` or create `.venv`.
- Summaries empty/missing: ensure `OPENAI_API_KEY` is set or run with `--no-summarize`.
- “Found 0 actions”: verify the site is reachable and selectors still match (temporary outages happen).

