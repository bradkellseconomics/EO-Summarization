EO Summarization Toolkit

This repo has two parts:
- Daily Scheduled EO Reviewer (CLI/scraper) → see docs/scraper.md
- Streamlit Dashboard (visualize and explore) → see docs/streamlit.md

EO Summarization CLI

This script scrapes recent items from the White House Presidential Actions pages, saves the page text to files, and optionally summarizes each item with an OpenAI model. Results are appended to a CSV.

Usage

- Basic scrape only (no summaries):
  `python eo_summarization_tool.py --single-page`

- Scrape + summarize (requires environment variable `OPENAI_API_KEY`):
  `python eo_summarization_tool.py --summarize --single-page`

Key options

- `--output-dir` default `whitehouse_texts`
- `--csv-file` default `whitehouse_presidential_actions.csv`
- `--start-page` default `1`
- `--max-pages` limit number of pages
- `--delay` default `1.0` seconds between page fetches
- `--single-page` scrape only one page
- `--summarize` enable OpenAI summarization (set `OPENAI_API_KEY`)
- `--model` OpenAI model (default `gpt-4o`)

Install

Create a virtualenv and install dependencies:

`pip install -r requirements.txt`

Notes

- The tool reads existing URLs from the CSV to avoid duplicates. It stops immediately when it reaches the first previously-scraped URL, after preserving any new items found earlier on that page.
- API keys are not stored in code; set `OPENAI_API_KEY` in your environment when using `--summarize`.

Summarization focus

- The prompt asks for a one‑sentence summary and whether the action is relevant to the analysis scope (Yes/No/Maybe). If Yes/Maybe, it includes a short detailed summary.


Scheduled Execution (Windows)

- Use `run_scraper.bat` as a portable launcher for daily scraping/summarization via Task Scheduler.
- To override interpreter paths or include machine-specific logic, create `run_scraper.local.bat` in the repo folder. It is gitignored and called by `run_scraper.bat` if present.
- `run_scraper.bat` behavior:
  - Runs from its own directory so relative paths resolve.
  - Honors `PYTHON_EXE` if set; otherwise prefers `.venv\\Scripts\\python.exe`, then `py -3`, then `python`.
  - Exits with the Python process code. The script prints a completion message and pauses when run interactively.

Task Scheduler quick setup

- Action: Start a program → Program/script: `run_scraper.bat`
- Start in: leave blank (the batch changes to its own folder)
- If Python is not on PATH, either set a user/system env var `PYTHON_EXE` to your interpreter or create a local `.venv` and install `requirements.txt`.
