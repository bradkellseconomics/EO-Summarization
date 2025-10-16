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

- The current prompt assesses whether an action concerns pesticide use (Yes/No/Maybe) and provides a short detailed summary when relevant.
- If you have an existing CSV created before this change, it may have a column named `immigration_relevant`. The tool will continue writing to that column for backward compatibility; new CSVs will use `pesticide_relevant`.
