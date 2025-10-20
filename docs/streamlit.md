# Streamlit Dashboard

Interactive dashboard for exploring scraped Presidential actions and GPT summaries.

## Quick Start

- Install deps: `pip install -r requirements.txt`
- Ensure you have at least one CSV (run the scraper first)
- Launch: `streamlit run streamlit_app.py`

## Features

- Sidebar filters: date range, relevance (Yes/Maybe/No), types, free-text search
- KPIs and charts: relevance breakdown and actions over time
- Results table with details expanders and a filtered CSV download
- “Run Tools” sidebar section:
  - Choose a base CSV and optional topic; shows the derived output CSV path
  - Configure start page, lookback days (0 = no cutoff), max pages, delay, and model
  - Trigger a scrape/update directly from the UI (writes to the derived CSV)

## CSV Discovery

The app scans for `whitehouse_presidential_actions.csv` and topic-suffixed files like `whitehouse_presidential_actions - <Topic>.csv`. It lists them by detected topic or filename.

## Theme

Customize Streamlit theme via `.streamlit/config.toml` in the repo folder. Example:

```
[theme]
base = "dark"   # or "light"
primaryColor = "#2da44e"
```

This file is safe to commit; omit secrets.

## Notes

- Summarization uses the `OPENAI_API_KEY` if set (via `.env` or environment). The UI can also run without summaries.
- The dashboard reads from CSVs produced by the scraper; it doesn’t require network access to display data.

