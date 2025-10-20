"""
Single-topic CSV updater for White House Presidential Actions.

This script is a simplified wrapper around eo_summarization_tool.run().
It fixes a single topic and always writes to the same CSV file so it can be
used reliably in scheduled tasks.

To change the topic or output CSV path, edit the constants below.
"""

from eo_summarization_tool import run, setup_logging


# ------------------- Configuration (edit these) -------------------
# Pre-determined topic to evaluate relevance against.
TOPIC = "environmental impacts"

# Always update this CSV file (fixed path for scheduling).
# Note: This file will be created if it does not exist.
CSV_FILE = "whitehouse_presidential_actions - Environmental impacts.csv"

# Directory to store fetched text content for each action.
OUTPUT_DIR = "whitehouse_texts"

# Summarization model and behavior.
MODEL = "gpt-4o"
SUMMARIZE = True  # Set to False to skip OpenAI summarization

# Scrape behavior.
START_PAGE = 1
MAX_PAGES = None  # None to continue until known/cutoff; set an int to limit
DELAY_SECONDS = 1.0

# Only include actions from the last N days. Set to None to disable time cutoff.
LOOKBACK_DAYS = 15

# Whether to scan the CSV after scraping and backfill missing summaries.
UPDATE_MISSING = False

# Verbose logging (set True for more detail)
VERBOSE = False


def main() -> None:
    setup_logging(VERBOSE)

    run(
        output_dir=OUTPUT_DIR,
        csv_file=CSV_FILE,  # fixed file path; does not change with topic
        start_page=START_PAGE,
        max_pages=MAX_PAGES,
        delay=DELAY_SECONDS,
        test_single_page=False,
        summarize=SUMMARIZE,
        model=MODEL,
        lookback_days=LOOKBACK_DAYS,
        update_missing=UPDATE_MISSING,
        relevance_topic=TOPIC,
    )


if __name__ == "__main__":
    main()

