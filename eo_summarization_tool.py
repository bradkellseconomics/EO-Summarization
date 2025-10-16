import argparse
import csv
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.whitehouse.gov"
ACTIONS_URL = "https://www.whitehouse.gov/presidential-actions/page/"


# ------------------------------ Models ------------------------------


@dataclass
class Action:
    date: str
    title: str
    url: str
    types: List[str]
    text_file: str = ""
    description: str = ""
    pesticide_relevant: str = ""
    full_summary: str = ""


# ---------------------------- Utilities -----------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ua_headers() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; EO-Summarizer/1.0)"}


def clean_text_for_gpt(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_filename(name: str, max_len: int = 120) -> str:
    name = name[:max_len]
    # replace path separators and bad chars
    return re.sub(r"[\\/:*?\"<>|]", "_", name)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------- Scraping -----------------------------


def fetch_actions_page(page_num: int) -> List[Action]:
    url = f"{ACTIONS_URL}{page_num}/"
    logging.info("Scraping page %s -> %s", page_num, url)
    try:
        resp = requests.get(url, headers=ua_headers(), timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.warning("Error fetching page %s: %s", page_num, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select("li.wp-block-post")
    logging.info("Found %d actions", len(items))

    results: List[Action] = []
    for item in items:
        title_tag = item.select_one("h2.wp-block-post-title a")
        date_tag = item.select_one(".wp-block-post-date time")
        type_tags = item.select(".taxonomy-category a")
        if not title_tag or not date_tag:
            continue

        title = title_tag.text.strip()
        date_str = date_tag.text.strip()
        url = title_tag.get("href", "").strip()
        types = [t.text.strip() for t in type_tags] if type_tags else ["no types"]
        results.append(Action(date=date_str, title=title, url=url, types=types))

    return results


def fetch_action_content(url: str) -> str:
    try:
        resp = requests.get(url, headers=ua_headers(), timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.select("main p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return clean_text_for_gpt(content)
    except Exception as e:
        logging.warning("Could not fetch content for %s: %s", url, e)
        return "[no content]"


# ------------------------------ Storage ------------------------------


def parse_date_maybe(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%B %d, %Y", "%d-%b-%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_known_urls(csv_file: str) -> Tuple[set, Optional[datetime]]:
    if not os.path.exists(csv_file):
        return set(), None

    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        logging.warning("Could not read CSV %s: %s", csv_file, e)
        return set(), None

    known = {r.get("url", "") for r in rows if r.get("url")}
    dates = [parse_date_maybe(r.get("date", "")) for r in rows]
    dates = [d for d in dates if d]
    latest = max(dates) if dates else None
    return known, latest


def save_text(output_dir: str, date_str: str, title: str, content: str) -> str:
    ensure_dir(output_dir)
    filename = f"{date_str} - {safe_filename(title, 60)}.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def append_actions(csv_file: str, actions: List[Action]) -> None:
    ensure_dir(os.path.dirname(csv_file) or ".")

    fieldnames = [
        "date",
        "title",
        "url",
        "types",
        "text_file",
        "description",
        "pesticide_relevant",
        "full_summary",
    ]
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for a in actions:
            row = asdict(a)
            row["types"] = "; ".join(a.types)
            writer.writerow(row)


# ----------------------------- Summarizer ----------------------------


def build_openai_client_from_env():
    # Load .env if available
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    # Lazy import to keep the tool usable without openai installed if summarization is off
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key)


def summarize_action_text(client, text: str, model: str = "gpt-4o", temperature: float = 0.2) -> Tuple[str, str, str]:
    prompt = (
        "You are a policy analyst focusing on pesticide use and regulation. Review the following executive action and answer:\n\n"
        "1. Write a one-sentence summary of what this executive action does.\n"
        "2. Is this executive action about pesticide use? Answer Yes, No, or Maybe.\n"
        "3. If Yes or Maybe, provide a 3-5 sentence summary with additional detail.\n\n"
        "Executive Action:\n"
    )

    messages = [
        {"role": "system", "content": "You are a careful and concise policy analyst."},
        {"role": "user", "content": prompt + text},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    gpt_text = (resp.choices[0].message.content or "").strip()
    lines = [l.strip() for l in gpt_text.split("\n") if l.strip()]

    description = lines[0] if lines else ""
    pesticide_relevant = "No"
    full_summary = ""
    for idx, line in enumerate(lines):
        if any(tok in line for tok in ("Yes", "Maybe")):
            pesticide_relevant = line
            full_summary = " ".join(lines[idx + 1:]).strip()
            break

    return description, pesticide_relevant, full_summary


# ------------------------------ Orchestration ------------------------------


def process_page(
    page_num: int,
    known_urls: set,
    output_dir: str,
    client=None,
    model: str = "gpt-4o",
) -> Tuple[List[Action], bool]:
    """Process a listing page.

    Returns (results, known_encountered). Stops at the first known URL,
    but preserves any new items found earlier on the page.
    """
    actions = fetch_actions_page(page_num)
    results: List[Action] = []
    known_encountered = False

    for a in actions:
        if a.url in known_urls:
            known_encountered = True
            break

        content = fetch_action_content(a.url)
        a.text_file = save_text(output_dir, a.date, a.title, content)

        if client is not None and content and content != "[no content]":
            try:
                desc, pest_rel, full = summarize_action_text(client, content, model=model)
                a.description = desc
                a.pesticide_relevant = pest_rel
                a.full_summary = full
            except Exception as e:
                logging.warning("Summarization failed for %s: %s", a.url, e)

        results.append(a)

    return results, known_encountered


def run(
    output_dir: str,
    csv_file: str,
    start_page: int,
    max_pages: Optional[int],
    delay: float,
    test_single_page: bool,
    summarize: bool,
    model: str,
):
    known_urls, latest = load_known_urls(csv_file)
    if latest:
        logging.info("Latest date in CSV: %s", latest.strftime("%Y-%m-%d"))
    if known_urls:
        logging.info("Known URLs: %d", len(known_urls))

    client = None
    if summarize:
        client = build_openai_client_from_env()

    all_actions: List[Action] = []
    page = start_page
    pages_processed = 0
    # Stop immediately once a known URL is encountered
    
    while True:
        page_results, known_hit = process_page(page, known_urls, output_dir, client=client, model=model)
        if page_results:
            all_actions.extend(page_results)
        # If we encountered a known URL on this page, we're caught up — stop now.
        if known_hit:
            break
        # If the page produced no results (empty listing or errors), stop.
        if not page_results:
            break

        page += 1
        pages_processed += 1
        if test_single_page:
            break
        if max_pages is not None and pages_processed >= max_pages:
            break
        time.sleep(delay)

    logging.info("Total new actions scraped: %d", len(all_actions))
    if all_actions:
        append_actions(csv_file, all_actions)


# ------------------------------- CLI --------------------------------


def main():
    parser = argparse.ArgumentParser(description="EO Summarization Tool")
    parser.add_argument("--output-dir", default="whitehouse_texts", help="Directory to save text files")
    parser.add_argument("--csv-file", default="whitehouse_presidential_actions.csv", help="CSV output path")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number")
    parser.add_argument("--max-pages", type=int, default=None, help="Max number of pages to scrape (None=until stop)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between page requests (seconds)")
    parser.add_argument("--single-page", action="store_true", help="Scrape only a single page and exit")
    parser.add_argument("--summarize", action="store_true", help="Enable OpenAI summarization (requires OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for summarization")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    run(
        output_dir=args.output_dir,
        csv_file=args.csv_file,
        start_page=args.start_page,
        max_pages=args.max_pages,
        delay=args.delay,
        test_single_page=args.single_page,
        summarize=args.summarize,
        model=args.model,
    )


if __name__ == "__main__":
    main()
