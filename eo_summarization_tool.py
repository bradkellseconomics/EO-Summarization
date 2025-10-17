import argparse
import csv
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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
    relevance_topic: str = ""
    relevant_eo: str = ""
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


def topic_csv_path(base_csv: str, relevance_topic: Optional[str]) -> str:
    """Derive a topic-specific CSV path by inserting the topic before the extension.

    Examples:
      base_csv = "whitehouse_presidential_actions.csv", topic "Immigration" ->
      "whitehouse_presidential_actions - Immigration.csv"
    """
    if not relevance_topic:
        return base_csv
    directory = os.path.dirname(base_csv)
    base = os.path.basename(base_csv)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".csv"
    topic_part = safe_filename(relevance_topic.strip(), 80)
    filename = f"{root} - {topic_part}{ext}"
    return os.path.join(directory or ".", filename)


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
        "relevance_topic",
        "relevant_eo",
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


def summarize_action_text(
    client,
    text: str,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    relevance_topic: Optional[str] = None,
) -> Tuple[str, str, str]:
    scope_line = ""
    if relevance_topic:
        topic = relevance_topic.strip()
        scope_line = (
            f"Relevance scope: {topic}. Mark 'Relevant' only if the action materially concerns {topic} (not just tangential mentions).\n"
        )
    prompt = (
        "You are a policy analyst reviewing U.S. Presidential actions. Read the text and answer.\n"
        + scope_line +
        "\nRespond in exactly three labeled lines, without numbering, like:\n"
        "Summary: <one sentence>\n"
        "Relevant: <Yes|No|Maybe>\n"
        "Details: <3-5 sentences if Relevant is Yes or Maybe; otherwise leave blank>\n\n"
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

    def _strip_prefix(s: str) -> str:
        return re.sub(r"^\s*(?:\d+[\.)]\s*|[-*]\s*)?(?:Summary|Relevant|Relevance|Details?):?\s*", "", s, flags=re.IGNORECASE).strip()

    def _extract_relevance_from_labeled(s: str) -> Optional[str]:
        m = re.search(r"\b(Yes|No|Maybe)\b", s, flags=re.IGNORECASE)
        return m.group(1).capitalize() if m else None

    lines = [l.strip() for l in gpt_text.split("\n") if l.strip()]

    desc, rel, details = "", "No", ""
    labeled_found = False
    for idx, line in enumerate(lines):
        if re.match(r"(?i)^\s*summary\s*:\s*", line):
            desc = _strip_prefix(line)
            labeled_found = True
        elif re.match(r"(?i)^\s*(?:relevant|relevance)\s*:\s*", line):
            r = _extract_relevance_from_labeled(line)
            if r:
                rel = r
            labeled_found = True
        elif re.match(r"(?i)^\s*details?\s*:\s*", line):
            details = _strip_prefix(line)
            if idx + 1 < len(lines):
                tail = " ".join(_strip_prefix(l) for l in lines[idx + 1:]).strip()
                if tail:
                    details = (details + " " + tail).strip() if details else tail
            labeled_found = True

    if not labeled_found:
        # Fallback: summary from first line; relevance from a clean answer line
        desc = _strip_prefix(lines[0]) if lines else ""
        rel_line_idx = None
        answer_pattern = re.compile(r"^\s*(?:\d+[\.)]\s*)?(?:[-*]\s*)?(?:relevant|relevance)?[:\-\s]*\s*(Yes|No|Maybe)\b", re.IGNORECASE)
        for idx, line in enumerate(lines):
            m = answer_pattern.match(line)
            if m:
                rel = m.group(1).capitalize()
                rel_line_idx = idx
                break
        if rel_line_idx is not None and rel in ("Yes", "Maybe"):
            details = " ".join(_strip_prefix(l) for l in lines[rel_line_idx + 1:]).strip()

    if rel not in ("Yes", "Maybe"):
        details = ""

    return desc, rel, details


def summarize_with_fallback(client, text: str, model: str, relevance_topic: Optional[str]) -> Tuple[str, str, str]:
    """Try summarization with the requested model, then fall back to gpt-4o-mini."""
    try:
        return summarize_action_text(client, text, model=model, relevance_topic=relevance_topic)
    except Exception as e:
        logging.warning("Primary model '%s' failed: %s. Falling back to 'gpt-4o-mini'", model, e)
        return summarize_action_text(client, text, model="gpt-4o-mini", relevance_topic=relevance_topic)


# ------------------------------ Orchestration ------------------------------


def process_page(
    page_num: int,
    known_urls: set,
    output_dir: str,
    client=None,
    model: str = "gpt-4o",
    cutoff_date: Optional[datetime] = None,
    relevance_topic: Optional[str] = None,
) -> Tuple[List[Action], bool, bool]:
    """Process a listing page.

    Returns (results, known_encountered). Stops at the first known URL,
    but preserves any new items found earlier on the page.
    """
    actions = fetch_actions_page(page_num)
    results: List[Action] = []
    known_encountered = False
    cutoff_encountered = False

    for a in actions:
        if a.url in known_urls:
            known_encountered = True
            break

        # Respect lookback window: stop when encountering an action older than cutoff
        if cutoff_date is not None:
            a_date = parse_date_maybe(a.date)
            if a_date is not None and a_date < cutoff_date:
                cutoff_encountered = True
                break

        content = fetch_action_content(a.url)
        a.text_file = save_text(output_dir, a.date, a.title, content)

        if client is not None and content and content != "[no content]":
            try:
                logging.debug("Summarizing URL=%s model=%s content_len=%d", a.url, model, len(content))
                _t0 = time.perf_counter()
                desc, pest_rel, full = summarize_with_fallback(client, content, model=model, relevance_topic=relevance_topic)
                _dt = time.perf_counter() - _t0
                a.description = desc
                a.relevance_topic = relevance_topic or ""
                a.relevant_eo = pest_rel
                a.full_summary = full
                logging.info("Summarized: '%s' [relevance=%s, %.2fs]", a.title, pest_rel or "", _dt)
                if desc:
                    logging.debug("Summary preview: %s", desc[:140])
            except Exception as e:
                logging.warning("Summarization failed for %s: %s", a.url, e)

        results.append(a)

    return results, known_encountered, cutoff_encountered


def run(
    output_dir: str,
    csv_file: str,
    start_page: int,
    max_pages: Optional[int],
    delay: float,
    test_single_page: bool,
    summarize: bool,
    model: str,
    lookback_days: Optional[int],
    update_missing: bool = False,
    relevance_topic: Optional[str] = None,
):
    known_urls, latest = load_known_urls(csv_file)
    if latest:
        logging.info("Latest date in CSV: %s", latest.strftime("%Y-%m-%d"))
    if known_urls:
        logging.info("Known URLs: %d", len(known_urls))

    client = None
    if summarize or update_missing:
        try:
            client = build_openai_client_from_env()
            logging.info("OpenAI client initialized; summarization enabled (summarize=%s, update_missing=%s)", summarize, update_missing)
        except Exception as e:
            logging.warning("Summarization disabled: %s", e)
            client = None
            summarize = False
            update_missing = False

    all_actions: List[Action] = []
    page = start_page
    pages_processed = 0
    # Determine cutoff date if a lookback window is set
    cutoff_date: Optional[datetime] = None
    if lookback_days is not None and lookback_days > 0:
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        logging.info("Applying lookback window: last %d days (cutoff %s)", lookback_days, cutoff_date.strftime("%Y-%m-%d"))

    while True:
        page_results, known_hit, cutoff_hit = process_page(
            page, known_urls, output_dir, client=client, model=model, cutoff_date=cutoff_date, relevance_topic=relevance_topic
        )
        if page_results:
            all_actions.extend(page_results)
        # If we encountered a known URL on this page, we're caught up — stop now.
        if known_hit:
            break
        if cutoff_hit:
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

    # Optionally backfill missing summaries in existing CSV rows
    if update_missing:
        updated = update_missing_summaries(csv_file, output_dir, client, model=model, delay=delay)
        logging.info("Updated missing summaries: %d rows", updated)


def update_missing_summaries(
    csv_file: str,
    output_dir: str,
    client,
    model: str = "gpt-4o",
    delay: float = 0.0,
    relevance_topic: Optional[str] = None,
) -> int:
    """Scan CSV for rows missing summaries and fill them in.

    Returns the count of rows updated.
    """
    if not os.path.exists(csv_file):
        logging.info("CSV file not found: %s", csv_file)
        return 0

    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            rows: List[Dict[str, str]] = list(csv.DictReader(f))
    except Exception as e:
        logging.warning("Could not read CSV %s: %s", csv_file, e)
        return 0

    if not rows:
        return 0

    fieldnames = [
        "date",
        "title",
        "url",
        "types",
        "text_file",
        "description",
        "relevance_topic",
        "relevant_eo",
        "full_summary",
    ]

    updated_count = 0
    for r in rows:
        desc = (r.get("description") or "").strip()
        pest = (r.get("relevant_eo") or "").strip()
        full = (r.get("full_summary") or "").strip()
        if desc and pest:
            continue  # already summarized

        # Obtain content from saved text file or by fetching
        content: str = ""
        text_path = (r.get("text_file") or "").strip()
        if text_path and os.path.exists(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as tf:
                    content = tf.read()
            except Exception as e:
                logging.debug("Failed reading text file %s: %s", text_path, e)

        if not content:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            content = fetch_action_content(url)
            # Cache content to a file for future runs
            date_str = r.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
            title = r.get("title") or "untitled"
            text_path = save_text(output_dir, date_str, title, content)
            r["text_file"] = text_path

        try:
            logging.debug("Backfill summarize URL=%s model=%s content_len=%d", r.get("url"), model, len(content))
            _t0 = time.perf_counter()
            desc, pest_rel, full = summarize_with_fallback(client, content, model=model, relevance_topic=relevance_topic)
            _dt = time.perf_counter() - _t0
            r["description"] = desc
            r["relevance_topic"] = relevance_topic or r.get("relevance_topic") or ""
            r["relevant_eo"] = pest_rel
            r["full_summary"] = full
            updated_count += 1
            logging.info("Backfilled summary for title='%s' [relevance=%s, %.2fs]", (r.get("title") or ""), pest_rel or "", _dt)
            if desc:
                logging.debug("Backfill summary preview: %s", desc[:140])
            if delay:
                time.sleep(delay)
        except Exception as e:
            logging.warning("Summarization failed during update for URL %s: %s", r.get("url"), e)

    # Rewrite CSV with updated rows
    tmp_path = csv_file + ".tmp"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        os.replace(tmp_path, csv_file)
    except Exception as e:
        logging.warning("Failed writing updated CSV %s: %s", csv_file, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return updated_count


# ------------------------------- CLI --------------------------------


def main():
    parser = argparse.ArgumentParser(description="EO Summarization Tool")
    parser.add_argument("--output-dir", default="whitehouse_texts", help="Directory to save text files")
    parser.add_argument("--csv-file", default="whitehouse_presidential_actions.csv", help="CSV output path")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number")
    parser.add_argument("--max-pages", type=int, default=None, help="Max number of pages to scrape (None=until stop)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between page requests (seconds)")
    parser.add_argument("--single-page", action="store_true", help="Scrape only a single page and exit")
    parser.add_argument("--summarize", action="store_true", help="Enable OpenAI summarization (default enabled if API key present)")
    parser.add_argument("--no-summarize", action="store_true", help="Disable OpenAI summarization")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for summarization")
    parser.add_argument("--lookback-days", type=int, default=15, help="Only include actions from the last N days (default: 60)")
    parser.add_argument("--update-missing-summaries", action="store_true", help="Scan CSV and summarize rows missing summaries")
    parser.add_argument("--relevance-topic", default="", help="Topic to evaluate relevance against (e.g., 'Immigration', 'Pesticide', 'Education')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    # Summarization is enabled by default unless explicitly disabled
    summarize_flag = not args.no_summarize
    # If a relevance topic is provided, write to a topic-specific CSV
    topic_aware_csv = topic_csv_path(args.csv_file, args.relevance_topic or None)
    if args.relevance_topic:
        logging.info("Using topic-specific CSV: %s", topic_aware_csv)

    run(
        output_dir=args.output_dir,
        csv_file=topic_aware_csv,
        start_page=args.start_page,
        max_pages=args.max_pages,
        delay=args.delay,
        test_single_page=args.single_page,
        summarize=summarize_flag,
        model=args.model,
        lookback_days=args.lookback_days,
        update_missing=args.update_missing_summaries,
        relevance_topic=(args.relevance_topic or None),
    )


if __name__ == "__main__":
    main()
