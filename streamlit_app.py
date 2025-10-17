import os
import csv
import re
from collections import Counter
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd
import eo_summarization_tool as tool


DEFAULT_CSV = "whitehouse_presidential_actions.csv"


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        st.error(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    # Normalize columns that might be missing
    for col in [
        "date",
        "title",
        "url",
        "types",
        "text_file",
        "description",
        "relevance_topic",
        "full_summary",
    ]:
        if col not in df.columns:
            df[col] = ""

    # Parse dates to datetime; keep original for display
    def parse_dt(s: str):
        for fmt in ("%B %d, %Y", "%Y-%m-%d", "%d-%b-%y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        return pd.to_datetime("NaT")

    df["date_dt"] = df["date"].astype(str).apply(parse_dt)
    # Clean categorical fields (standardized to 'relevance')
    df["types"] = df["types"].fillna("").astype(str)
    # Build unified 'relevance' directly from 'relevant_eo'
    if "relevant_eo" not in df.columns:
        df["relevant_eo"] = ""
    df["relevance"] = (
        df["relevant_eo"].astype(str)
        .str.extract(r"\b(Yes|No|Maybe)\b", expand=False)
        .fillna("")
    )
    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Date range filter
    min_date = pd.to_datetime(df["date_dt"].min()) if not df.empty else None
    max_date = pd.to_datetime(df["date_dt"].max()) if not df.empty else None
    if pd.isna(min_date) or pd.isna(max_date):
        date_range = None
    else:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    # Relevance filter
    rel_options = ["Any", "Yes", "Maybe", "No", ""]
    rel_choice = st.sidebar.selectbox("EO relevance", rel_options, index=0)

    # Types filter
    unique_types: List[str] = sorted({t.strip() for ts in df["types"].dropna().tolist() for t in str(ts).split("; ") if t.strip()})
    types_choice = st.sidebar.multiselect("Types", options=unique_types, default=[])

    # Search filter
    query = st.sidebar.text_input("Search (title/summary)", value="")

    filtered = df.copy()
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask = (filtered["date_dt"] >= start) & (filtered["date_dt"] <= end + pd.Timedelta(days=1))
        filtered = filtered[mask]

    if rel_choice != "Any":
        filtered = filtered[filtered["relevance"].fillna("") == rel_choice]

    if types_choice:
        tmask = filtered["types"].apply(lambda s: any(t in str(s) for t in types_choice))
        filtered = filtered[tmask]

    if query:
        q = query.lower()
        qmask = (
            filtered["title"].astype(str).str.lower().str.contains(q)
            | filtered["description"].astype(str).str.lower().str.contains(q)
            | filtered["full_summary"].astype(str).str.lower().str.contains(q)
        )
        filtered = filtered[qmask]

    return filtered


def discover_topic_csvs(base_csv: str) -> List[Tuple[str, str]]:
    """Find CSVs alongside base_csv and label them by topic.

    Returns a list of (label, path). Label prefers the CSV's 'relevance_topic' values
    (most frequent non-empty), then filename pattern '<root> - <Topic>.csv', else basename.
    The base CSV (no topic suffix) is listed first, then others alphabetically by label.
    """
    directory = os.path.dirname(base_csv) or "."
    base = os.path.basename(base_csv)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".csv"

    candidates: List[Tuple[str, str]] = []
    try:
        names = [n for n in os.listdir(directory) if n.lower().endswith(".csv")]
    except Exception:
        names = []

    for name in names:
        if not (name == f"{root}{ext}" or name.startswith(f"{root} - ")):
            continue
        path = os.path.join(directory, name)
        label: Optional[str] = None
        # Try reading a small sample to detect the topic
        try:
            df_head = pd.read_csv(path, nrows=200)
            if "relevance_topic" in df_head.columns:
                topics = [str(v).strip() for v in df_head["relevance_topic"].dropna().tolist() if str(v).strip()]
                if topics:
                    label = Counter(topics).most_common(1)[0][0]
        except Exception:
            pass

        if not label:
            m = re.match(rf"^{re.escape(root)} - (.+)\.csv$", name, flags=re.IGNORECASE)
            if m:
                label = m.group(1)

        if not label:
            label = name

        candidates.append((label, path))

    # Sort: base first, then others by label
    base_first: List[Tuple[str, str]] = []
    others: List[Tuple[str, str]] = []
    for label, path in candidates:
        if os.path.basename(path).lower() == f"{root}{ext}".lower():
            base_first.append((label, path))
        else:
            others.append((label, path))
    others.sort(key=lambda x: x[0].lower())
    return base_first + others


def run_tools_ui(base_csv: str) -> Optional[str]:
    st.sidebar.header("Run Tools")
    topic = st.sidebar.text_input("Relevance Topic (optional)", value="")
    base_csv_input = st.sidebar.text_input("Base CSV path", value=base_csv)
    topic_csv = tool.topic_csv_path(base_csv_input, topic or None)
    st.sidebar.caption(f"Output CSV: {topic_csv}")

    start_page = st.sidebar.number_input("Start page", min_value=1, value=1, step=1)
    lookback_days = st.sidebar.number_input("Lookback days", min_value=0, value=15, step=1,
                                            help="0 means no lookback cutoff")
    max_pages = st.sidebar.number_input("Max pages (0 = unlimited)", min_value=0, value=0, step=1)
    delay = st.sidebar.number_input("Delay between pages (s)", min_value=0.0, value=1.0, step=0.5)
    summarize = st.sidebar.checkbox("Summarize with OpenAI", value=True)
    update_missing = st.sidebar.checkbox("Update missing summaries in CSV", value=False)
    model = st.sidebar.text_input("OpenAI model", value="gpt-4o")

    if st.sidebar.button("Run scrape/update"):
        with st.status("Running scrape/update...", expanded=True) as status:
            try:
                tool.run(
                    output_dir="whitehouse_texts",
                    csv_file=topic_csv,
                    start_page=int(start_page),
                    max_pages=None if int(max_pages) == 0 else int(max_pages),
                    delay=float(delay),
                    test_single_page=False,
                    summarize=bool(summarize),
                    model=model,
                    lookback_days=None if int(lookback_days) == 0 else int(lookback_days),
                    update_missing=bool(update_missing),
                    relevance_topic=(topic or None),
                )
                status.update(label="Completed", state="complete")
                st.success(f"Wrote updates to {topic_csv}")
                return topic_csv
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Run failed: {e}")
    return None


def render_kpis(df: pd.DataFrame):
    total = len(df)
    rel_yes = (df["relevance"] == "Yes").sum()
    rel_maybe = (df["relevance"] == "Maybe").sum()
    rel_no = (df["relevance"] == "No").sum()
    latest = pd.to_datetime(df["date_dt"].max()) if total else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Actions", f"{total}")
    c2.metric("Yes", f"{rel_yes}")
    c3.metric("Maybe", f"{rel_maybe}")
    c4.metric("No", f"{rel_no}")
    if latest and not pd.isna(latest):
        st.caption(f"Latest date: {latest.date()}")


def render_charts(df: pd.DataFrame):
    if df.empty:
        return
    rel_counts = df["relevance"].replace({"": "Unlabeled"}).value_counts().sort_index()
    st.subheader("Relevance Breakdown")
    st.bar_chart(rel_counts)

    ts = df.dropna(subset=["date_dt"]).copy()
    if not ts.empty:
        daily = ts.groupby(ts["date_dt"].dt.to_period("D")).size().rename("count").to_timestamp()
        st.subheader("Actions Over Time")
        st.line_chart(daily)


def render_table(df: pd.DataFrame):
    if df.empty:
        st.info("No rows match current filters.")
        return

    show_cols = ["date", "title", "types", "relevance", "description"]
    st.subheader("Filtered Results")
    # Sort by date_dt (present in df) before selecting display columns
    st.dataframe(df.sort_values(by=["date_dt"], ascending=False)[show_cols], use_container_width=True)

    st.subheader("Details")
    for _, row in df.sort_values(by=["date_dt"], ascending=False).iterrows():
        title = str(row.get("title") or "(untitled)")
        with st.expander(f"{row.get('date', '')} — {title}"):
            url = str(row.get("url") or "")
            text_file = str(row.get("text_file") or "")
            st.markdown(f"- Title: **{title}**")
            if url:
                st.markdown(f"- URL: [{url}]({url})")
            if text_file and os.path.exists(text_file):
                st.markdown(f"- Text file: `{text_file}`")
            st.markdown(f"- Types: {row.get('types') or ''}")
            st.markdown(f"- Relevance: {row.get('relevance') or ''}")
            desc = str(row.get("description") or "")
            if desc:
                st.markdown(f"- Summary: {desc}")
            full = str(row.get("full_summary") or "")
            if full:
                st.markdown("Full Summary")
                st.write(full)

    # Download filtered CSV
    filtered_csv = df.drop(columns=["date_dt"]).to_csv(index=False)
    st.download_button("Download Filtered CSV", filtered_csv, file_name="eo_summaries_filtered.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="EO Summaries Dashboard", layout="wide")
    st.title("Executive Actions Summaries")
    st.caption("Dashboard for scraped Presidential actions and GPT summaries")

    # Allow running tools and deriving topic-specific CSV paths
    updated_csv = run_tools_ui(DEFAULT_CSV)
    # Discover available topic CSVs and select by label (topic)
    csv_options = discover_topic_csvs(DEFAULT_CSV)
    label_to_path = {label: path for label, path in csv_options}
    labels = [label for label, _ in csv_options]

    default_index = 0
    if updated_csv and labels:
        for i, (label, path) in enumerate(csv_options):
            if path == updated_csv:
                default_index = i
                break

    selected_label = None
    if labels:
        selected_label = st.sidebar.selectbox("Available CSVs (by topic)", labels, index=default_index)
        csv_path = label_to_path[selected_label]
    else:
        st.sidebar.info("No topic CSVs found. Run a scrape to create one.")
        csv_path = DEFAULT_CSV

    df = load_data(csv_path)
    if df.empty:
        st.stop()

    render_kpis(df)
    render_charts(df)

    filtered = sidebar_filters(df)
    render_table(filtered)


if __name__ == "__main__":
    # If not running under Streamlit, bail with a clear message to avoid noisy warnings.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    except Exception:
        get_script_run_ctx = None  # type: ignore

    if get_script_run_ctx is None or get_script_run_ctx() is None:
        import sys
        print(
            "This is a Streamlit app. Please launch via:\n\n"
            "    streamlit run streamlit_app.py\n\n"
            "Optionally add: --logger.level=debug to see detailed errors."
        )
        sys.exit(1)
    else:
        main()
