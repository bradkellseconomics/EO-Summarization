import os
import csv
from typing import List

import streamlit as st
import pandas as pd


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

    csv_path = st.sidebar.text_input("CSV path", value=DEFAULT_CSV)
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
