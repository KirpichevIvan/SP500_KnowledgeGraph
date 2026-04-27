import argparse
import os
import time
from typing import List

import pandas as pd
import wikipedia
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXCEL = os.path.abspath(os.path.join(BASE_DIR, "../data/sp500_graph_ready.xlsx"))
DEFAULT_NEWS_CSV = os.path.abspath(os.path.join(BASE_DIR, "../data/classified_reuters_news_mapped.csv"))
DEFAULT_OUT_TXT = os.path.abspath(os.path.join(BASE_DIR, "../data/graph_source_corpus.txt"))


def retry(func, max_retries=3, delay=2):
    for i in range(max_retries):
        try:
            return func()
        except Exception:
            if i == max_retries - 1:
                return None
            time.sleep(delay)
    return None


def get_wiki_summary_safe(name: str) -> str:
    def _inner():
        results = wikipedia.search(f"{name} company")
        if not results:
            return ""
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.summary[:1000]

    return retry(_inner, max_retries=2, delay=1) or ""


def format_company_block(row: pd.Series, wiki_summary: str) -> str:
    """Только сырой текст без служебных подписей/меток."""
    ordered_fields = [
        "Ticker",
        "Name",
        "Sector",
        "Industry",
        "Market Cap",
        "Employees",
        "Website",
        "Description",
        "Officers_JSON",
        "Holders_JSON",
        "Address_JSON",
    ]
    pieces: List[str] = []

    # Поля в исходном виде (включая JSON-строки как есть)
    for field in ordered_fields:
        val = str(row.get(field, "") or "").strip()
        if val and val.lower() != "nan":
            pieces.append(val)

    # Вики-текст (тот же объем, что использовался при построении)
    if wiki_summary:
        pieces.append(wiki_summary)

    # Только сплошной текст
    return "\n".join(pieces).strip()


def export_corpus(
    excel_path: str,
    output_txt: str,
    include_news_csv: bool,
    news_csv_path: str,
    limit: int,
):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if limit > 0:
        df = df.head(limit)

    with open(output_txt, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            name = str(row.get("Name", "") or "")
            print(f"[{i + 1}/{len(df)}] Exporting {name}...")

            wiki_summary = get_wiki_summary_safe(name)
            block = format_company_block(row, wiki_summary)
            if block:
                f.write(block + "\n\n")

        if include_news_csv and os.path.exists(news_csv_path):
            news_df = pd.read_csv(news_csv_path)
            for _, nrow in news_df.iterrows():
                values = [str(v) for v in nrow.to_dict().values() if str(v).strip() and str(v).lower() != "nan"]
                f.write("\n".join(values) + "\n\n")

    print(f"Done: {output_txt}")


def parse_args():
    p = argparse.ArgumentParser(description="Export all source info used for graph build into one TXT corpus.")
    p.add_argument("--excel", default=DEFAULT_EXCEL)
    p.add_argument("--output", default=DEFAULT_OUT_TXT)
    p.add_argument("--include-news-csv", action="store_true")
    p.add_argument("--news-csv", default=DEFAULT_NEWS_CSV)
    p.add_argument("--limit", type=int, default=0, help="0 = all companies")
    return p.parse_args()


def main():
    args = parse_args()
    export_corpus(
        excel_path=args.excel,
        output_txt=args.output,
        include_news_csv=args.include_news_csv,
        news_csv_path=args.news_csv,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()