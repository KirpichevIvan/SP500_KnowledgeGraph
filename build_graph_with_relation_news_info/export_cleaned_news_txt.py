import argparse
import datetime as dt
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_CSV = os.path.abspath(os.path.join(BASE_DIR, "../gdelt_entity_news_cleaned.csv"))
DEFAULT_OUTPUT_TXT = os.path.abspath(os.path.join(BASE_DIR, "../gdelt_entity_news_cleaned.txt"))


def parse_date(row: pd.Series) -> str:
    candidates = [row.get("seendate", ""), row.get("collected_at_utc", "")]
    for c in candidates:
        text = str(c or "").strip()
        if not text:
            continue
        try:
            if "T" in text and text.endswith("Z") and len(text) >= 15:
                return dt.datetime.strptime(text, "%Y%m%dT%H%M%SZ").date().isoformat()
            return pd.to_datetime(text).date().isoformat()
        except Exception:
            pass
    return dt.date.today().isoformat()


def safe_text(v: object) -> str:
    t = str(v or "").strip()
    return "" if t.lower() == "nan" else t


def build_content(row: pd.Series) -> str:
    snippet = safe_text(row.get("news_snippet", ""))
    title = safe_text(row.get("news_title", ""))
    if snippet:
        return snippet
    return title


def export_txt(input_csv: str, output_txt: str, limit: int = 0):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = ["entity_name", "entity_type", "news_title", "news_url"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if limit > 0:
        df = df.head(limit)

    with open(output_txt, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            entity_name = safe_text(row.get("entity_name", "Unknown"))
            entity_type = safe_text(row.get("entity_type", "Entity"))
            date_iso = parse_date(row)
            source = safe_text(row.get("domain", "")) or safe_text(row.get("sourcecountry", ""))
            title = safe_text(row.get("news_title", ""))
            content = build_content(row)
            url = safe_text(row.get("news_url", ""))

            f.write(f"[ENTITY] {entity_name} ({entity_type})\n")
            f.write(f"[DATE] {date_iso}\n")
            f.write(f"[SOURCE] {source}\n")
            f.write(f"[TITLE] {title}\n")
            f.write(f"[CONTENT] {content}\n")
            f.write(f"[URL] {url}\n\n")

    print(f"Exported {len(df)} records to {output_txt}")


def parse_args():
    p = argparse.ArgumentParser(description="Export cleaned GDELT CSV into tagged TXT format.")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    return p.parse_args()


def main():
    args = parse_args()
    export_txt(input_csv=args.input_csv, output_txt=args.output_txt, limit=args.limit)


if __name__ == "__main__":
    main()