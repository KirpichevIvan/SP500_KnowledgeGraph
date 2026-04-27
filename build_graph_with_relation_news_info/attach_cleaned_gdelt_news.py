import argparse
import datetime as dt
import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.abspath(os.path.join(BASE_DIR, "../gdelt_entity_news_cleaned.csv"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEWS_RELATION_BY_LABEL = {
    "Company": "NEWS_ABOUT_COMPANY",
    "Fund": "NEWS_ABOUT_FUND",
    "City": "NEWS_ABOUT_LOCATION",
    "State": "NEWS_ABOUT_LOCATION",
    "Country": "NEWS_ABOUT_LOCATION",
    "Sector": "NEWS_ABOUT_SECTOR",
    "Industry": "NEWS_ABOUT_INDUSTRY",
    "Product": "NEWS_ABOUT_PRODUCT",
    "Resource": "NEWS_ABOUT_RESOURCE",
}


def parse_iso_date(row: pd.Series) -> str:
    candidates = [
        row.get("seendate", ""),
        row.get("collected_at_utc", ""),
    ]

    for c in candidates:
        text = str(c or "").strip()
        if not text:
            continue
        try:
            if "T" in text and text.endswith("Z") and len(text) >= 15:
                # формат GDELT: 20260405T233000Z
                return dt.datetime.strptime(text, "%Y%m%dT%H%M%SZ").date().isoformat()
            return pd.to_datetime(text).date().isoformat()
        except Exception:
            pass

    return dt.date.today().isoformat()


def relation_for(label: str) -> str:
    return NEWS_RELATION_BY_LABEL.get(label, "MENTIONS")


def attach_news(csv_path: str, dry_run: bool = False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["entity_type", "entity_name", "news_title", "news_url"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Rows in cleaned CSV: {len(df)}")

    supported_labels = set(NEWS_RELATION_BY_LABEL.keys())
    df = df[df["entity_type"].astype(str).isin(supported_labels)].copy()
    print(f"Rows with supported entity_type: {len(df)}")

    if dry_run:
        print("Dry-run mode: no Neo4j writes")
        print(df["entity_type"].value_counts().to_dict())
        return

    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD are required")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        keep_alive=True,
        connection_timeout=10,
    )

    linked = 0
    missed_entities = 0

    with driver.session() as session:
        session.run("CREATE INDEX news_url IF NOT EXISTS FOR (n:News) ON (n.url)")

        for _, row in df.iterrows():
            label = str(row.get("entity_type", "")).strip()
            entity_name = str(row.get("entity_name", "")).strip()
            rel_type = relation_for(label)
            iso_date = parse_iso_date(row)

            title = str(row.get("news_title", "") or "")
            snippet = str(row.get("news_snippet", "") or "")
            url = str(row.get("news_url", "") or "").strip()
            domain = str(row.get("domain", "") or "")
            news_query = str(row.get("query", "") or "")
            language = str(row.get("language", "") or "")
            sourcecountry = str(row.get("sourcecountry", "") or "")
            seendate = str(row.get("seendate", "") or "")
            collected_at = str(row.get("collected_at_utc", "") or "")

            if not entity_name or not url:
                continue

            # Company: пробуем сначала по name, затем по ticker
            if label == "Company":
                ent_query = """
                    MATCH (e:Company)
                    WHERE e.name = $entity_name OR e.ticker = $entity_name
                    RETURN e LIMIT 1
                """
            else:
                ent_query = f"MATCH (e:{label} {{name: $entity_name}}) RETURN e LIMIT 1"

            ent = session.run(ent_query, entity_name=entity_name).single()
            if not ent:
                missed_entities += 1
                continue

            session.run(
                f"""
                MATCH (e:{label})
                WHERE {("e.name = $entity_name OR e.ticker = $entity_name") if label == "Company" else "e.name = $entity_name"}
                MERGE (n:News {{url: $url}})
                SET n.headline = $title,
                    n.snippet = $snippet,
                    n.date = date($iso_date),
                    n.domain = $domain,
                    n.query = $news_query,
                    n.language = $language,
                    n.sourcecountry = $sourcecountry,
                    n.seendate = $seendate,
                    n.collected_at_utc = $collected_at,
                    n.raw_text = trim($title + ' ' + $snippet)
                MERGE (n)-[r:{rel_type}]->(e)
                SET r.entity_type = $entity_type,
                    r.entity_label = $entity_label,
                    r.entity_name = $entity_name
                """,
                entity_name=entity_name,
                url=url,
                title=title,
                snippet=snippet,
                iso_date=iso_date,
                domain=domain,
                news_query=news_query,
                language=language,
                sourcecountry=sourcecountry,
                seendate=seendate,
                collected_at=collected_at,
                entity_type=("Location" if label in {"City", "State", "Country"} else label),
                entity_label=label,
            )
            linked += 1

    driver.close()
    print(f"Linked news rows: {linked}")
    print(f"Missed entities: {missed_entities}")


def parse_args():
    p = argparse.ArgumentParser(description="Attach cleaned GDELT news CSV to existing Neo4j graph.")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    attach_news(csv_path=args.csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()