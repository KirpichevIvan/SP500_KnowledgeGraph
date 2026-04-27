import csv
import datetime as dt
import json
import os
import random
import re
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(find_dotenv())

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/sp500_graph_ready.xlsx"))
OUT_CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "gdelt_entity_news.csv"))
OUT_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "gdelt_entity_news_summary.json"))

ENTITY_ORDER = ["Company", "Sector", "Industry", "Fund", "City", "Product", "Resource"]

COLUMN_MAP = {
    "Company": "Name",
    "Sector": "Sector",
    "Industry": "Industry",
}

PRODUCT_HINTS = (
    "software", "platform", "service", "services", "device", "equipment", "chip", "semiconductor",
    "pharmaceutical", "drug", "vaccine", "insurance", "payment", "cloud", "network", "vehicle",
    "aircraft", "processor", "application", "solution", "battery"
)
RESOURCE_HINTS = (
    "oil", "gas", "natural gas", "lithium", "copper", "steel", "aluminum", "water",
    "electricity", "power", "coal", "uranium", "silicon", "nickel", "grain", "timber"
)

class GdeltEntityNewsAgent:
    def __init__(self):
        self.excel_path = os.getenv("GDELT_ENTITY_EXCEL_PATH", EXCEL_PATH)
        self.out_csv_path = os.getenv("GDELT_OUT_CSV", OUT_CSV_PATH)
        self.out_json_path = os.getenv("GDELT_OUT_JSON", OUT_JSON_PATH)
        self.entity_source = os.getenv("GDELT_ENTITY_SOURCE", "neo4j_first")
        self.max_records = int(os.getenv("GDELT_MAXRECORDS", "20"))
        self.timespan = os.getenv("GDELT_TIMESPAN", "3m")
        self.min_request_delay = float(os.getenv("GDELT_MIN_DELAY_SEC", "1.2"))
        self.max_retries = int(os.getenv("GDELT_MAX_RETRIES", "5"))
        self.request_timeout = int(os.getenv("GDELT_TIMEOUT_SEC", "35"))
        self.require_neo4j = os.getenv("GDELT_REQUIRE_NEO4J", "0") == "1"

        # Можно ограничить количество сущностей каждого типа (0 = без лимита)
        self.per_type_limit = int(os.getenv("ENTITY_PER_TYPE_LIMIT", "0"))

        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.session = self._build_session()
        self._last_call_ts = 0.0
        self.dynamic_cooldown_sec = 0.0
        self.consecutive_429 = 0
        self.entity_stats = {k: {"nodes_total": 0, "unique_names": 0, "query_names": 0} for k in ENTITY_ORDER}

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; SP500NewsGraphBot/1.0; +https://example.local)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
        )

        retry = Retry(
            total=self.max_retries,
            backoff_factor=1.2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _safe_json_loads(value: object):
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    @staticmethod
    def _clean_phrase(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip(" ,.;:-")
        text = re.sub(r"\b(the|and|for|with|from|into|that|this)\b", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
        if len(text) < 3:
            return ""
        return text

    def _extract_products_resources_from_description(self, description: str) -> Tuple[List[str], List[str]]:
        if not isinstance(description, str) or not description.strip():
            return [], []

        desc = description.lower()
        chunks = re.split(r"[.;]", desc)
        products: List[str] = []
        resources: List[str] = []

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if any(h in chunk for h in PRODUCT_HINTS):
                phrase = self._clean_phrase(chunk)
                if phrase:
                    products.append(phrase[:90])

            if any(h in chunk for h in RESOURCE_HINTS):
                phrase = self._clean_phrase(chunk)
                if phrase:
                    resources.append(phrase[:90])

        # Дедуп + ограничение шума
        products = list(OrderedDict((p, None) for p in products if len(p.split()) <= 12))[:5]
        resources = list(OrderedDict((r, None) for r in resources if len(r.split()) <= 10))[:5]
        return products, resources

    def _normalize_entities(self, entities: Dict[str, OrderedDict]) -> Dict[str, List[str]]:
        out = {}
        for label in ENTITY_ORDER:
            values = list(entities[label].keys())
            self.entity_stats[label]["unique_names"] = len(values)
            if self.per_type_limit > 0:
                values = values[: self.per_type_limit]
            self.entity_stats[label]["query_names"] = len(values)
            out[label] = values
        return out

    def _load_entities_from_excel(self) -> Dict[str, List[str]]:
        df = pd.read_excel(self.excel_path)

        entities: Dict[str, OrderedDict] = {k: OrderedDict() for k in ENTITY_ORDER}
        self.entity_stats = {k: {"nodes_total": 0, "unique_names": 0, "query_names": 0} for k in ENTITY_ORDER}

        # Company / Sector / Industry
        for label, col in COLUMN_MAP.items():
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    cleaned = val.strip()
                    if cleaned and cleaned.lower() != "unknown":
                        entities[label][cleaned] = None

        # Fund из Holders_JSON
        if "Holders_JSON" in df.columns:
            for raw in df["Holders_JSON"].dropna().tolist():
                parsed = self._safe_json_loads(raw)
                if not isinstance(parsed, list):
                    continue
                for holder in parsed:
                    if isinstance(holder, dict):
                        h_name = str(holder.get("Holder", "")).strip()
                        if h_name and h_name.lower() != "nan":
                            entities["Fund"][h_name] = None

        # City из Address_JSON
        if "Address_JSON" in df.columns:
            for raw in df["Address_JSON"].dropna().tolist():
                parsed = self._safe_json_loads(raw)
                if not isinstance(parsed, dict):
                    continue
                city = str(parsed.get("city", "")).strip()
                if city and city.lower() != "nan":
                    entities["City"][city] = None

        # Product / Resource из колонок или эвристики description
        product_cols = [c for c in ["Products_JSON", "Products", "Product"] if c in df.columns]
        resource_cols = [c for c in ["Resources_JSON", "Resources", "Resource"] if c in df.columns]

        for _, row in df.iterrows():
            for col in product_cols:
                raw = row.get(col)
                parsed = self._safe_json_loads(raw)
                if isinstance(parsed, list):
                    for p in parsed:
                        p_name = str(p).strip()
                        if p_name:
                            entities["Product"][p_name] = None
                elif isinstance(raw, str):
                    for p in re.split(r"[,;|]", raw):
                        p_name = p.strip()
                        if p_name:
                            entities["Product"][p_name] = None

            for col in resource_cols:
                raw = row.get(col)
                parsed = self._safe_json_loads(raw)
                if isinstance(parsed, list):
                    for r in parsed:
                        r_name = str(r).strip()
                        if r_name:
                            entities["Resource"][r_name] = None
                elif isinstance(raw, str):
                    for r in re.split(r"[,;|]", raw):
                        r_name = r.strip()
                        if r_name:
                            entities["Resource"][r_name] = None

            description = row.get("Description", "")
            p_candidates, r_candidates = self._extract_products_resources_from_description(description)
            for p in p_candidates:
                entities["Product"][p] = None
            for r in r_candidates:
                entities["Resource"][r] = None

        return self._normalize_entities(entities)

    def _load_entities_from_neo4j(self) -> Dict[str, List[str]]:
        if not (self.neo4j_uri and self.neo4j_user and self.neo4j_password):
            raise RuntimeError("NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD not set")

        entities: Dict[str, OrderedDict] = {k: OrderedDict() for k in ENTITY_ORDER}
        self.entity_stats = {k: {"nodes_total": 0, "unique_names": 0, "query_names": 0} for k in ENTITY_ORDER}
        label_to_prop = {
            "Company": "name",
            "Sector": "name",
            "Industry": "name",
            "Fund": "name",
            "City": "name",
            "Product": "name",
            "Resource": "name",
        }

        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            max_connection_lifetime=200,
            keep_alive=True,
        )

        try:
            with driver.session() as session:
                for label in ENTITY_ORDER:
                    key = label_to_prop[label]
                    stats_query = (
                        f"MATCH (n:{label}) "
                        f"WHERE n.{key} IS NOT NULL AND trim(toString(n.{key})) <> '' "
                        f"RETURN count(n) AS total_nodes, count(DISTINCT toString(n.{key})) AS unique_names"
                    )
                    stats_rec = session.run(stats_query).single()
                    self.entity_stats[label]["nodes_total"] = int(stats_rec["total_nodes"] or 0)
                    self.entity_stats[label]["unique_names"] = int(stats_rec["unique_names"] or 0)

                    query = (
                        f"MATCH (n:{label}) "
                        f"WHERE n.{key} IS NOT NULL AND trim(toString(n.{key})) <> '' "
                        f"RETURN DISTINCT toString(n.{key}) AS value "
                        "ORDER BY value"
                    )
                    records = session.run(query)
                    for rec in records:
                        value = str(rec["value"]).strip()
                        if value and value.lower() != "unknown":
                            entities[label][value] = None
        finally:
            driver.close()

        return self._normalize_entities(entities)

    def load_entities(self) -> Dict[str, List[str]]:
        if self.entity_source not in {"neo4j", "excel", "neo4j_first"}:
            print(f"[WARN] Unknown GDELT_ENTITY_SOURCE={self.entity_source}, fallback to neo4j_first")
            self.entity_source = "neo4j_first"

        if self.entity_source == "excel":
            print("[INFO] Entity source: Excel")
            return self._load_entities_from_excel()

        if self.entity_source == "neo4j":
            print("[INFO] Entity source: Neo4j")
            return self._load_entities_from_neo4j()

        print("[INFO] Entity source: Neo4j first, then Excel fallback")
        try:
            entities = self._load_entities_from_neo4j()
            if sum(len(v) for v in entities.values()) > 0:
                return entities
            print("[WARN] Neo4j returned 0 entities, fallback to Excel")
        except Exception as e:
            if self.require_neo4j:
                raise RuntimeError(f"Neo4j required but unavailable: {e}") from e
            print(f"[WARN] Neo4j unavailable ({e}), fallback to Excel")

        if self.require_neo4j:
            raise RuntimeError("Neo4j required but no entities loaded")
        return self._load_entities_from_excel()

    def _respect_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_call_ts
        effective_delay = self.min_request_delay + self.dynamic_cooldown_sec
        if elapsed < effective_delay:
            time.sleep((effective_delay - elapsed) + random.uniform(0.05, 0.35))
        self._last_call_ts = time.time()

    def _build_queries(self, entity_name: str, entity_type: str) -> List[str]:
        base = f'"{entity_name}"'
        type_context = {
            "Company": [base, f'{base} AND (stock OR earnings OR shares OR company)'],
            "Sector": [f'{base} AND (sector OR industry OR companies)'],
            "Industry": [f'{base} AND (industry OR market OR companies)'],
            "Fund": [f'{base} AND (fund OR etf OR holdings OR asset manager)'],
            "City": [f'{base} AND (company OR business OR economy OR headquarters)'],
            "Product": [f'{base} AND (product OR launch OR sales OR demand)'],
            "Resource": [f'{base} AND (commodity OR supply OR price OR production)'],
        }
        return type_context.get(entity_type, [base])

    def _request_gdelt(self, query: str) -> Tuple[List[dict], int]:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "timespan": self.timespan,
            "maxrecords": self.max_records,
            "sort": "datedesc",
        }

        self._respect_rate_limit()
        try:
            response = self.session.get(GDELT_URL, params=params, timeout=self.request_timeout)
        except requests.RequestException as e:
            print(f"      [!] request error: {e}")
            return [], 0

        if response.status_code == 429:
            self.consecutive_429 += 1
            self.dynamic_cooldown_sec = min(20.0, self.dynamic_cooldown_sec + 1.5)
            print(
                f"      [!] HTTP 429 for query={query}. "
                f"Cooldown={self.dynamic_cooldown_sec:.1f}s, streak={self.consecutive_429}"
            )
            return [], 429

        # при успешном/другом ответе сбрасываем streak 429
        self.consecutive_429 = 0
        self.dynamic_cooldown_sec = max(0.0, self.dynamic_cooldown_sec - 0.3)
        if response.status_code != 200:
            print(f"      [!] HTTP {response.status_code} for query={query}")
            return [], response.status_code

        try:
            payload = response.json()
        except Exception as e:
            print(f"      [!] json decode error: {e}")
            return [], 200

        if not isinstance(payload, dict):
            return [], 200
        return payload.get("articles", []) or [], 200

    def fetch_news_for_entity(self, entity_name: str, entity_type: str) -> List[dict]:
        all_articles = OrderedDict()
        queries = self._build_queries(entity_name, entity_type)

        for query in queries:
            print(f"   -> {entity_type}: {entity_name} | query={query}")
            articles, status_code = self._request_gdelt(query)

            for art in articles:
                url = str(art.get("url", "")).strip()
                if not url:
                    continue
                if url not in all_articles:
                    all_articles[url] = {
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "query": query,
                        "news_title": art.get("title", ""),
                        "news_snippet": art.get("snippet", ""),
                        "news_url": url,
                        "domain": art.get("domain", ""),
                        "seendate": art.get("seendate", ""),
                        "language": art.get("language", ""),
                        "sourcecountry": art.get("sourcecountry", ""),
                    }

            if len(all_articles) >= 5:
                break
            if status_code == 429 and not all_articles:
                break

        return list(all_articles.values())

    def _init_output_csv(self):
        with open(self.out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "entity_type",
                    "entity_name",
                    "query",
                    "news_title",
                    "news_snippet",
                    "news_url",
                    "domain",
                    "seendate",
                    "language",
                    "sourcecountry",
                    "collected_at_utc",
                ],
            )
            writer.writeheader()

    def _append_rows_csv(self, rows: Iterable[dict]):
        rows = list(rows)
        if not rows:
            return

        with open(self.out_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "entity_type",
                    "entity_name",
                    "query",
                    "news_title",
                    "news_snippet",
                    "news_url",
                    "domain",
                    "seendate",
                    "language",
                    "sourcecountry",
                    "collected_at_utc",
                ],
            )
            for row in rows:
                row["collected_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                writer.writerow(row)

    def run(self):
        print("[1/3] Загружаю сущности...")
        entities = self.load_entities()

        total_entities = sum(len(v) for v in entities.values())
        print(f"[INFO] Всего сущностей к обходу: {total_entities}")
        for t in ENTITY_ORDER:
            st = self.entity_stats.get(t, {})
            if st.get("nodes_total", 0):
                print(
                    f"   - {t}: query_names={len(entities[t])}, "
                    f"unique_names={st.get('unique_names', 0)}, nodes_total={st.get('nodes_total', 0)}"
                )
            else:
                print(f"   - {t}: {len(entities[t])}")

        print("[2/3] Запрашиваю новости GDELT...")
        self._init_output_csv()
        summary = {k: {"entities": len(v), "with_news": 0, "articles": 0} for k, v in entities.items()}

        for entity_type in ENTITY_ORDER:
            for entity_name in entities[entity_type]:
                try:
                    news_rows = self.fetch_news_for_entity(entity_name, entity_type)
                    self._append_rows_csv(news_rows)

                    if news_rows:
                        summary[entity_type]["with_news"] += 1
                        summary[entity_type]["articles"] += len(news_rows)

                except Exception as e:
                    # Не падаем на одной сущности — просто логируем и едем дальше
                    print(f"      [!] entity failed: {entity_type}/{entity_name}: {e}")
                    continue

        print("[3/3] Сохраняю сводку...")
        with open(self.out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nГотово.")
        print(f"CSV: {self.out_csv_path}")
        print(f"JSON summary: {self.out_json_path}")


if __name__ == "__main__":
    agent = GdeltEntityNewsAgent()
    agent.run()