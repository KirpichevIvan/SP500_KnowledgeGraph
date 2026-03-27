import os
import json
import pandas as pd
import datetime
import time
import wikipedia
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import functools


def retry(max_retries=3, delay=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        print(f"Ошибка в {func.__name__}: {e}")
                        return None
                    time.sleep(delay)

        return wrapper

    return decorator


load_dotenv(find_dotenv())

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MODEL_PATH = r"C:\sp500_models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'all-MiniLM-L6-v2'

EXCEL_PATH = '../data/sp500_graph_ready.xlsx'
NEWS_CSV_PATH = '../data/classified_reuters_news_mapped.csv'

THRESHOLD_MIN_CHECK = 0.70

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

SUPPORTED_NEWS_ENTITY_LABELS = tuple(NEWS_RELATION_BY_LABEL.keys())


class BaseGraphBuilder:
    def __init__(self):
        print("Инициализация Base Graph Builder...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            keep_alive=True,
            connection_timeout=10
        )
        self.client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1", timeout=30.0)

        print(f"Загрузка модели эмбеддингов...")
        try:
            self.embedder = SentenceTransformer(MODEL_PATH)
            self.vector_dim = 384
        except:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_dim = 384

    def close(self):
        self.driver.close()

    def _get_vector(self, text):
        if not text: return None
        return self.embedder.encode(str(text)).tolist()

    def reset_db(self):
        print("\nПолная очистка базы данных...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("DROP INDEX entity_vector_index IF EXISTS")
            except:
                pass

            try:
                session.run("CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
            except:
                pass

            for label in ["Person", "Product", "Resource", "News", "Fund", "City", "State", "Country", "Sector", "Industry"]:
                try:
                    session.run(f"CREATE INDEX {label.lower()}_name IF NOT EXISTS FOR (n:{label}) ON (n.name)")
                except:
                    pass

        print("База чиста.")

    @retry(max_retries=2, delay=1)
    def get_wiki_summary_safe(self, name):
        results = wikipedia.search(f"{name} company")
        if not results: return ""
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.summary[:1000]

    @retry(max_retries=3, delay=3)
    def extract_attributes_llm_safe(self, name, full_desc):
        prompt = f"""
        Analyze company: "{name}".
        Context: {full_desc[:1500]}
        Task: Extract two lists:
        1. "products": Specific products/services they SELL.
        2. "resources": Key raw materials/technologies/services they BUY (Inputs/Needs).
        Return JSON: {{ "products": ["A", "B"], "resources": ["X", "Y"] }}
        """
        resp = self.client.chat.completions.create(
            model='qwen/qwen-2.5-7b-instruct', messages=[{'role': 'user', 'content': prompt}], temperature=0.0
        )
        txt = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        if "{" in txt: txt = txt[txt.find("{"):txt.rfind("}") + 1]
        return json.loads(txt)

    def process_companies(self):
        print(f"\nЗагрузка ядра графа из {EXCEL_PATH}...")
        if not os.path.exists(EXCEL_PATH): return

        df = pd.read_excel(EXCEL_PATH)
        today = datetime.date.today().isoformat()

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Building Core"):
            ticker = row['Ticker']
            name = row['Name']
            yf_desc = str(row['Description'])

            wiki_text = self.get_wiki_summary_safe(name) or ""
            full_desc = f"Yahoo: {yf_desc}\nWikipedia: {wiki_text}"

            with self.driver.session() as session:
                session.run("""
                    MERGE (c:Company {ticker: $t})
                    SET c.name = $n, c.description = $d, c.wiki_summary = $w, c.last_updated = $dt
                    MERGE (s:Sector {name: $sect})
                    MERGE (i:Industry {name: $ind})
                    MERGE (c)-[:OPERATES_IN_INDUSTRY]->(i)
                    MERGE (i)-[:PART_OF]->(s)
                """, t=ticker, n=name, d=yf_desc[:600], w=wiki_text, dt=today,
                            sect=row.get('Sector', 'Unknown'), ind=row.get('Industry', 'Unknown'))

                try:
                    addr = json.loads(row['Address_JSON'])
                    city = addr.get('city')
                    state = addr.get('state')
                    country = addr.get('country')

                    if city and city != 'N/A':
                        session.run("""
                            MATCH (c:Company {ticker: $t})
                            MERGE (city:City {name: $city})
                            MERGE (cntry:Country {name: $country})
                            MERGE (c)-[:LOCATED_IN]->(city)

                            // Если есть штат -> связываем через штат
                            FOREACH (_ IN CASE WHEN $state IS NOT NULL AND $state <> 'N/A' THEN [1] ELSE [] END |
                                MERGE (s:State {name: $state})
                                MERGE (city)-[:IN_STATE]->(s)
                                MERGE (s)-[:IN_COUNTRY]->(cntry)
                            )

                            // Если штата нет -> связываем напрямую
                            FOREACH (_ IN CASE WHEN $state IS NULL OR $state = 'N/A' THEN [1] ELSE [] END |
                                MERGE (city)-[:IN_COUNTRY]->(cntry)
                            )
                        """, t=ticker, city=city, state=state, country=country)
                except:
                    pass

                try:
                    officers = json.loads(row['Officers_JSON'])
                    for off in officers:
                        if off.get('name'):
                            session.run("""
                                MATCH (c:Company {ticker: $t})
                                MERGE (p:Person {name: $name})
                                MERGE (p)-[:WORKS_FOR {title: $title}]->(c)
                            """, t=ticker, name=off['name'], title=off.get('title', 'Unknown'))
                except:
                    pass

                try:
                    holders = json.loads(row['Holders_JSON'])
                    for h in holders:
                        if h.get('Holder'):
                            session.run("""
                                MATCH (c:Company {ticker: $t})
                                MERGE (f:Fund {name: $fname})
                                MERGE (f)-[:OWNS {pct: $pct}]->(c)
                            """, t=ticker, fname=h['Holder'], pct=h.get('pctHeld', 0))
                except:
                    pass

                attrs = self.extract_attributes_llm_safe(name, full_desc)
                if attrs:
                    for prod in attrs.get('products', []):
                        session.run(
                            "MATCH (c:Company {ticker:$t}) MERGE (p:Product {name:$n}) MERGE (c)-[:PRODUCES]->(p)",
                            t=ticker, n=prod)
                    for res in attrs.get('resources', []):
                        session.run(
                            "MATCH (c:Company {ticker:$t}) MERGE (r:Resource {name:$n}) MERGE (c)-[:REQUIRES]->(r)",
                            t=ticker, n=res)

    def build_vector_index(self):
        print("\nСоздание векторного индекса...")
        target_labels = [
            "Company",
            "Fund",
            "Sector",
            "Industry",
            "City",
            "State",
            "Country",
            "Product",
            "Resource",
        ]

        with self.driver.session() as session:
            session.run(f"MATCH (n) WHERE any(l in labels(n) WHERE l IN {json.dumps(target_labels)}) SET n:Searchable")

            nodes = list(session.run("""
                MATCH (n:Searchable) 
                RETURN elementId(n) as id, labels(n) as labels, 
                       n.name as name, n.ticker as ticker, 
                       n.description as desc, n.wiki_summary as wiki
            """))

            for record in tqdm(nodes, desc="Vectorizing"):
                labels = [l for l in record['labels'] if l != 'Searchable']
                main_label = labels[0] if labels else "Entity"

                name_val = record.get('name', 'Unknown')
                text = f"{main_label}: {name_val}"

                if record.get('ticker'): text += f" ({record['ticker']})"

                desc_parts = []
                if record.get('desc'): desc_parts.append(str(record['desc'])[:300])
                if record.get('wiki'): desc_parts.append(str(record['wiki'])[:300])
                if desc_parts: text += " - " + " ".join(desc_parts)

                vec = self._get_vector(text)
                session.run("MATCH (n) WHERE elementId(n)=$id CALL db.create.setNodeVectorProperty(n, 'embedding', $v)",
                            id=record['id'], v=vec)

            session.run(f"""
                CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS FOR (n:Searchable) ON (n.embedding) 
                OPTIONS {{indexConfig: {{`vector.dimensions`: {self.vector_dim}, `vector.similarity_function`: 'cosine'}}}}
            """)

            print("Ожидание индексации...")
            while True:
                r = session.run("SHOW INDEXES YIELD name, state WHERE name='entity_vector_index' RETURN state").single()
                if r and r['state'] == 'ONLINE': break
                time.sleep(1)
            print("Индекс готов.")

    def find_node(self, text_query, allowed_labels=None):
        if not text_query: return None
        vec = self._get_vector(text_query)
        with self.driver.session() as session:
            res = list(session.run("""
                CALL db.index.vector.queryNodes('entity_vector_index', 5, $v) 
                YIELD node, score WHERE score > $th RETURN node, score, labels(node) as lbl
            """, v=vec, th=THRESHOLD_MIN_CHECK))

            if not res: return None

            candidates = []
            normalized_allowed_labels = set(allowed_labels or [])

            for r in res:
                name = r['node'].get('name', '')
                lbl = [l for l in r['lbl'] if l != 'Searchable'][0]
                if normalized_allowed_labels and lbl not in normalized_allowed_labels:
                    continue
                ts = fuzz.token_sort_ratio(text_query.lower(), name.lower())
                candidates.append({'node': r['node'], 'score': r['score'], 'ts': ts, 'label': lbl, 'name': name})

            if not candidates:
                return None

            best = max(candidates, key=lambda x: x['score'] + (0.05 if x['label'] == 'Company' else 0))

            if best['score'] > 0.84 or best['ts'] > 80:
                n = best['node']
                key = "ticker" if best['label'] == 'Company' else "name"
                return {"id": n.get(key) or n.get("name"), "key": key, "label": best['label']}

            return None

    @retry(max_retries=3, delay=1)
    def analyze_news_llm_safe(self, headline):
        prompt = f"""
        Analyze news: "{headline}"
        Task: Extract MAIN entities with their type.
        Allowed types: Company, Fund, City, State, Country, Sector, Industry, Product, Resource.
        Do not extract Person entities because person names are often abbreviated or ambiguous in headlines.
        Include only entities that are really central to the news.
        Use the exact type only when you are confident, because downstream matching will only search inside that entity type.
        Determine Sentiment (-1.0 to 1.0).
        Return JSON: {{
          "entities": [
            {{"name": "Name1", "type": "Company"}},
            {{"name": "Name2", "type": "Country"}}
          ],
          "sentiment": 0.5
        }}
        """
        resp = self.client.chat.completions.create(model='qwen/qwen-2.5-7b-instruct',
                                                   messages=[{'role': 'user', 'content': prompt}], temperature=0.0)
        txt = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        if "{" in txt: txt = txt[txt.find("{"):txt.rfind("}") + 1]
        return json.loads(txt)

    def _normalize_news_entities(self, entities):
        normalized = []
        if not isinstance(entities, list):
            return normalized

        for ent in entities:
            if isinstance(ent, dict):
                name = str(ent.get("name", "")).strip()
                label = str(ent.get("type", "")).strip()
            else:
                name = str(ent).strip()
                label = ""

            if not name:
                continue

            if label and label not in SUPPORTED_NEWS_ENTITY_LABELS:
                continue

            normalized.append({"name": name, "type": label})

        return normalized

    def _match_news_entity(self, raw_entity):
        raw_name = raw_entity.get("name", "")
        preferred_type = raw_entity.get("type", "")
        allowed_labels = [preferred_type] if preferred_type else None

        match = self.find_node(raw_name, allowed_labels=allowed_labels)
        if not match:
            return None

        return match

    def _deduplicate_news_entities(self, matches):
        unique_matches = []
        seen = set()

        for match in matches:
            key = (match["label"], match["key"], match["id"])
            if key in seen:
                continue
            seen.add(key)
            unique_matches.append(match)

        return unique_matches

    def _news_relation_type_for_label(self, label):
        return NEWS_RELATION_BY_LABEL.get(label, "MENTIONS")

    def process_news(self):
        print(f"\nОбработка новостей из {NEWS_CSV_PATH}...")
        if not os.path.exists(NEWS_CSV_PATH): return
        df = pd.read_csv(NEWS_CSV_PATH)

        with self.driver.session() as session:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="News Linking"):
                headline = str(row.get('headline', row.get('Headlines', '')))
                date_str = str(row.get('Time', row.get('date', datetime.date.today())))
                try:
                    iso_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
                except:
                    iso_date = datetime.date.today().isoformat()

                an = self.analyze_news_llm_safe(headline)
                if not an: continue

                found = []
                for raw_ent in self._normalize_news_entities(an.get('entities', [])):
                    match = self._match_news_entity(raw_ent)
                    if match:
                        found.append(match)

                found = self._deduplicate_news_entities(found)

                if not found: continue

                for ent in found:
                    rel_type = self._news_relation_type_for_label(ent['label'])
                    session.run(f"""
                        MATCH (e:{ent['label']} {{ {ent['key']}: $eid }})
                        MERGE (n:News {{headline: $hl, date: date($dt)}})
                        SET n.sentiment = $sent
                        MERGE (n)-[r:{rel_type}]->(e)
                        SET r.entity_type = $entity_type,
                            r.entity_label = $entity_label
                    """, eid=ent['id'], hl=headline, dt=iso_date, sent=an.get('sentiment', 0),
                         entity_type=("Location" if ent['label'] in {"City", "State", "Country"} else ent['label']),
                         entity_label=ent['label'])

    def run(self):
        self.reset_db()
        self.process_companies()
        self.build_vector_index()
        self.process_news()
        print("\nБазовый граф готов.")


if __name__ == "__main__":
    builder = BaseGraphBuilder()
    builder.run()
    builder.close()