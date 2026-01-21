import os
import json
import pandas as pd
import datetime
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

load_dotenv(find_dotenv())

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MODEL_PATH = r"C:\sp500_models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'all-MiniLM-L6-v2'

THRESHOLD_AUTO_ACCEPT = 0.84
THRESHOLD_LLM_CHECK = 0.70


class NewsProcessor:
    def __init__(self):
        print("Инициализация News Processor (Final Stable)...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=200,
            keep_alive=True
        )
        self.client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")

        print(f"Загрузка модели ({MODEL_PATH})...")
        try:
            self.embedder = SentenceTransformer(MODEL_PATH)
            self.vector_dim = 384
            print("Модель загружена.")
        except Exception as e:
            print(f"Ошибка модели: {e}")
            exit()

    def close(self):
        self.driver.close()

    def _get_vector(self, text):
        if not text: return None
        return self.embedder.encode(str(text)).tolist()

    def setup_search_index(self):
        print("\n--- НАСТРОЙКА ВЕКТОРНОГО ПОИСКА ---")
        target_labels = ["Company", "Fund", "Industry", "City", "Country", "State"]

        with self.driver.session() as session:
            try:
                session.run("DROP INDEX entity_vector_index IF EXISTS")
            except:
                pass

            print("1. Очистка старых меток...")
            while True:
                r = session.run(
                    "MATCH (n:Searchable) WITH n LIMIT 1000 REMOVE n:Searchable, n.embedding RETURN count(n) as c")
                if r.single()['c'] == 0: break

            print("2. Маркировка и векторизация...")
            session.run(
                f"MATCH (n) WHERE any(l in labels(n) WHERE l IN {json.dumps(target_labels)}) SET n:Searchable")

            nodes = list(session.run("""
                MATCH (n:Searchable) 
                RETURN elementId(n) as id, labels(n) as labels, 
                       n.name as name, 
                       n.ticker as ticker, 
                       n.description as description
            """))

            for record in tqdm(nodes, desc="Vectorizing"):
                labels = [l for l in record['labels'] if l != 'Searchable']
                main_label = labels[0] if labels else "Entity"

                name_val = record.get('name', 'Unknown')
                ticker_val = record.get('ticker')
                desc_val = record.get('description')

                text = f"{main_label}: {name_val}"
                if ticker_val: text += f" ({ticker_val})"
                if desc_val: text += f" - {str(desc_val)[:200]}"

                vec = self._get_vector(text)
                session.run(
                    "MATCH (n) WHERE elementId(n)=$id CALL db.create.setNodeVectorProperty(n, 'embedding', $v)",
                    id=record['id'], v=vec)

            print("3. Создание индекса...")
            session.run(
                f"CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS FOR (n:Searchable) ON (n.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {self.vector_dim}, `vector.similarity_function`: 'cosine'}}}}")
            print("Индекс готов.")

    def verify_match_llm(self, text_mention, db_name, db_label):
        """Быстрая проверка: Это одно и то же?"""
        prompt = f"""
        Matching Check.
        Text in news: "{text_mention}"
        Database Node: "{db_name}" (Type: {db_label})

        Are these the SAME entity?
        - "Fed" vs "Federal Realty" -> NO
        - "Netflix" vs "Netflix Inc." -> YES
        - "BlackRock" vs "BlackRock Inc." -> YES

        Return JSON: {{ "match": true/false }}
        """
        try:
            completion = self.client.chat.completions.create(model='qwen/qwen-2.5-7b-instruct',
                                                             messages=[{'role': 'user', 'content': prompt}],
                                                             temperature=0.0)
            res = json.loads(completion.choices[0].message.content.replace("```json", "").replace("```", "").strip())
            return res.get('match', False)
        except:
            return False

    def find_node_by_vector(self, text_query):
        if not text_query or len(text_query) < 2: return None
        vector = self._get_vector(text_query)

        query = """
        CALL db.index.vector.queryNodes('entity_vector_index', 5, $vec)
        YIELD node, score WHERE score >= $thresh
        RETURN node, labels(node) as lbls, score
        """
        with self.driver.session() as session:
            results = list(session.run(query, vec=vector, thresh=THRESHOLD_LLM_CHECK))

            if not results: return None

            candidates = []
            for res in results:
                node = res['node']
                name = node.get('name', '')
                lbls = [l for l in res['lbls'] if l != 'Searchable']
                label = lbls[0] if lbls else "Unknown"

                text_score = fuzz.token_sort_ratio(text_query.lower(), name.lower())

                candidates.append({
                    "node": node, "label": label, "name": name,
                    "v_score": res['score'], "t_score": text_score
                })

            if not candidates: return None


            exact_matches = [c for c in candidates if c['t_score'] > 90]
            if exact_matches:
                companies = [c for c in exact_matches if c['label'] == 'Company']
                best = companies[0] if companies else exact_matches[0]
                return self._format_node(best)

            def sort_key(c):
                bonus = 0.05 if c['label'] == 'Company' else 0
                return c['v_score'] + bonus

            best = max(candidates, key=sort_key)

            if best['v_score'] > THRESHOLD_AUTO_ACCEPT:
                return self._format_node(best)

            if best['t_score'] > 50:
                if self.verify_match_llm(text_query, best['name'], best['label']):
                    return self._format_node(best)

            return None

    def _format_node(self, c):
        n = c['node']
        key = "ticker" if c['label'] == 'Company' else "name"
        return {"id_val": n.get(key) or n.get("name"), "key_field": key, "label": c['label'], "name": c['name'],
                "score": c['v_score']}

    def analyze_news_with_llm(self, headline, body):
        text = f"HEADLINE: {headline}\nCONTENT: {str(body)[:800]}"
        prompt = f"""
        Analyze financial news. TEXT: {text}

        TASK:
        1. Extract MAIN entities: Companies, Funds, Industries, Cities, Countries.
           - IGNORE People/Analysts.
        2. Determine interaction between TWO main entities.
        3. Sentiment (-1.0 to 1.0).

        Return JSON:
        {{
            "entities": ["Name1", "Name2"],
            "interaction": {{
                "source": "Name1", "target": "Name2",
                "relation": "PARTNER_WITH" (or "COMPETES_WITH" only),
                "summary": "Short explanation"
            }}, 
            "sentiment": 0.5
        }}
        """
        try:
            completion = self.client.chat.completions.create(model='qwen/qwen-2.5-7b-instruct',
                                                             messages=[{'role': 'user', 'content': prompt}],
                                                             temperature=0.0)
            content = completion.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            if "{" in content: content = content[content.find("{"):content.rfind("}") + 1]
            return json.loads(content)
        except:
            return None

    def process_csv(self, csv_path):
        print(f"\nОбработка новостей: {csv_path}")
        if not os.path.exists(csv_path): return

        df = pd.read_csv(csv_path)
        stats = {"news_nodes": 0, "relations_updated": 0, "skipped": 0}

        with self.driver.session() as session:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                headline = str(row.get('headline', row.get('Headlines', '')))
                body = str(row.get('text', row.get('Description', '')))
                date_str = str(row.get('Time', row.get('date', datetime.date.today())))

                analysis = self.analyze_news_with_llm(headline, body)
                if not analysis: stats['skipped'] += 1; continue

                found_entities = []
                seen_ids = set()
                raw_entities = analysis.get('entities', [])
                if isinstance(raw_entities, str): raw_entities = [raw_entities]

                raw_entities = [x for x in raw_entities if x]

                matched_names = []

                for raw_name in raw_entities:
                    match = self.find_node_by_vector(raw_name)
                    if match:
                        uid = f"{match['label']}_{match['id_val']}"
                        if uid not in seen_ids:
                            found_entities.append(match)
                            seen_ids.add(uid)
                            matched_names.append(f"{match['name']} ({match['label']})")

                if not found_entities: stats['skipped'] += 1; continue

                tqdm.write(f"News: '{headline[:35]}...' -> Found: {', '.join(matched_names)}")

                try:
                    iso_date = pd.to_datetime(date_str).strftime("%Y-%m-%d")
                except:
                    iso_date = datetime.date.today().isoformat()

                for ent in found_entities:
                    session.run(f"""
                        MATCH (e:{ent['label']} {{ {ent['key_field']}: $eid }})
                        MERGE (n:News {{headline: $hl, date: date($dt)}})
                        SET n.sentiment = $sent, n.source = 'Reuters'
                        MERGE (n)-[:MENTIONS]->(e)
                    """, eid=ent['id_val'], hl=headline, dt=iso_date, sent=analysis.get('sentiment', 0))
                    stats['news_nodes'] += 1

                interaction = analysis.get('interaction')
                if interaction and isinstance(interaction, dict) and len(found_entities) >= 2:
                    src_match = self.find_node_by_vector(interaction.get('source'))
                    trg_match = self.find_node_by_vector(interaction.get('target'))

                    if src_match and trg_match and src_match['id_val'] != trg_match['id_val']:
                        allowed = ['Company', 'Fund', 'Organization']
                        if src_match['label'] not in allowed or trg_match['label'] not in allowed:
                            continue

                        raw_rel = interaction.get('relation', 'PARTNER_WITH').upper()
                        rel_type = "COMPETES_WITH" if "COMPETE" in raw_rel or "DISPUTE" in raw_rel else "PARTNER_WITH"
                        summary = interaction.get('summary', headline)

                        fact_json = json.dumps({
                            "topic": "News Event", "specific_evidence": summary,
                            "headline": headline, "source": "Reuters News Analysis",
                            "source_script": "script_5_reuters", "date_recorded": iso_date
                        }, ensure_ascii=False)

                        query = f"""
                        MATCH (a:{src_match['label']} {{ {src_match['key_field']}: $id1 }})
                        MATCH (b:{trg_match['label']} {{ {trg_match['key_field']}: $id2 }})
                        MERGE (a)-[r:{rel_type}]->(b)
                        ON CREATE SET r.created_at = date($dt), r.updated_at = date($dt), r.source = 'Reuters News Analysis', r.evidence_log = [$fact]
                        ON MATCH SET r.updated_at = date($dt), r.evidence_log = CASE WHEN r.evidence_log IS NULL THEN [$fact] ELSE r.evidence_log + $fact END
                        """
                        session.run(query, id1=src_match['id_val'], id2=trg_match['id_val'], dt=iso_date,
                                    fact=fact_json)
                        tqdm.write(f"   LINK: {src_match['name']} -[{rel_type}]-> {trg_match['name']}")
                        stats['relations_updated'] += 1

        print(f"\nОбработка завершена. Статистика: {stats}")


if __name__ == "__main__":
    processor = NewsProcessor()
    processor.setup_search_index()
    processor.process_csv('../data/classified_reuters_news_mapped.csv')
    processor.close()