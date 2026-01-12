import os
import json
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(find_dotenv())

class NewsGraphPipeline:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"
        password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_auth = (user, password)

        self.polza_key = os.getenv("POLZA_API_KEY")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        self.client = OpenAI(api_key=self.polza_key, base_url="https://api.polza.ai/api/v1")

        model_path = r"C:\sp500_models\all-MiniLM-L6-v2"

        print(f"⏳ Загрузка модели из {model_path}...")
        try:
            self.embedder = SentenceTransformer(model_path)
            self.vector_dim = 384
            print("✅ Модель загружена успешно (CPU).")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("Проверь, что файлы лежат в C:\\sp500_models\\all-MiniLM-L6-v2")
            exit()

    def close(self):
        self.driver.close()

    def _get_vector(self, text):
        """Превращает текст в список float (используя SentenceTransformer)"""
        if not text: return None
        return self.embedder.encode(text).tolist()

    def setup_knowledge_base_vectors(self):
        print("\n🏗️ === ПОДГОТОВКА ВЕКТОРНОЙ БАЗЫ (SETUP) ===")
        target_labels = ["Company", "Person", "Fund", "City", "Country"]

        # Удаление старых данных
        with self.driver.session() as session:
            print("1. Очистка старых индексов и векторов...")
            session.run("DROP INDEX unified_entity_index IF EXISTS")
            session.run("MATCH (n:Searchable) REMOVE n.embedding")
            session.run("MATCH (n:Searchable) REMOVE n:Searchable")

        # Выборка и векторизация
        query_fetch = f"""
        MATCH (n)
        WHERE any(label in labels(n) WHERE label IN $targets)
        RETURN elementId(n) as id, labels(n) as labels, n.name as name, 
               n.ticker as ticker, n.description as desc
        """

        with self.driver.session() as session:
            print("2. Добавляем метку :Searchable...")
            for label in target_labels:
                session.run(f"MATCH (n:{label}) SET n:Searchable")

            result = session.run(query_fetch, targets=target_labels)
            nodes = list(result)
            print(f"   Найдено узлов: {len(nodes)}")

            print("3. Генерация векторов...")
            for record in tqdm(nodes, desc="Vectorizing"):
                main_label = [l for l in record['labels'] if l != 'Searchable'][0]
                text = f"{main_label}: {record['name']}"

                if record['ticker']:
                    text += f" ({record['ticker']})"
                if record['desc']:
                    text += f". Info: {str(record['desc'])[:300]}"

                vector = self._get_vector(text)

                if vector:
                    session.run("""
                        MATCH (n) WHERE elementId(n) = $id
                        CALL db.create.setNodeVectorProperty(n, 'embedding', $vec)
                    """, id=record['id'], vec=vector)

            print("4. Создание индекса...")
            try:
                session.run(f"""
                    CREATE VECTOR INDEX unified_entity_index IF NOT EXISTS
                    FOR (n:Searchable) ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {self.vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Index info: {e}")

        print("✅ База готова.")

    def resolve_entity(self, name_query, threshold=0.70):
        if not name_query or len(name_query) < 2: return None

        vector = self._get_vector(name_query)
        if not vector: return None

        query = """
        CALL db.index.vector.queryNodes('unified_entity_index', 1, $vec)
        YIELD node, score
        WHERE score >= $thresh
        RETURN node, labels(node) as lbls, score
        """

        with self.driver.session() as session:
            result = session.run(query, vec=vector, thresh=threshold).single()

            if result:
                node = result['node']
                labels = result['lbls']
                real_type = [l for l in labels if l != 'Searchable'][0]
                key_field = "ticker" if real_type == 'Company' else "name"
                entity_id = node.get(key_field)

                return {
                    "id": entity_id,
                    "type": real_type,
                    "key_field": key_field,
                    "name": node.get('name'),
                    "score": result['score']
                }
        return None

    def analyze_news_llm(self, headline, description):
        text = f"Headline: {headline}\nDescription: {description}"
        prompt = f"""
        Analyze this news item.
        TEXT: {text}

        TASK:
        1. Extract MAIN entities (Companies, Funds, People, Locations).
        2. Determine interaction.

        Return JSON structure:
        {{
            "entities": ["Entity1", "Entity2"],
            "interaction": {{
                "source": "Entity1",
                "target": "Entity2",
                "relation": "PARTNERSHIP",
                "summary": "Short summary"
            }},
            "sentiment": "POSITIVE" / "NEGATIVE" / "NEUTRAL"
        }}
        """
        try:
            completion = self.client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}], temperature=0.0
            )
            content = completion.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            if isinstance(parsed, list): return parsed[0] if len(parsed) > 0 else None
            return parsed
        except Exception:
            return None

    def process_csv(self, file_path):
        print(f"\n🚀 Запуск обработки новостей: {file_path}")
        if not os.path.exists(file_path):
            print("❌ Файл не найден"); return

        df = pd.read_csv(file_path)
        stats = {"edges": 0, "news_nodes": 0, "skipped": 0}

        with self.driver.session() as session:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                headline = str(row.get('Headlines', row.get('headline', '')))
                desc = str(row.get('Description', row.get('text', '')))
                date_str = str(row.get('Time', datetime.now()))

                # LLM
                analysis = self.analyze_news_llm(headline, desc)
                if not analysis:
                    stats['skipped'] += 1; continue

                raw_names = analysis.get('entities', [])
                if isinstance(raw_names, str): raw_names = [raw_names]

                found_entities = []
                seen_uids = set()

                for name in raw_names:
                    match = self.resolve_entity(name, threshold=0.70)
                    if match:
                        uid = f"{match['type']}_{match['id']}"
                        if uid not in seen_uids:
                            found_entities.append(match); seen_uids.add(uid)
                            tqdm.write(f"   ✅ '{name}' -> {match['name']} ({match['type']})")

                if not found_entities:
                    # tqdm.write("   ⚠️ No match.")
                    stats['skipped'] += 1; continue

                try: iso_date = datetime.strptime(str(date_str).strip(), "%b %d %Y").strftime("%Y-%m-%d")
                except: iso_date = datetime.today().strftime("%Y-%m-%d")

                sentiment = analysis.get('sentiment', 'NEUTRAL')
                interaction = analysis.get('interaction')
                link_created = False

                # Сценарий: Связь
                if interaction and isinstance(interaction, dict) and len(found_entities) >= 2:
                    src = self.resolve_entity(interaction.get('source'), 0.65) or found_entities[0]
                    trg = self.resolve_entity(interaction.get('target'), 0.65) or (found_entities[1] if len(found_entities)>1 else None)

                    if src and trg and src['id'] != trg['id']:
                        rel = "RELATED_TO"
                        rel_raw = str(interaction.get('relation')).upper()
                        if "PARTNER" in rel_raw: rel = "PARTNER_WITH"
                        elif "DISPUTE" in rel_raw: rel = "IN_DISPUTE_WITH"
                        elif "INVEST" in rel_raw: rel = "INVESTED_IN"
                        elif trg['type'] in ['City', 'Country']: rel = "AFFECTS_REGION"

                        tqdm.write(f"   🔗 LINK: {src['name']} -[{rel}]-> {trg['name']}")

                        session.run(f"""
                            MATCH (a:{src['type']} {{ {src['key_field']}: $id1 }}), (b:{trg['type']} {{ {trg['key_field']}: $id2 }})
                            MERGE (a)-[r:{rel}]->(b) 
                            ON CREATE SET r.created_at = date($d), r.news_history = [$h]
                            ON MATCH SET r.news_history = r.news_history + $h
                        """, id1=src['id'], id2=trg['id'], d=iso_date, h=headline)
                        link_created = True; stats['edges'] += 1

                # Сценарий: Узел
                if not link_created:
                    for ent in found_entities:
                        session.run(f"""
                            MATCH (e:{ent['type']} {{ {ent['key_field']}: $id }})
                            MERGE (n:News {{headline: $h, date: date($d)}})
                            SET n.sentiment = $sent
                            MERGE (n)-[:MENTIONS]->(e)
                        """, id=ent['id'], h=headline, d=iso_date, sent=sentiment)
                    stats['news_nodes'] += 1

        print(f"\n🏁 Итоги: {stats}")

if __name__ == "__main__":
    rag = NewsGraphPipeline()

    rag.setup_knowledge_base_vectors()

    rag.process_csv('../data/classified_reuters_news_mapped.csv')

    rag.close()