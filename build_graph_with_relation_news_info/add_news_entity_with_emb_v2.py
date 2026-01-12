import os
import json
import uuid
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class FinancialGraphPipeline:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME") or "neo4j"
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        self.polza_key = os.getenv("POLZA_API_KEY")
        self.client = OpenAI(api_key=self.polza_key, base_url="https://api.polza.ai/api/v1")

        model_path = r"C:\sp500_models\all-MiniLM-L6-v2"
        print(f"⏳ Загрузка модели из {model_path}...")
        try:
            self.embedder = SentenceTransformer(model_path)
            self.vector_dim = 384
            print("✅ Модель загружена успешно.")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            exit()

    def close(self):
        self.driver.close()

    def _get_vector(self, text):
        if not text: return None
        clean_text = str(text).replace("\n", " ").strip()
        return self.embedder.encode(clean_text).tolist()

    def setup_knowledge_base(self):
        print("\n🏗️  Генерация насыщенных векторов (Rich Embeddings)...")
        target_labels = ["Company", "Person", "Fund", "City", "Country", "Sector", "Subsector"]

        with self.driver.session() as session:
            session.run("DROP INDEX unified_entity_index IF EXISTS")
            session.run("MATCH (n:Searchable) REMOVE n.embedding, n:Searchable")

            query_fetch = """
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
            RETURN elementId(n) as id, labels(n) as labels, n as props
            """
            nodes = list(session.run(query_fetch, labels=target_labels))
            print(f"Найдено {len(nodes)} узлов для индексации.")

            for record in tqdm(nodes, desc="Vectorizing Nodes"):
                p = record['props']
                main_label = [l for l in record['labels'] if l != 'Searchable'][0]

                if main_label == 'Company':
                    rich_text = f"Company: {p.get('name')} ({p.get('ticker')}). Info: {p.get('description', '')[:400]}"
                elif main_label == 'Person':
                    rich_text = f"Person: {p.get('name')}. Position: {p.get('title', 'Expert')} in {p.get('company', 'S&P 500 context')}"
                elif main_label == 'Fund':
                    rich_text = f"Financial Fund: {p.get('name')}. Institutional holder."
                else:
                    rich_text = f"{main_label}: {p.get('name')}"

                vector = self._get_vector(rich_text)

                session.run("""
                    MATCH (n) WHERE elementId(n) = $id
                    SET n:Searchable
                    WITH n
                    CALL db.create.setNodeVectorProperty(n, 'embedding', $vec)
                """, id=record['id'], vec=vector)

            session.run(f"""
                CREATE VECTOR INDEX unified_entity_index IF NOT EXISTS
                FOR (n:Searchable) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {self.vector_dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)
            print("✅ База знаний готова.")

    def resolve_entity(self, name_query, threshold=0.84):
        if not name_query or len(str(name_query)) < 2: return None
        name_query = str(name_query).strip()

        with self.driver.session() as session:
            # 1. Точный поиск (Исправлено: добавлено AS node)
            exact_q = """
            MATCH (n:Searchable)
            WHERE toLower(n.ticker) = toLower($q) OR toLower(n.name) = toLower($q)
            RETURN n AS node, labels(n) as lbls LIMIT 1
            """
            res = session.run(exact_q, q=name_query).single()
            if res:
                return self._format_resolve_data(res, 1.0)

            # 2. Векторный поиск
            vec = self._get_vector(name_query)
            vec_q = """
            CALL db.index.vector.queryNodes('unified_entity_index', 1, $vec)
            YIELD node, score WHERE score >= $thresh
            RETURN node, labels(node) as lbls, score
            """
            res = session.run(vec_q, vec=vec, thresh=threshold).single()
            if res:
                return self._format_resolve_data(res, res['score'])
        return None

    def _format_resolve_data(self, record, score):
        node = record['node']
        label = [l for l in record['lbls'] if l != 'Searchable'][0]
        key = "ticker" if label == 'Company' else "name"
        return {
            "id": node.get(key),
            "type": label,
            "name": node.get('name'),
            "key_field": key,
            "score": score
        }

    def process_csv(self, file_path):
        print(f"\n🚀 Запуск обработки новостей: {file_path}")
        if not os.path.exists(file_path): return

        df = pd.read_csv(file_path)
        stats = {"added": 0, "skipped": 0}

        with self.driver.session() as session:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing News"):
                headline = str(row.get('headline', row.get('Headlines', '')))
                desc = str(row.get('text', row.get('Description', '')))

                # Анализ LLM
                analysis = self._analyze_news_llm(headline, desc)
                if not analysis:
                    stats['skipped'] += 1;
                    continue

                # Разрешение сущностей
                found_entities = []
                seen_uids = set()

                raw_names = analysis.get('entities', [])
                for name in raw_names:
                    match = self.resolve_entity(name)
                    if match:
                        uid = f"{match['type']}_{match['id']}"
                        if uid not in seen_uids:
                            found_entities.append(match)
                            seen_uids.add(uid)
                            # ВЫВОД: ЧТО НАШЛИ
                            tqdm.write(
                                f"   ✅ '{name}' -> {match['name']} ({match['type']}) [score: {match['score']:.2f}]")

                if not found_entities:
                    # tqdm.write(f"   ⚠️ Скип: '{headline[:50]}...' (нет совпадений)")
                    stats['skipped'] += 1;
                    continue

                # Дата
                date_str = str(row.get('Time', row.get('date', datetime.today())))
                try:
                    iso_date = datetime.strptime(date_str.strip(), "%b %d %Y").strftime("%Y-%m-%d")
                except:
                    iso_date = datetime.today().strftime("%Y-%m-%d")

                sentiment = analysis.get('sentiment', 'NEUTRAL')
                interaction = analysis.get('interaction')

                # 1. Создаем узел Новости
                session.run("""
                    MERGE (n:News {headline: $h, date: date($d)})
                    SET n.sentiment = $s
                """, h=headline, d=iso_date, s=sentiment)

                # 2. Связываем (MENTIONS)
                for ent in found_entities:
                    session.run(f"""
                        MATCH (n:News {{headline: $h, date: date($d)}}), (e:{ent['type']} {{ {ent['key_field']}: $eid }})
                        MERGE (n)-[r:MENTIONS]->(e)
                        SET r.match_score = $score
                    """, h=headline, d=iso_date, eid=ent['id'], score=ent['score'])

                # 3. Взаимодействие (Interaction)
                if interaction and len(found_entities) >= 2:
                    src = self.resolve_entity(interaction.get('source')) or found_entities[0]
                    trg = self.resolve_entity(interaction.get('target')) or found_entities[1]

                    if src and trg and src['id'] != trg['id']:
                        rel = str(interaction.get('relation', 'RELATED_TO')).upper().replace(" ", "_")

                        # ВЫВОД: СВЯЗЬ
                        tqdm.write(f"   🔗 LINK: {src['name']} -[{rel}]-> {trg['name']}")

                        session.run(f"""
                            MATCH (a:{src['type']} {{ {src['key_field']}: $id1 }}), 
                                  (b:{trg['type']} {{ {trg['key_field']}: $id2 }})
                            MERGE (a)-[r:{rel}]->(b)
                            ON CREATE SET r.created_at = date($d), r.news_history = [$h]
                            ON MATCH SET r.news_history = r.news_history + $h
                        """, id1=src['id'], id2=trg['id'], d=iso_date, h=headline)

                stats['added'] += 1

        print(f"\n🏁 Итоги: {stats}")

    def _analyze_news_llm(self, headline, description):
        prompt = f"""
        Extract entities and interaction from financial news.
        Text: {headline}. {description}
        Return JSON:
        {{
            "entities": ["Name1", "Name2"],
            "interaction": {{"source": "Name1", "target": "Name2", "relation": "PARTNERSHIP"}},
            "sentiment": "POSITIVE"
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}], temperature=0.0
            )
            content = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            return json.loads(content)
        except:
            return None


if __name__ == "__main__":
    pipeline = FinancialGraphPipeline()
    pipeline.setup_knowledge_base() # Выполни один раз
    pipeline.process_csv('../data/classified_reuters_news_mapped.csv')
    pipeline.close()