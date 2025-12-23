import os
import json
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import ollama

load_dotenv(find_dotenv())


class NewsGraphPipeline:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        self.polza_key = os.getenv("POLZA_API_KEY")

        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        self.client = OpenAI(api_key=self.polza_key, base_url="https://api.polza.ai/api/v1")

        self.embedding_model = "nomic-embed-text"
        self.vector_dim = 768
        print(f"✅ Используем локальную Ollama: {self.embedding_model}")

    def close(self):
        self.driver.close()

    def _get_vector(self, text):
        """Превращает текст в список float через Ollama"""
        if not text: return None
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Ошибка Ollama: {e}")
            return None

    def setup_knowledge_base_vectors(self):
        """
        ПРОЦЕДУРА ПОДГОТОВКИ:
        1. Находит все узлы (Company, Fund, Person, Location).
        2. Вешает на них метку :Searchable.
        3. Генерирует вектор по описанию и сохраняет в узел.
        4. Создает единый индекс.
        """
        print("\n🏗️ === ПОДГОТОВКА ВЕКТОРНОЙ БАЗЫ ===")

        target_labels = ["Company", "Person", "Fund", "City", "Country"]

        query_fetch = f"""
        MATCH (n)
        WHERE any(label in labels(n) WHERE label IN $targets)
        RETURN elementId(n) as id, labels(n) as labels, n.name as name, 
               n.ticker as ticker, n.description as desc
        """

        with self.driver.session() as session:
            print("1. Добавляем метку :Searchable всем сущностям...")
            for label in target_labels:
                session.run(f"MATCH (n:{label}) SET n:Searchable")

            result = session.run(query_fetch, targets=target_labels)
            nodes = list(result)
            print(f"2. Генерация векторов для {len(nodes)} узлов (через Ollama)...")

            for record in tqdm(nodes, desc="Vectorizing"):
                main_label = [l for l in record['labels'] if l != 'Searchable'][0]
                text = f"{main_label}: {record['name']}"

                if record['ticker']:
                    text += f" ({record['ticker']})"
                if record['desc']:
                    text += f". Info: {str(record['desc'])[:200]}"

                vector = self._get_vector(text)

                if vector:
                    session.run("""
                        MATCH (n) WHERE elementId(n) = $id
                        CALL db.create.setNodeVectorProperty(n, 'embedding', $vec)
                    """, id=record['id'], vec=vector)

            print("3. Пересоздание индекса 'unified_entity_index'...")
            try:
                session.run("DROP INDEX unified_entity_index IF EXISTS")

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
                print(f"⚠️ Index creation info: {e}")

        print("✅ База готова к семантическому поиску.")

    def resolve_entity(self, name_query, threshold=0.60):
        """
        Ищет ближайший узел в базе по смыслу.
        """
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

        TEXT:
        {text}

        TASK:
        1. Extract MAIN entities (Companies, Funds, People, Locations).
        2. Determine if there is a direct interaction between two entities.

        Return JSON structure:
        {{
            "entities": ["Entity1", "Entity2"],
            "interaction": {{
                "source": "Entity1",
                "target": "Entity2",
                "relation": "PARTNERSHIP",
                "summary": "Max 10 words summary"
            }},
            "sentiment": "POSITIVE" / "NEGATIVE" / "NEUTRAL"
        }}

        Rules:
        - If no interaction, set "interaction": null.
        - JSON ONLY.
        """
        try:
            completion = self.client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0
            )
            content = completion.choices[0].message.content

            clean_content = content.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(clean_content)
            if isinstance(parsed, list):
                return parsed[0] if len(parsed) > 0 else None

            return parsed

        except Exception as e:
            tqdm.write(f"🛑 API ERROR: {str(e)}")
            return None


    def process_csv(self, file_path):
        print(f"\n🚀 Запуск обработки новостей: {file_path}")

        if not os.path.exists(file_path):
            print("❌ Файл не найден")
            return

        df = pd.read_csv(file_path)

        stats = {"edges": 0, "news_nodes": 0, "skipped": 0}

        with self.driver.session() as session:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):

                headline = str(row.get('headline', ''))
                desc = str(row.get('text', ''))
                date_str = str(row.get('Time', datetime.now()))

                tqdm.write(f"\n📄 News [{i + 1}]: {headline[:100]}...")

                analysis = self.analyze_news_llm(headline, desc)
                if not analysis:
                    tqdm.write("   ⚠️ LLM вернула пустой результат.")
                    stats['skipped'] += 1
                    continue

                raw_names = analysis.get('entities', [])
                if isinstance(raw_names, str): raw_names = [raw_names]

                tqdm.write(f"   🧠 LLM выделила: {raw_names}")

                found_entities = []
                seen_uids = set()

                for name in raw_names:
                    match = self.resolve_entity(name, threshold=0.62)
                    if match:
                        uid = f"{match['type']}_{match['id']}"
                        if uid not in seen_uids:
                            found_entities.append(match)
                            seen_uids.add(uid)
                            tqdm.write(
                                f"      ✅ Match: '{name}' -> {match['name']} ({match['type']}) [Score: {match['score']:.2f}]")
                    else:
                        pass
                        # tqdm.write(f"      ❌ No match: '{name}'")

                if not found_entities:
                    tqdm.write("   ⚠️ В базе не найдено ни одной сущности из списка.")
                    stats['skipped'] += 1
                    continue

                try:
                    iso_date = datetime.strptime(str(date_str).strip(), "%b %d %Y").strftime("%Y-%m-%d")
                except:
                    iso_date = datetime.today().strftime("%Y-%m-%d")

                sentiment = analysis.get('sentiment', 'NEUTRAL')
                interaction = analysis.get('interaction')
                log_entry = f"[{iso_date}] {sentiment}: {headline}"

                link_created = False

                # Создаем связь
                if interaction and isinstance(interaction, dict) and len(found_entities) >= 2:
                    src_obj = self.resolve_entity(interaction.get('source'), 0.60)
                    trg_obj = self.resolve_entity(interaction.get('target'), 0.60)

                    if not src_obj: src_obj = found_entities[0]
                    if not trg_obj and len(found_entities) > 1: trg_obj = found_entities[1]

                    if src_obj and trg_obj and src_obj['id'] != trg_obj['id']:
                        rel_type = interaction.get('relation', 'RELATED_TO').upper().replace(" ", "_")
                        summary = interaction.get('summary', headline)

                        cypher_rel = "RELATED_TO"
                        if "PARTNER" in rel_type:
                            cypher_rel = "PARTNER_WITH"
                        elif "DISPUTE" in rel_type:
                            cypher_rel = "IN_DISPUTE_WITH"
                        elif "INVEST" in rel_type:
                            cypher_rel = "INVESTED_IN"
                        elif trg_obj['type'] in ['City', 'Country']:
                            cypher_rel = "AFFECTS_REGION"

                        tqdm.write(f"   🔗 LINK: {src_obj['name']} -[{cypher_rel}]-> {trg_obj['name']}")

                        query_edge = f"""
                        MATCH (a:{src_obj['type']} {{ {src_obj['key_field']}: $id1 }})
                        MATCH (b:{trg_obj['type']} {{ {trg_obj['key_field']}: $id2 }})
                        MERGE (a)-[r:{cypher_rel}]->(b)
                        ON CREATE SET r.created_at = date($date), r.news_history = [$log], r.last_summary = $sum
                        ON MATCH SET r.news_history = r.news_history + $log, r.last_updated = date($date)
                        """
                        session.run(query_edge, id1=src_obj['id'], id2=trg_obj['id'],
                                    date=iso_date, log=log_entry, sum=summary)

                        stats['edges'] += 1
                        link_created = True

                # Вешаем новость на узлы
                if not link_created:
                    names_list = [e['name'] for e in found_entities]
                    tqdm.write(f"   📌 ATTACH: Новость прикреплена к: {names_list}")

                    for ent in found_entities:
                        query_news = f"""
                        MATCH (e:{ent['type']} {{ {ent['key_field']}: $id }})
                        MERGE (n:News {{headline: $headline, date: date($date)}})
                        SET n.sentiment = $sent
                        MERGE (n)-[:MENTIONS]->(e)
                        """
                        session.run(query_news, id=ent['id'], headline=headline, date=iso_date, sent=sentiment)
                    stats['news_nodes'] += 1

        print(
            f"\n🏁 Итоги: Связей создано: {stats['edges']}, Новостей-узлов: {stats['news_nodes']}, Пропущено: {stats['skipped']}")

if __name__ == "__main__":
    rag = NewsGraphPipeline()

    rag.setup_knowledge_base_vectors()

    rag.process_csv('../data/classified_reuters_news_mapped.csv')

    rag.close()