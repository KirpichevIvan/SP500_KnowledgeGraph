import os
import json
import numpy as np
import pandas as pd
import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import time

load_dotenv(find_dotenv())

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MODEL_PATH = r"C:\sp500_models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'all-MiniLM-L6-v2'

PRODUCT_SIMILARITY_THRESHOLD = 0.75
COMPANY_DESC_THRESHOLD = 0.82

client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class GraphEnricher:
    def __init__(self):
        print("Инициализация модели эмбеддингов...")
        try:
            self.model = SentenceTransformer(MODEL_PATH)
            print(f"Модель {MODEL_PATH} загружена.")
        except Exception as e:
            print(f"Ошибка загрузки локальной модели: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_all_products(self):
        """Достаем все продукты и их владельцев из Neo4j"""
        print("Скачиваем продукты из графа...")
        query = """
        MATCH (c:Company)-[:PRODUCES]->(p:Product)
        RETURN c.ticker as ticker, c.name as company_name, p.name as product_name, elementId(p) as p_id
        """
        with driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
        df = pd.DataFrame(data)
        print(f"   Найдено {len(df)} продуктов.")
        return df

    def get_all_companies(self):
        """Достаем компании, их описания и индустрии"""
        print("Скачиваем компании из графа...")
        query = """
        MATCH (c:Company)
        OPTIONAL MATCH (c)-[:OPERATES_IN_INDUSTRY]->(i:Industry)
        RETURN c.ticker as ticker, c.name as name, c.description as description, i.name as industry
        """
        with driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
        df = pd.DataFrame(data)
        df['description'] = df['description'].fillna("")
        df['industry'] = df['industry'].fillna("Unknown")

        df['embed_text'] = df['industry'] + ": " + df['description']
        print(f"   Найдено {len(df)} компаний.")
        return df

    def analyze_product_overlaps(self, products_df):
        """
        Кластеризация продуктов через попарное сравнение.
        Если продукты похожи, но компании разные -> Кандидаты в конкуренты.
        """
        print("Векторизация продуктов...")
        if products_df.empty:
            return {}

        embeddings = self.model.encode(products_df['product_name'].tolist(), convert_to_tensor=True,
                                       show_progress_bar=True)

        print("Поиск семантических пересечений продуктов...")
        cosine_scores = util.cos_sim(embeddings, embeddings)

        potential_links = {}  # Key: (TickerA, TickerB), Value: List of reasons

        rows, cols = cosine_scores.shape
        scores_np = cosine_scores.cpu().numpy()

        count = 0
        for i in range(rows):
            for j in range(i + 1, cols):
                score = scores_np[i][j]

                if score > PRODUCT_SIMILARITY_THRESHOLD:
                    comp_a = products_df.iloc[i]['ticker']
                    comp_b = products_df.iloc[j]['ticker']

                    if comp_a != comp_b:
                        prod_a = products_df.iloc[i]['product_name']
                        prod_b = products_df.iloc[j]['product_name']

                        key = tuple(sorted((comp_a, comp_b)))

                        if key not in potential_links:
                            potential_links[key] = []

                        potential_links[key].append(f"Similar products ({score:.2f}): '{prod_a}' vs '{prod_b}'")
                        count += 1

        print(f"   Найдено {count} пересечений по продуктам.")
        return potential_links

    def analyze_company_similarity(self, companies_df):
        """
        Анализ схожести описаний компаний внутри одной индустрии.
        """
        print("Векторизация описаний компаний...")
        embeddings = self.model.encode(companies_df['embed_text'].tolist(), convert_to_tensor=True,
                                       show_progress_bar=True)

        cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

        high_level_links = {}
        rows = cosine_scores.shape[0]

        for i in range(rows):
            for j in range(i + 1, rows):
                score = cosine_scores[i][j]
                if score > COMPANY_DESC_THRESHOLD:
                    row_a = companies_df.iloc[i]
                    row_b = companies_df.iloc[j]

                    key = tuple(sorted((row_a['ticker'], row_b['ticker'])))
                    if key not in high_level_links:
                        high_level_links[key] = []

                    high_level_links[key].append(
                        f"High business description similarity ({score:.2f}) in sector {row_a['industry']}")

        return high_level_links

    def verify_with_llm(self, ticker_a, ticker_b, evidence_list):
        evidence_str = "\n".join(evidence_list[:10])

        prompt = f"""
        Analyze the relationship between {ticker_a} and {ticker_b}.

        [EVIDENCE LOG]:
        {evidence_str}

        CRITICAL RULES FOR CLASSIFICATION:
        1. "COMPETITOR": If they sell similar products or operate in the same sector (e.g. both sell Furniture). Overlap = Competition.
        2. "PARTNER": Only if there is specific evidence of a deal, supply chain relationship, integration, or joint venture.
        3. "RELATED_ENTITY": If the tickers belong to the same parent company (e.g. GOOG and GOOGL, FOX and FOXA).

        Task:
        Identify DISTINCT areas of relationship.

        Return JSON ONLY:
        {{
            "relationships": [
                {{
                    "type": "COMPETITOR" | "PARTNER" | "RELATED_ENTITY",
                    "topic": "Specific Market",
                    "reason": "Why?",
                    "evidence_used": "Quote from evidence" 
                }}
            ]
        }}
        """

        try:
            completion = client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0
            )
            content = completion.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            if "{" in content:
                content = content[content.find("{"):content.rfind("}") + 1]

            data = json.loads(content)
            return data.get("relationships", [])

        except Exception as e:
            print(f"      LLM Error: {e}")
            return []

    def save_relationship(self, ticker_a, ticker_b, rel_data):
        rel_type = rel_data.get('type', 'NONE').upper()

        relations_to_process = []

        if "COMPETE" in rel_type or "COMPETITOR" in rel_type:
            relations_to_process = ["COMPETES_WITH"]
        elif "PARTNER" in rel_type:
            relations_to_process = ["PARTNER_WITH"]
        elif "RELATED" in rel_type or "SAME" in rel_type:
            print(f"         [i] Skipped: Same entity/Related ({rel_type})")
            return
        else:
            print(f"         Unknown Relation Type from LLM: '{rel_type}' - Skipped.")
            return

        fact_object = {
            "topic": rel_data.get('topic', 'General'),
            "reason": rel_data.get('reason', ''),
            "specific_evidence": rel_data.get('evidence_used', 'Vector Inference'),
            "source": "AI Inference",
            "date_recorded": datetime.date.today().isoformat()
        }

        fact_json = json.dumps(fact_object, ensure_ascii=False)

        query = """
                MATCH (a:Company {ticker: $t1})
                MATCH (b:Company {ticker: $t2})

                // Находим или создаем связь
                MERGE (a)-[r:REL_TYPE]->(b)

                // Если связи не было (создал Скрипт 3)
                ON CREATE SET 
                    r.created_at = date(),
                    r.updated_at = date(),
                    r.source = 'Inference + Others',
                    r.evidence_log = [$new_fact]

                // Если связь уже была (создал Скрипт 2 или прошлый запуск Скрипта 3)
                ON MATCH SET 
                    r.updated_at = date(),
                    // COALESCE: Если списка еще нет (связь от Скрипта 2), создай пустой и добавь.
                    // Если список есть, просто добавь.
                    r.evidence_log = CASE 
                        WHEN r.evidence_log IS NULL THEN [$new_fact]
                        WHEN NOT $new_fact IN r.evidence_log THEN r.evidence_log + $new_fact
                        ELSE r.evidence_log
                    END
                """

        with driver.session() as session:
            for rel_name in relations_to_process:
                try:
                    final_query = query.replace("REL_TYPE", rel_name)
                    session.run(final_query, t1=ticker_a, t2=ticker_b, new_fact=fact_json)

                    ev_short = fact_object['specific_evidence']
                    if len(ev_short) > 150: ev_short = ev_short[:150] + "..."

                    print(f"         -> Saved: {rel_name} [{fact_object['topic']}] (Ev: {ev_short})")
                except Exception as e:
                    print(f"         Neo4j Error: {e}")

    def clear_inference_data(self):
        """
        Удаляет только связи, созданные этим скриптом (Inference),
        не трогая базовый граф из Википедии/GDELT.
        """
        print("Очистка предыдущих результатов Inference...")
        query = """
        MATCH ()-[r]->()
        WHERE r.source CONTAINS 'Inference' OR r.evidence_log IS NOT NULL
        DELETE r
        """
        with driver.session() as session:
            session.run(query)
        print("Очистка завершена. База готова к новому прогону.")

    def run_pipeline(self):
        self.clear_inference_data()

        prods_df = self.get_all_products()
        comps_df = self.get_all_companies()

        print("\n--- ЭТАП 1: Поиск продуктовых пересечений ---")
        product_links = self.analyze_product_overlaps(prods_df)

        print("\n--- ЭТАП 2: Поиск семантических близнецов (Компании) ---")
        desc_links = self.analyze_company_similarity(comps_df)

        all_candidates = {}  # Key: (TickerA, TickerB), Value: List of evidence

        for k, v in product_links.items():
            all_candidates[k] = all_candidates.get(k, []) + v

        for k, v in desc_links.items():
            all_candidates[k] = all_candidates.get(k, []) + v

        print(f"\n Всего найдено {len(all_candidates)} пар кандидатов на новые связи.")

        sorted_candidates = sorted(all_candidates.items(), key=lambda x: len(x[1]), reverse=True)

        total_pairs = len(sorted_candidates)
        print(f"--- ЭТАП 3: LLM Валидация (Полная проверка: {total_pairs} пар) ---")

        for i, (pair, evidence) in enumerate(sorted_candidates):
            ticker_a, ticker_b = pair

            percent = round((i / total_pairs) * 100, 1)
            print(f"[{i + 1}/{total_pairs} | {percent}%] Анализ {ticker_a} <-> {ticker_b} ({len(evidence)} ev)...",
                  end=" ")

            try:
                relationships_list = self.verify_with_llm(ticker_a, ticker_b, evidence)

                if relationships_list:
                    print(f"Found {len(relationships_list)}")
                    for rel_item in relationships_list:
                        self.save_relationship(ticker_a, ticker_b, rel_item)
                else:
                    print("Skipped")

            except Exception as e:
                print(f"Error processing pair: {e}")

            time.sleep(0.2)

if __name__ == "__main__":
    enricher = GraphEnricher()
    enricher.run_pipeline()