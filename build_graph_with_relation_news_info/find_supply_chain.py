import os
import json
import pandas as pd
import datetime
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

load_dotenv(find_dotenv())

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MODEL_PATH = r"C:\sp500_models\all-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'all-MiniLM-L6-v2'

MATCH_THRESHOLD = 0.70

client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class SupplyChainHunter:
    def __init__(self):
        print("Инициализация Supply Chain Hunter...")
        try:
            self.model = SentenceTransformer(MODEL_PATH)
            print("Модель загружена.")
        except Exception as e:
            print(f"Ошибка модели: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def clear_supply_chain_data(self):
        print("Умная очистка данных Supply Chain...")
        with driver.session() as session:
            session.run("""
                MATCH ()-[r:PARTNER_WITH]->()
                WHERE r.source = 'AI Supply Chain Analysis'
                DELETE r
            """)
            session.run("MATCH (r:Resource) DETACH DELETE r")

            session.run("""
                MATCH ()-[r:PARTNER_WITH]->()
                WHERE r.evidence_log IS NOT NULL
                WITH r, [entry IN r.evidence_log WHERE NOT (entry CONTAINS 'AI Supply Chain Analysis')] AS clean_log
                SET r.evidence_log = clean_log
            """)
        print("Очистка завершена.")

    def get_companies(self):
        query = "MATCH (c:Company) RETURN c.ticker as ticker, c.name as name, c.description as description"
        with driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.data() for r in result])

    def get_products(self):
        print("Загрузка каталога продуктов...")
        query = "MATCH (c:Company)-[:PRODUCES]->(p:Product) RETURN c.ticker as ticker, p.name as product_name"
        with driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([r.data() for r in result])

    def safe_json_parse(self, content):
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                return json.loads(json_str)
            return json.loads(content)
        except:
            return None

    def extract_needs_with_llm(self, company_row):
        name = company_row['name']
        desc = company_row['description'][:800]

        prompt = f"""
        Analyze "{name}". Description: {desc}
        Task: List 3-5 specific B2B INPUTS (raw materials, components, specialized services) this company MUST buy to operate.

        Rules:
        1. Be specific (e.g. "Lithium", "Cloud Storage", "Steel").
        2. IGNORE generic items (Electricity, Water, Office Supplies).
        3. JSON format: {{ "needs": ["Item1", "Item2"] }}
        """
        try:
            completion = client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}], temperature=0.0
            )
            return self.safe_json_parse(completion.choices[0].message.content).get("needs", [])
        except:
            return []

    def find_potential_suppliers(self, customer_ticker, needs_list, products_df, product_embeddings):
        if not needs_list or products_df.empty: return []

        needs_emb = self.model.encode(needs_list, convert_to_tensor=True)
        cosine_scores = util.cos_sim(needs_emb, product_embeddings)
        scores = cosine_scores.cpu().numpy()
        candidates = []

        for n_idx, need_txt in enumerate(needs_list):
            best_product_indices = scores[n_idx].argsort()[-3:][::-1]
            for p_idx in best_product_indices:
                score = scores[n_idx][p_idx]
                if score > MATCH_THRESHOLD:
                    supplier_ticker = products_df.iloc[p_idx]['ticker']
                    product_txt = products_df.iloc[p_idx]['product_name']
                    if supplier_ticker != customer_ticker:
                        candidates.append({
                            "supplier": supplier_ticker,
                            "product": product_txt,
                            "need": need_txt,
                            "score": float(score)
                        })
        return candidates

    def verify_partnership(self, customer_name, supplier_ticker, need, product):
        """
        Сбалансированный промпт: отсекает ритейл, но разрешает нормальный B2B.
        """
        prompt = f"""
        Task: Logic Check for Supply Chain.

        Scenario: 
        - Customer: "{customer_name}" (Needs: {need})
        - Potential Supplier: "{supplier_ticker}" (Produces: {product})

        Question: Is it plausible that "{supplier_ticker}" supplies "{product}" to "{customer_name}"?

        GUIDELINES:
        1. REJECT if Supplier is a RETAILER (e.g. Walmart, Target, Dollar General) selling to a MANUFACTURER. Factories don't buy raw materials at retail stores.
        2. REJECT if Supplier is a RAILROAD/LOGISTICS co (Union Pacific, FedEx) and the product is a physical good (like "Plastic"). They transport it, they don't make it.
        3. ACCEPT if it is a standard B2B relationship (e.g. Chemical co -> Pharma co, Tech co -> Bank, Steel -> Auto).

        Return JSON ONLY: 
        {{
            "is_supply_chain": true, 
            "confidence": "High",
            "reason": "Standard B2B relationship"
        }}
        OR {{ "is_supply_chain": false, "reason": "Supplier is a retailer" }}
        """
        try:
            completion = client.chat.completions.create(
                model='qwen/qwen-2.5-7b-instruct',
                messages=[{'role': 'user', 'content': prompt}], temperature=0.0
            )
            content = completion.choices[0].message.content
            parsed = self.safe_json_parse(content)

            if not parsed:
                print(f"   JSON Fail: {content[:50]}...")
                return None

            return parsed
        except Exception as e:
            print(f"   API Error: {e}")
            return None

    def save_supply_link(self, customer_ticker, supplier_ticker, details):
        fact = {
            "topic": "Supply Chain / Procurement",
            "specific_evidence": f"Supplier produces '{details['product']}' matching need '{details['need']}'",
            "reason": details.get('reason', 'Vector Match'),
            "source": "AI Supply Chain Analysis",
            "date_recorded": datetime.date.today().isoformat()
        }
        fact_json = json.dumps(fact, ensure_ascii=False)

        query = """
        MATCH (sup:Company {ticker: $sup_t})
        MATCH (cust:Company {ticker: $cust_t})
        MERGE (sup)-[r:PARTNER_WITH]->(cust)
        ON CREATE SET 
            r.created_at = date(), r.updated_at = date(),
            r.source = 'AI Supply Chain Analysis', r.subtype = 'Potential Supply Chain',
            r.evidence_log = [$new_fact]
        ON MATCH SET 
            r.updated_at = date(),
            r.subtype = CASE WHEN r.subtype IS NULL THEN 'Hybrid' ELSE r.subtype END,
            r.evidence_log = CASE 
                WHEN r.evidence_log IS NULL THEN [$new_fact]
                WHEN NOT $new_fact IN r.evidence_log THEN r.evidence_log + $new_fact
                ELSE r.evidence_log
            END
        """
        with driver.session() as session:
            session.run(query, sup_t=supplier_ticker, cust_t=customer_ticker, new_fact=fact_json)
            print(f"      Saved: {supplier_ticker} -> {customer_ticker} ({details['product']})")

    def run(self):
        self.clear_supply_chain_data()

        print("--- ЭТАП 1: Подготовка данных ---")
        companies_df = self.get_companies()
        products_df = self.get_products()

        print(f"Векторизация {len(products_df)} продуктов...")
        product_embeddings = self.model.encode(products_df['product_name'].tolist(), convert_to_tensor=True,
                                               show_progress_bar=True)

        print(f"\n--- ЭТАП 2: Анализ Supply Chain ({len(companies_df)} компаний) ---")

        for i, row in companies_df.iterrows():
            ticker = row['ticker']
            name = row['name']

            if i % 10 == 0: print(f"Processing {i}/{len(companies_df)} ({ticker})...")

            needs = self.extract_needs_with_llm(row)
            if not needs: continue

            try:
                with driver.session() as session:
                    for need_item in needs:
                        clean_need = need_item.strip().title()
                        session.run("""
                            MATCH (c:Company {ticker: $ticker})
                            MERGE (r:Resource {name: $r_name})
                            MERGE (c)-[:REQUIRES]->(r)
                        """, ticker=ticker, r_name=clean_need)
            except:
                pass

            matches = self.find_potential_suppliers(ticker, needs, products_df, product_embeddings)

            if matches:
                top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]

                for match in top_matches:
                    llm_check = self.verify_partnership(name, match['supplier'], match['need'], match['product'])

                    if llm_check and llm_check.get('is_supply_chain') and llm_check.get('confidence') == 'High':
                        match['reason'] = llm_check.get('reason')
                        self.save_supply_link(ticker, match['supplier'], match)
                    else:
                        reason = llm_check.get('reason') if llm_check else "LLM Error"
                        print(f"      Rejected: {match['supplier']} -> {ticker} ({reason})")

                    time.sleep(0.2)


if __name__ == "__main__":
    hunter = SupplyChainHunter()
    hunter.run()