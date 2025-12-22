import pandas as pd
import json
import os
import ollama
from typing import List, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Настройки Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Инициализация драйвера Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- Pydantic модели для Structured Output ---
# Это определяет схему, которой ОБЯЗАНА следовать модель
class CompanyDetails(BaseModel):
    products: List[str] = Field(default_factory=list, description="Ключевые продукты, бренды или технологии")
    markets: List[str] = Field(default_factory=list, description="Рынки или сферы деятельности")
    subsidiaries: List[str] = Field(default_factory=list, description="Дочерние компании или приобретенные бренды")
    partners: List[str] = Field(default_factory=list, description="Упомянутые партнеры")
    competitors: List[str] = Field(default_factory=list, description="Упомянутые конкуренты")

def clean_money(value):
    """
    Возвращает число, если оно есть.
    Возвращает None, если данных нет (в базе это будет null).
    """
    try:
        if pd.isna(value) or value == 'N/A' or value == '':
            return None
        cleaned_val = float(value)
        if pd.isna(cleaned_val):
            return None
        return cleaned_val
    except:
        return None

def ask_llm_for_details(row) -> dict:
    """
    Использует Ollama для извлечения структурированных данных.
    """
    name = row['Name']
    desc = str(row['Description'])[:1500]

    prompt = f"""
    Проанализируй описание компании "{name}".
    Текст: {desc}
    
    Извлеки информацию о продуктах, рынках, дочерних компаниях, партнерах и конкурентах.
    Будь точен. Если информации нет, оставляй список пустым.
    """

    try:
        # Использование client.chat с параметром format (schema)
        # Это заставляет модель генерировать JSON строго по схеме Pydantic
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': 'Ты аналитик данных. Извлекай сущности строго по схеме JSON.'},
                {'role': 'user', 'content': prompt},
            ],
            format=CompanyDetails.model_json_schema(),
            options={'temperature': 0.0}
        )

        json_str = response['message']['content']
        
        # Валидация через Pydantic (превращает JSON строку в объект Python)
        details = CompanyDetails.model_validate_json(json_str)
        
        # Возвращаем как словарь для совместимости с остальным кодом
        return details.model_dump()

    except Exception as e:
        print(f"⚠️ Ошибка Ollama для {name}: {e}")
        # Возвращаем пустую структуру в случае ошибки
        return CompanyDetails().model_dump()


def build_graph(session, row, llm_data):
    ticker = row['Ticker']

    # УЗЕЛ КОМПАНИИ
    query_company = """
    MERGE (c:Company {ticker: $ticker})
    SET c.name = $name,
        c.description = $desc,
        c.website = $website,
        c.market_cap = $mcap,
        c.employees = $emp
    """
    session.run(query_company,
                ticker=ticker,
                name=row['Name'],
                desc=str(row['Description'])[:500],
                website=row.get('Website', 'N/A'),
                mcap=clean_money(row.get('Market Cap')),
                emp=clean_money(row.get('Employees')))

    # ГЕОГРАФИЯ (Город -> Штат -> Страна)
    try:
        addr = json.loads(row['Address_JSON'])
        city = addr.get('city')
        state = addr.get('state')
        country = addr.get('country')

        if city and city != 'N/A':
            query_geo = """
            MATCH (c:Company {ticker: $ticker})

            MERGE (city:City {name: $city})
            MERGE (cntry:Country {name: $country})
            MERGE (c)-[:LOCATED_IN]->(city)

            FOREACH (ignoreMe IN CASE WHEN $state IS NOT NULL AND $state <> 'N/A' THEN [1] ELSE [] END |
                MERGE (s:State {name: $state})
                MERGE (city)-[:IN_STATE]->(s)
                MERGE (s)-[:IN_COUNTRY]->(cntry)
            )

            FOREACH (ignoreMe IN CASE WHEN $state IS NULL OR $state = 'N/A' THEN [1] ELSE [] END |
                MERGE (city)-[:IN_COUNTRY]->(cntry)
            )
            """
            session.run(query_geo, ticker=ticker, city=city, state=state, country=country)
    except Exception as e:
        # print(f"Geodata warning: {e}") 
        pass

    # ИЕРАРХИЯ
    sector = row.get('Sector')
    industry = row.get('Industry')

    if sector and sector != 'N/A':
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (s:Sector {name: $sector})
            MERGE (c)-[:OPERATES_IN_SECTOR]->(s)
        """, ticker=ticker, sector=sector)

    if industry and industry != 'N/A':
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (i:Industry {name: $industry})
            MERGE (c)-[:OPERATES_IN_INDUSTRY]->(i)
            WITH i
            MATCH (s:Sector {name: $sector})
            MERGE (i)-[:PART_OF]->(s)
        """, ticker=ticker, industry=industry, sector=sector)

    # ЛЮДИ
    try:
        officers = json.loads(row['Officers_JSON'])
        for p in officers:
            if p.get('name'):
                session.run("""
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (p:Person {name: $p_name})
                    SET p.age = $age
                    MERGE (p)-[:WORKS_FOR {title: $title}]->(c)
                """, ticker=ticker, p_name=p['name'], title=p.get('title', ''), age=p.get('age'))
    except:
        pass

    # ВЛАДЕЛЬЦЫ
    try:
        holders = json.loads(row['Holders_JSON'])
        for h in holders:
            if h.get('Holder'):
                session.run("""
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (f:Fund {name: $h_name})
                    MERGE (f)-[:OWNS {percentage: $pct}]->(c)
                """, ticker=ticker, h_name=h['Holder'], pct=h.get('pctHeld', 0))
    except:
        pass

    # LLM DATA (Интеграция данных от Ollama)
    
    # Вспомогательная функция для чистой вставки
    def merge_relation(item_list, node_label, rel_type):
        for item in item_list:
            if item and isinstance(item, str):
                session.run(f"""
                    MATCH (c:Company {{ticker: $ticker}})
                    MERGE (n:{node_label} {{name: $name}})
                    MERGE (c)-[:{rel_type}]->(n)
                """, ticker=ticker, name=item)

    merge_relation(llm_data.get('products', []), 'Product', 'PRODUCES')
    merge_relation(llm_data.get('markets', []), 'Market', 'SERVES_MARKET')
    merge_relation(llm_data.get('subsidiaries', []), 'Organization', 'OWNS_SUBSIDIARY')
    merge_relation(llm_data.get('partners', []), 'Organization', 'PARTNER_WITH')
    merge_relation(llm_data.get('competitors', []), 'Organization', 'COMPETES_WITH')


def main():
    # Проверка наличия файла
    file_path = './data/sp500_graph_ready.xlsx'
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return

    print("Загружаем Excel...")
    df = pd.read_excel(file_path)
    df = df.head(100)

    with driver.session() as session:
        # Создание индексов (Constraints быстрее и надежнее индексов для уникальности)
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (f:Fund) ON (f.name)")
        
        total = len(df)
        print(f"Начинаем построение графа для {total} компаний, используя модель {OLLAMA_MODEL}.")

        for i, row in df.iterrows():
            ticker = row['Ticker']
            print(f"[{i + 1}/{total}] {ticker}...", end=" ", flush=True)

            # 1. Запрос к Ollama
            llm_data = ask_llm_for_details(row)

            # 2. Запись в Neo4j
            try:
                build_graph(session, row, llm_data)
                print("✅ Готово")
            except Exception as e:
                print(f"❌ Ошибка Neo4j: {e}")

    driver.close()
    print("Граф построен успешно!")

if __name__ == '__main__':
    main()