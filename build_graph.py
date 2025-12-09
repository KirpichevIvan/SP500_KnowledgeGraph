import pandas as pd
import json
import os
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


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


def ask_llm_for_details(row):
    """
    Просим LLM структурировать неструктурированный текст описания.
    """
    name = row['Name']
    desc = str(row['Description'])[:1500]

    prompt = f"""
    Проанализируй описание компании "{name}".
    Текст: {desc}

    Извлеки информацию и верни JSON объект со следующими полями (списками строк):
    1. "products": Ключевые продукты, бренды или технологии (например: "iPhone", "Azure", "mRNA").
    2. "markets": Рынки или сферы деятельности (например: "Cloud Computing", "E-commerce").
    3. "subsidiaries": Дочерние компании или приобретенные бренды (например: "YouTube", "Instagram").
    4. "partners": Упомянутые партнеры.
    5. "competitors": Упомянутые конкуренты.

    Отвечай ТОЛЬКО валидным JSON. Если какой-то список пуст, оставь [].
    Пример:
    {{
        "products": ["Windows", "Office"],
        "markets": ["Software", "Gaming"],
        "subsidiaries": ["GitHub"],
        "partners": ["OpenAI"],
        "competitors": ["Apple"]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model='qwen/qwen-2.5-7b-instruct',
            messages=[
                {'role': 'system', 'content': 'You are a strict data extraction assistant. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0
        )
        content = completion.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ Ошибка LLM для {name}: {e}")
        return {}

def clear_database(session):
    """Полная очистка базы перед новой загрузкой"""
    print("Очищаем базу данных...")
    session.run("MATCH (n) DETACH DELETE n")
    print("База пуста и готова к работе.")

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

            // 1. Создаем Город и Страну (они есть почти всегда)
            MERGE (city:City {name: $city})
            MERGE (cntry:Country {name: $country})

            // 2. Связываем Компанию с Городом
            MERGE (c)-[:LOCATED_IN]->(city)

            // 3. Логика со Штатом (он есть не у всех стран)
            FOREACH (ignoreMe IN CASE WHEN $state IS NOT NULL AND $state <> 'N/A' THEN [1] ELSE [] END |
                MERGE (s:State {name: $state})
                MERGE (city)-[:IN_STATE]->(s)
                MERGE (s)-[:IN_COUNTRY]->(cntry)
            )

            // 4. Если Штата нет, связываем Город напрямую со Страной
            FOREACH (ignoreMe IN CASE WHEN $state IS NULL OR $state = 'N/A' THEN [1] ELSE [] END |
                MERGE (city)-[:IN_COUNTRY]->(cntry)
            )
            """
            session.run(query_geo, ticker=ticker, city=city, state=state, country=country)
    except Exception as e:
        print(f"Geodata error: {e}")

    # ИЕРАРХИЯ
    sector = row.get('Sector')
    industry = row.get('Industry')

    if industry and industry != 'N/A' and sector and sector != 'N/A':
        session.run("""
                MATCH (c:Company {ticker: $ticker})
                MERGE (i:Industry {name: $industry})
                MERGE (s:Sector {name: $sector})

                // 1. Компания входит в Индустрию (Подсектор)
                MERGE (c)-[:OPERATES_IN_INDUSTRY]->(i)

                // 2. Индустрия входит в Сектор
                MERGE (i)-[:PART_OF]->(s)
            """, ticker=ticker, industry=industry, sector=sector)

    elif sector and sector != 'N/A':
        session.run("""
                MATCH (c:Company {ticker: $ticker})
                MERGE (s:Sector {name: $sector})
                MERGE (c)-[:OPERATES_IN_SECTOR]->(s)
            """, ticker=ticker, sector=sector)

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

    # LLM DATA

    # Продукты
    for prod in llm_data.get('products', []):
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (p:Product {name: $prod})
            MERGE (c)-[:PRODUCES]->(p)
        """, ticker=ticker, prod=prod)

    # Рынки
    for mkt in llm_data.get('markets', []):
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (m:Market {name: $mkt})
            MERGE (c)-[:SERVES_MARKET]->(m)
        """, ticker=ticker, mkt=mkt)

    # Дочки
    for sub in llm_data.get('subsidiaries', []):
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (s:Organization {name: $sub})
            MERGE (c)-[:OWNS_SUBSIDIARY]->(s)
        """, ticker=ticker, sub=sub)

    # Партнеры
    for partner in llm_data.get('partners', []):
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (o:Organization {name: $partner})
            MERGE (c)-[:PARTNER_WITH]->(o)
        """, ticker=ticker, partner=partner)

    # Конкуренты
    for comp in llm_data.get('competitors', []):
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            MERGE (o:Organization {name: $comp})
            MERGE (c)-[:COMPETES_WITH]->(o)
        """, ticker=ticker, comp=comp)


def main():
    print("Загружаем Excel...")
    df = pd.read_excel('data/sp500_graph_ready.xlsx')

    df = df.head(20)

    with driver.session() as session:
        clear_database(session)

        session.run("CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)")
        session.run("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)")
        session.run("CREATE INDEX fund_name IF NOT EXISTS FOR (f:Fund) ON (f.name)")
        session.run("CREATE INDEX city_name IF NOT EXISTS FOR (c:City) ON (c.name)")


        total = len(df)
        print(f"Начинаем построение графа для {total} компаний.")

        for i, row in df.iterrows():
            ticker = row['Ticker']
            print(f"[{i + 1}/{total}] {ticker}...", end=" ")

            llm_data = ask_llm_for_details(row)

            try:
                build_graph(session, row, llm_data)
                print("✅ Готово")
            except Exception as e:
                print(f"❌ Ошибка записи в Neo4j: {e}")

    driver.close()
    print("Граф построен успешно!")


if __name__ == '__main__':
    main()