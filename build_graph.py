import pandas as pd
import json
import os
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
import wikipedia
from rapidfuzz import process, fuzz

load_dotenv()

POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_lifetime=200,
    keep_alive=True
)

SP500_MAPPING = {}

def load_sp500_whitelist(df):
    """Загружает список компаний"""
    global SP500_MAPPING
    SP500_MAPPING = pd.Series(df.Ticker.values, index=df.Name).to_dict()
    print(f"список S&P 500 загружен: {len(SP500_MAPPING)} компаний.")

def find_sp500_ticker(company_name):
    """Ищет компанию в списке S&P 500"""
    if not company_name or not isinstance(company_name, str): return None

    if company_name in SP500_MAPPING: return SP500_MAPPING[company_name]

    try:
        match = process.extractOne(company_name, SP500_MAPPING.keys(), scorer=fuzz.token_sort_ratio)
        if match:
            best_name, score, _ = match
            if score > 85: return SP500_MAPPING[best_name]
    except:
        pass
    return None


def get_wiki_intel(company_name):
    """
    Ищет страницу компании в Википедии и берет оттуда текст.
    """
    try:
        search_results = wikipedia.search(f"{company_name} company")

        if not search_results:
            return ""

        page_title = search_results[0]

        page = wikipedia.page(page_title, auto_suggest=False)

        print(f"Готово: {page.content[:2000]}")

        return f"Wikipedia Title: {page.title}\nContent: {page.content[:2000]}"

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            print(f"Готово: {page.content[:2000]}")
            return f"Wikipedia Title: {page.title}\nContent: {page.content[:2000]}"
        except:
            return ""
    except Exception as e:
        print(f"   ⚠️ Wiki Error: {e}")
        return ""

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
    desc = str(row['Description'])[:800]

    wiki_data = get_wiki_intel(name)
    if wiki_data:
        print(f"   📖 Wiki found: {wiki_data.splitlines()[0]}")
    else:
        print(f"   ⚠️ Wiki not found, using only Yahoo desc.")
    prompt = f"""
    Context about company "{name}":
    1. Official Description: {desc}
    2. Web Search Results: {wiki_data}

    Task: Extract structured lists of entities based on the context.
    
    CRITICAL RULES:
    1. OUTPUT MUST BE IN ENGLISH ONLY. Translate if the source is not English.
    2. "products": Extract specific product names (e.g. "Windows", "Tylenol") or key service categories.
    3. "competitors": Extract specific company names.
    4. "partners": Extract specific company names mentioned as partners or suppliers.
    
    Return ONLY JSON. No markdown. No comments.
    Format: {{ "products": [...], "competitors": [...], "partners": [...] }}
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

        prods = len(data.get('products', []))
        comps = len(data.get('competitors', []))
        parts = len(data.get('partners', []))
        print(f"   🤖 LLM Extracted: {prods} Products, {comps} Competitors, {parts} Partners.")
        print(f"      -> Prods: {data.get('products')[:3]}...")
        if prods > 0: print(f"      Example Prod: {data.get('products')[0]}")

        return data
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return {}

def clear_database():
    """Чистит базу и создает индексы"""
    print("Очистка базы данных...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        try:
            session.run("DROP INDEX company_ticker IF EXISTS")
            session.run("DROP CONSTRAINT company_ticker IF EXISTS")
            session.run("DROP CONSTRAINT company_ticker_unique IF EXISTS")
        except Exception as e:
            print(f"⚠️ Warning cleaning schema: {e}")

        session.run("CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")

        session.run("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)")
        session.run("CREATE INDEX fund_name IF NOT EXISTS FOR (f:Fund) ON (f.name)")
        session.run("CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)")
    print("База чиста, индексы созданы.")

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
                    SET p.age = $age, p.title = $title
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
    for part in llm_data.get('partners', []):
        target = find_sp500_ticker(part)
        if target and target != ticker:
            session.run(
                "MATCH (c1:Company {ticker: $t1}) MERGE (c2:Company {ticker: $t2}) MERGE (c1)-[:PARTNER_WITH]->(c2)",
                t1=ticker, t2=target)
            print(f"      🔗 Link: Partner -> {part} ({target})")
        else:
            print(f"      ✂️ Skip: Partner {part} (Not in S&P500)")


    # Конкуренты
    for comp in llm_data.get('competitors', []):
        target = find_sp500_ticker(comp)
        if target and target != ticker:
            session.run(
                "MATCH (c1:Company {ticker: $t1}) MERGE (c2:Company {ticker: $t2}) MERGE (c1)-[:COMPETES_WITH]->(c2)",
                t1=ticker, t2=target)
            print(f"      ⚔️ Link: Competitor -> {comp} ({target})")


def main():
    print("Загружаем Excel...")
    df = pd.read_excel('data/sp500_graph_ready.xlsx')

    load_sp500_whitelist(df)

    clear_database()

    df = df.head(20)

    total = len(df)
    print(f"Начинаем обработку {total} компаний.")

    for i, row in df.iterrows():
        ticker = row['Ticker']
        print(f"[{i + 1}/{total}] {ticker}...", end=" ")

        llm_data = ask_llm_for_details(row)

        try:
            with driver.session() as session:
                build_graph(session, row, llm_data)
            print("✅ Готово")
        except Exception as e:
            print(f"❌ Ошибка записи в Neo4j: {e}")

    driver.close()
    print("Граф успешно построен!")


if __name__ == '__main__':
    main()