import pandas as pd
import json
import os
from openai import OpenAI
import requests
import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
import wikipedia
from rapidfuzz import process, fuzz

load_dotenv(find_dotenv())

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

def get_llm_match_decision(entity_name, candidates):
    """
    Спрашивает у LLM, подходит ли какой-то кандидат под имя.
    """
    candidates_str = "\n".join([f"- {name} (Ticker: {ticker})" for name, ticker in candidates])

    prompt = f"""
    Task: Match the entity name found in text to the official S&P 500 company list.

    [ENTITY FOUND IN TEXT]: "{entity_name}"

    [OFFICIAL CANDIDATES]:
    {candidates_str}

    Instructions:
    1. If the entity is definitely one of the candidates (even if names slightly differ, e.g. "Google" -> "Alphabet"), return its Ticker.
    2. If NONE match, return null.

    Return JSON ONLY: {{ "match_ticker": "XYZ" }} or {{ "match_ticker": null }}
    """

    try:
        completion = client.chat.completions.create(
            model='qwen/qwen-2.5-7b-instruct',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0
        )
        content = completion.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data.get("match_ticker")
    except Exception as e:
        print(f"      ⚠️ LLM Match Error: {e}")
        return None

def find_sp500_ticker(company_name):
    """
    Умный поиск через LLM.
    """
    if not company_name or not isinstance(company_name, str): return None

    if company_name in SP500_MAPPING:
        return SP500_MAPPING[company_name]

    matches = process.extract(company_name, SP500_MAPPING.keys(), limit=5, scorer=fuzz.WRatio)

    candidates = []
    for match_name, score, _ in matches:
        if score > 50:
            candidates.append((match_name, SP500_MAPPING[match_name]))

    if not candidates:
        return None

    print(f"      🔎 LLM Checking: '{company_name}' vs {len(candidates)} options...", end="")
    best_ticker = get_llm_match_decision(company_name, candidates)

    if best_ticker:
        print(f" ✅ Match: {best_ticker}")
        return best_ticker
    else:
        print(f" ❌ No match")
        return None


def get_wiki_intel(company_name):
    """
    Ищет страницу компании в Википедии и берет оттуда текст.
    """
    try:
        results = wikipedia.search(f"{company_name} company")

        if not results:
            return ""

        page = wikipedia.page(results[0], auto_suggest=False)

        content = f"SUMMARY:\n{page.summary[:800]}\n\n"

        keywords = ['product', 'service', 'operation', 'division', 'segment', 'business']

        found_sections = 0
        for section in page.sections:
            if any(k in section.lower() for k in keywords):
                try:
                    sec_content = page.section(section)
                    if sec_content:
                        content += f"SECTION '{section.upper()}':\n{sec_content[:1500]}\n\n"
                        found_sections += 1
                except:
                    pass

            if found_sections >= 2:
                break

        if found_sections == 0:
            content += f"CONTENT:\n{page.content[:1500]}"

        return content[:3500]

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


def get_gdelt_partnerships(company_name):
    """
    Ищет в GDELT новости о партнерствах за последний год.
    """
    print(f"   📡 GDELT: Ищем сделки для {company_name}...")

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    query = f'"{company_name}" (partnership OR collaboration OR "joint venture" OR acquisition) sourcelang:eng'

    params = {
        'query': query,
        'mode': 'artlist',  # Список статей
        'maxrecords': '5',  # Максимум статей
        'format': 'json',  # Формат ответа
        'timespan': '18m'  # За последний период
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    text_result = ""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        try:
            data = response.json()
        except json.JSONDecodeError:
            return ""

        if 'articles' in data:
            for art in data['articles']:
                text_result += f"- News: {art.get('title', '')}\n"
    except Exception as e:
        print(f"   ⚠️ GDELT Error: {e}")

    return text_result

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
        print(f"   📖 Wiki found: {wiki_data}")
    else:
        print(f"   ⚠️ Wiki not found, using only Yahoo desc.")

    news_data = get_gdelt_partnerships(name)
    if news_data:
        print(f"   📖 News found: {news_data}")
    else:
        print(f"   ⚠️ News not found.")

    prompt = f"""
    Analyze data about "{name}".

    [DESCRIPTION]: {desc}
    [WIKIPEDIA]: {wiki_data}
    [NEWS (Partnerships)]: {news_data}

    Task: Extract structured lists with EVIDENCE.

    1. "products": List of key product names or service lines.
    2. "partners": List of strategic partners/suppliers.
       - "name": Company name.
       - "evidence": Short reason (e.g. "Joint venture for AI chips").
    3. "competitors": List of major competitors.
       - "name": Company name.
       - "evidence": Short reason (e.g. "Rival in streaming market").

    CRITICAL RULES:
    1. IGNORE Market Summaries: If a news headline lists multiple companies just for earnings...
       -> EXCEPTION: If the headline describes a specific INTERACTION...
    2. IGNORE Subsidiaries...
    3. CLEAN NAMES: In the "name" field, output ONLY the proper company name (e.g. "Samsung"). DO NOT write sentences like "Mentioned as key competitor" in the name field.
    4. OUTPUT ENGLISH ONLY.

    Return JSON Example:
    {{
        "products": ["iPhone", "Mac"],
        "partners": [ {{"name": "OpenAI", "evidence": "Integration deal"}} ],
        "competitors": [ {{"name": "Samsung", "evidence": "Competes in smartphones"}}, {{"name": "Netflix", "evidence": "Streaming rival"}} ]
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

        prods = len(data.get('products', []))
        comps = len(data.get('competitors', []))
        parts = len(data.get('partners', []))
        print(f'products {data.get('products', [])}, competitors {data.get('competitors', [])}, partners {data.get('partners', [])}')
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
            session.run("DROP CONSTRAINT company_ticker IF EXISTS")
            session.run("DROP CONSTRAINT company_ticker_unique IF EXISTS")
        except Exception as e:
            print(f"⚠️ Warning cleaning schema: {e}")

        session.run("CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")

        session.run("CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)")
        session.run("CREATE INDEX fund_name IF NOT EXISTS FOR (f:Fund) ON (f.name)")
        session.run("CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)")
        session.run("CREATE INDEX org_name IF NOT EXISTS FOR (o:Organization) ON (o.name)")
        session.run("CREATE INDEX industry_name IF NOT EXISTS FOR (i:Industry) ON (i.name)")
    print("База чиста, индексы созданы.")

def build_graph(session, row, llm_data):
    ticker = row['Ticker']
    today = datetime.date.today().isoformat()

    # УЗЕЛ КОМПАНИИ
    query_company = """
    MERGE (c:Company {ticker: $ticker})
    SET c.name = $name,
        c.description = $desc,
        c.website = $website,
        c.market_cap = $mcap,
        c.employees = $emp,
        c.last_updated = $date
    """
    session.run(query_company,
                ticker=ticker,
                name=row['Name'],
                desc=str(row['Description'])[:500],
                website=row.get('Website', 'N/A'),
                mcap=clean_money(row.get('Market Cap')),
                emp=clean_money(row.get('Employees')),
                date=today)

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
        p_name = part.get('name')
        evidence = part.get('evidence', 'No evidence provided')

        sp500_ticker = find_sp500_ticker(p_name)

        if sp500_ticker and sp500_ticker != ticker:
            session.run("""
                    MATCH (c1:Company {ticker: $t1})
                    MERGE (c2:Company {ticker: $t2})
                    MERGE (c1)-[r:PARTNER_WITH]->(c2)
                    SET r.source = 'LLM Extraction (News GDELT / Wiki / Description)', r.evidence = $ev,
                            r.last_updated = date()
                """, t1=ticker, t2=sp500_ticker, ev=evidence)
            print(f"      🔗 Link (Company): {ticker} <-> {sp500_ticker} (Ev: {evidence[:30]}...)")


    # Конкуренты
    for comp in llm_data.get('competitors', []):
        if isinstance(comp, str):
            comp_name = comp
            comp_evidence = "Mentioned as competitor in text"
        else:
            comp_name = comp.get('name')
            comp_evidence = comp.get('evidence', 'Mentioned as competitor')

        target = find_sp500_ticker(comp_name)

        if target and target != ticker:
            session.run("""
                        MATCH (c1:Company {ticker: $t1}) 
                        MERGE (c2:Company {ticker: $t2}) 
                        MERGE (c1)-[r:COMPETES_WITH]->(c2)
                        SET r.source = 'LLM Extraction (News GDELT / Wiki / Description)',
                            r.evidence = $ev,
                            r.last_updated = date()
                """, t1=ticker, t2=target, ev=comp_evidence)

            ev_short = comp_evidence[:30] + "..." if len(comp_evidence) > 30 else comp_evidence
            print(f"      ⚔️ Link: Competitor -> {comp_name} ({target}) [Ev: {ev_short}]")

def main():
    print("Загружаем Excel...")
    df = pd.read_excel('../data/sp500_graph_ready.xlsx')

    load_sp500_whitelist(df)

    clear_database()

    # df = df.head(20)

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