import pandas as pd
import json
import os
import re
from datetime import datetime
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from rapidfuzz import process, fuzz
import time

load_dotenv(find_dotenv())

# --- КОНФИГУРАЦИЯ ---
POLZA_KEY = os.getenv("POLZA_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=POLZA_KEY, base_url="https://api.polza.ai/api/v1")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

WHITELIST = {}


# --- ЗАГРУЗКА БАЗЫ (Компании, Люди, Фонды) ---
def clean_name(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Inc\.|Corp\.|plc|Ltd\.)\b', '', text, flags=re.IGNORECASE)
    return text.strip()


def load_comprehensive_whitelist():
    global WHITELIST
    print("📋 Loading Whitelist...")
    try:
        # Убедитесь, что файл существует
        df = pd.read_excel('../data/sp500_graph_ready.xlsx')
        for _, row in df.iterrows():
            c_name = str(row['Name'])
            ticker = str(row['Ticker'])
            WHITELIST[c_name] = {"id": ticker, "type": "Company", "key_field": "ticker"}
            WHITELIST[ticker] = {"id": ticker, "type": "Company", "key_field": "ticker"}

            try:
                officers = json.loads(row['Officers_JSON'])
                for off in officers:
                    p_name = clean_name(off.get('name'))
                    if len(p_name) > 3:
                        WHITELIST[p_name] = {"id": p_name, "type": "Person", "key_field": "name"}
            except:
                pass

            try:
                holders = json.loads(row['Holders_JSON'])
                for h in holders:
                    f_name = clean_name(h.get('Holder'))
                    if len(f_name) > 3:
                        WHITELIST[f_name] = {"id": f_name, "type": "Fund", "key_field": "name"}
            except:
                pass

            try:
                addr = json.loads(row['Address_JSON'])
                if addr.get('city') and addr.get('city') != 'N/A':
                    WHITELIST[addr['city']] = {"id": addr['city'], "type": "City", "key_field": "name"}
                if addr.get('country') and addr.get('country') != 'N/A':
                    WHITELIST[addr['country']] = {"id": addr['country'], "type": "Country", "key_field": "name"}
            except:
                pass

        print(f"   ✅ Whitelist loaded: {len(WHITELIST)} entries.")
    except Exception as e:
        print(f"   ❌ Error loading Excel: {e}")
        exit()


def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str.strip(), "%b %d %Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return datetime.today().strftime("%Y-%m-%d")


# --- ПОИСК СУЩНОСТИ ---
def resolve_alias_via_llm(name):
    prompt = f"""
    Identify the official entity name for "{name}".
    Context: S&P 500 companies, Funds, Locations.
    Examples: "Google"->"Alphabet Inc.", "Fed"->"Federal Reserve", "NY"->"New York".
    Return JSON: {{ "official_name": "..." }}
    """
    try:
        completion = client.chat.completions.create(
            model='qwen/qwen-2.5-7b-instruct',
            messages=[{'role': 'user', 'content': prompt}], temperature=0.0
        )
        content = completion.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        res = json.loads(content)
        return res.get('official_name')
    except:
        return None


def find_entity(name):
    if not name or not isinstance(name, str): return None
    clean_search = clean_name(name)

    if clean_search in WHITELIST:
        return WHITELIST[clean_search]

    fuzzy_candidates = [k for k in WHITELIST.keys() if len(k) > 3]

    match = process.extractOne(clean_search, fuzzy_candidates, scorer=fuzz.WRatio)

    if match and match[1] > 90:
        return WHITELIST[match[0]]

    if len(clean_search) > 2:
        official = resolve_alias_via_llm(clean_search)
        if official:
            if official in WHITELIST: return WHITELIST[official]
            match_llm = process.extractOne(official, fuzzy_candidates, scorer=fuzz.WRatio)
            if match_llm and match_llm[1] > 90: return WHITELIST[match_llm[0]]

    return None


# --- АНАЛИЗ ОДНОЙ НОВОСТИ (ИСПРАВЛЕНО) ---
def analyze_single_news(news_item):
    text = f"Headline: {news_item['headline']}\nDescription: {news_item['description']}"

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
            "relation": "PARTNERSHIP" (or DISPUTE, INVESTMENT, REGULATION, EXPANSION, AFFECTS),
            "summary": "Max 10 words summary"
        }},
        "sentiment": "POSITIVE" / "NEGATIVE" / "NEUTRAL"
    }}

    Rules:
    - If no interaction, set "interaction": null.
    - JSON ONLY.
    """
    try:
        completion = client.chat.completions.create(
            model='qwen/qwen-2.5-7b-instruct',
            messages=[{'role': 'user', 'content': prompt}], temperature=0.0
        )
        content = completion.choices[0].message.content.replace("```json", "").replace("```", "").strip()

        # --- ФИКС ЗДЕСЬ: Проверка на список ---
        parsed_data = json.loads(content)

        if isinstance(parsed_data, list):
            # Если LLM вернула список, берем первый элемент
            if len(parsed_data) > 0:
                return parsed_data[0]
            else:
                return None

        return parsed_data

    except Exception as e:
        # print(f"JSON Error: {e}")
        return None


# --- ЗАПИСЬ В БАЗУ ---
def save_to_graph(session, analysis, news_item):
    if not analysis: return 0

    # Теперь .get() точно сработает, так как analysis это dict
    entities_raw = analysis.get('entities', [])
    interaction = analysis.get('interaction')
    sentiment = analysis.get('sentiment', 'NEUTRAL')

    iso_date = parse_date(news_item['date'])
    headline = news_item['headline']

    valid_entities = []
    seen = set()
    for name in entities_raw:
        match = find_entity(name)
        if match:
            uid = f"{match['type']}_{match['id']}"
            if uid not in seen:
                valid_entities.append(match)
                seen.add(uid)

    if not valid_entities: return 0

    summary = headline
    if interaction and isinstance(interaction, dict):
        summary = interaction.get('summary', headline)

    log_entry = f"[{iso_date}] {sentiment}: {headline} -> {summary}"

    # СЦЕНАРИЙ А: Взаимодействие
    if interaction and isinstance(interaction, dict) and len(valid_entities) >= 2:
        src_match = find_entity(interaction.get('source'))
        trg_match = find_entity(interaction.get('target'))

        if src_match and trg_match and src_match != trg_match:
            ent1, ent2 = src_match, trg_match
            rel_type = interaction.get('relation', 'RELATED_TO').upper()

            cypher_rel = "RELATED_TO"
            if "PARTNER" in rel_type:
                cypher_rel = "PARTNER_WITH"
            elif "DISPUTE" in rel_type:
                cypher_rel = "IN_DISPUTE_WITH"
            elif "INVEST" in rel_type:
                cypher_rel = "INVESTED_IN"
            elif ent2['type'] in ['City', 'Country']:
                cypher_rel = "AFFECTS_REGION"

            print(f"   🔗 Link: {ent1['id']} <-> {ent2['id']} [{cypher_rel}]")

            query = f"""
            MATCH (a:{ent1['type']} {{ {ent1['key_field']}: $id1 }})
            MATCH (b:{ent2['type']} {{ {ent2['key_field']}: $id2 }})
            MERGE (a)-[r:{cypher_rel}]->(b)
            ON CREATE SET r.created_at = date($date), r.news_history = [$log]
            ON MATCH SET r.news_history = r.news_history + $log, r.last_updated = date($date)
            """
            session.run(query, id1=ent1['id'], id2=ent2['id'], date=iso_date, log=log_entry)
            return 1

    # СЦЕНАРИЙ Б: Новость-узел
    ent = valid_entities[0]
    print(f"   📰 News: {ent['id']} ({ent['type']})")

    query = f"""
    MATCH (e:{ent['type']} {{ {ent['key_field']}: $id }})
    MERGE (n:News {{headline: $headline, date: date($date)}})
    SET n.sentiment = $sent
    MERGE (n)-[:MENTIONS]->(e)
    """
    session.run(query, id=ent['id'], headline=headline, date=iso_date, sent=sentiment)
    return 1


def main():
    print("=== SINGLE NEWS PROCESSING ===")
    load_comprehensive_whitelist()

    try:
        df = pd.read_csv('../data/classified_reuters_news_mapped.csv')
    except:
        print("❌ CSV missing")
        return

    total = len(df)
    print(f"🚀 Processing {total} news items...")

    processed_count = 0
    with driver.session() as session:
        for i, row in df.iterrows():
            print(f"[{i + 1}/{total}]", end=" ")

            item = {
                "headline": str(row['headline']),
                "description": str(row['text']),
                "date": str(row['Time'])
            }

            analysis = analyze_single_news(item)
            if analysis:
                saved = save_to_graph(session, analysis, item)
                if saved == 0: print("No whitelist match.")
                processed_count += saved
            else:
                print("LLM failed.")

    driver.close()
    print(f"\n✅ DONE! Processed {processed_count} news events.")


if __name__ == "__main__":
    main()