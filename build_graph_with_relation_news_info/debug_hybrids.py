import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)


def debug_hybrids():
    print("\nДЕТАЛЬНЫЙ АНАЛИЗ ГИБРИДНЫХ СВЯЗЕЙ")
    print("=" * 60)

    query = """
    MATCH (a)-[r]->(b)
    WHERE type(r) IN ['COMPETES_WITH', 'PARTNER_WITH']
    RETURN a.name as src, type(r) as type, b.name as trg, 
           r.source as prime_source, r.evidence as legacy_ev, r.evidence_log as log
    """

    with driver.session() as session:
        results = list(session.run(query))

        hybrid_count = 0

        for record in results:
            sources_found = set()
            details = []

            if record['legacy_ev'] or "Wiki" in str(record['prime_source']) or "GDELT" in str(record['prime_source']):
                sources_found.add("Wiki")
                details.append("Found in Text/Wiki")

            if record['log']:
                for entry_str in record['log']:
                    try:
                        entry = json.loads(entry_str)
                        src_text = entry.get('source', '')

                        if "Vector" in src_text or "Inference" in src_text:
                            if "Supply Chain" not in src_text:
                                sources_found.add("Vector")
                                details.append("Found by Product Vector")

                        if "Supply Chain" in src_text:
                            sources_found.add("SupplyChain")
                            details.append("Found by Supply Chain Analysis")

                        if "Reuters" in src_text or "News" in src_text:
                            sources_found.add("News")
                            details.append(f"Found in News: {entry.get('headline', '')[:30]}...")

                        if "Wiki" in src_text:
                            sources_found.add("Wiki")
                            details.append("Found in Wiki (Log)")

                    except:
                        pass

            if len(sources_found) >= 2:
                hybrid_count += 1
                print(f"\nHYBRID #{hybrid_count}: {record['src']} --[{record['type']}]--> {record['trg']}")
                print(f"   Источники: {list(sources_found)}")
                print(f"   Детали: {details}")

    print(f"\nВсего найдено гибридов: {hybrid_count}")


if __name__ == "__main__":
    debug_hybrids()
    driver.close()