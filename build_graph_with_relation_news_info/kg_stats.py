import os
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv


def main():
    load_dotenv(find_dotenv())

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, pwd]):
        raise RuntimeError("Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    with driver.session() as session:
        totals = session.run("""
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS rels
        """).single()

        print("\n=== BASIC KG STATS ===")
        print(f"Nodes (|V|): {totals['nodes']}")
        print(f"Rels  (|E|): {totals['rels']}")

        print("\n--- Nodes by label ---")
        rows = session.run("""
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS cnt
            ORDER BY cnt DESC
        """)
        for r in rows:
            print(f"{r['label']}: {r['cnt']}")

        # 3) Relationships by type
        print("\n--- Relationships by type ---")
        rows = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(*) AS cnt
            ORDER BY cnt DESC
        """)
        for r in rows:
            print(f"{r['rel_type']}: {r['cnt']}")

        print("\n--- Top (Label)-[REL]->(Label) patterns (top 20) ---")
        rows = session.run("""
            MATCH (a)-[r]->(b)
            UNWIND labels(a) AS fromLabel
            UNWIND labels(b) AS toLabel
            RETURN fromLabel, type(r) AS rel_type, toLabel, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT 20
        """)
        for r in rows:
            print(f"{r['fromLabel']} -[{r['rel_type']}]-> {r['toLabel']}: {r['cnt']}")

    driver.close()
    print("\nDone.\n")


if __name__ == "__main__":
    main()