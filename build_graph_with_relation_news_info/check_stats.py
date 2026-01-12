import os
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def print_stat_line(label, value, percent=None):
    if percent is not None:
        print(f"{label:<55} | {value:>6}  ({percent:>5.1f}%)")
    else:
        print(f"{label:<55} | {value:>6}")


def check_stats():
    print("\n" + "=" * 75)
    print(f"{'📊 ПОДРОБНАЯ СТАТИСТИКА ГРАФА (ПЕРЕСЕЧЕНИЕ АЛГОРИТМОВ)':^75}")
    print("=" * 75)

    with driver.session() as session:
        query_total = """
        MATCH ()-[r]->()
        WHERE type(r) IN ['COMPETES_WITH', 'PARTNER_WITH']
        RETURN count(r) as cnt
        """
        total = session.run(query_total).single()["cnt"]

        if total == 0:
            print("Граф пуст или нет связей типа COMPETES/PARTNER.")
            return

        print_stat_line("ВСЕГО СВЯЗЕЙ (Competitors + Partners)", total)
        print("-" * 75)

        query_legacy = """
        MATCH ()-[r]->()
        WHERE type(r) IN ['COMPETES_WITH', 'PARTNER_WITH']
          AND r.evidence IS NOT NULL 
          AND r.evidence_log IS NULL
        RETURN count(r) as cnt
        """
        legacy = session.run(query_legacy).single()["cnt"]

        query_new = """
        MATCH ()-[r]->()
        WHERE type(r) IN ['COMPETES_WITH', 'PARTNER_WITH']
          AND r.evidence IS NULL 
          AND r.evidence_log IS NOT NULL
        RETURN count(r) as cnt
        """
        new_algo = session.run(query_new).single()["cnt"]

        query_hybrid = """
        MATCH ()-[r]->()
        WHERE type(r) IN ['COMPETES_WITH', 'PARTNER_WITH']
          AND r.evidence IS NOT NULL 
          AND r.evidence_log IS NOT NULL
        RETURN count(r) as cnt
        """
        hybrid = session.run(query_hybrid).single()["cnt"]

        others = total - (legacy + new_algo + hybrid)

        # ВЫВОД
        print_stat_line("Только из Вики/Новостей (Скрипт 2)", legacy, (legacy / total) * 100)
        print_stat_line("Только из Векторного анализа (Скрипт 3)", new_algo, (new_algo / total) * 100)
        print_stat_line("ГИБРИДНЫЕ (Подтверждены обоими методами)", hybrid, (hybrid / total) * 100)

        if others > 0:
            print_stat_line("Прочие (без метаданных)", others, (others / total) * 100)

        print("=" * 75 + "\n")


if __name__ == "__main__":
    check_stats()
    driver.close()