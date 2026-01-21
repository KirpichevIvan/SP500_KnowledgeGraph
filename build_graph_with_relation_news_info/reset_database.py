import os
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def nuclear_reset():
    print("ЗАПУСК ПОЛНОЙ ОЧИСТКИ БАЗЫ ДАННЫХ...")

    with driver.session() as session:
        print("1. Удаление всех узлов и связей...")
        session.run("MATCH (n) DETACH DELETE n")

        print("2. Удаление векторных индексов...")
        try:
            session.run("DROP INDEX entity_vector_index IF EXISTS")
            session.run("DROP INDEX unified_entity_index IF EXISTS")  # На случай если старое название осталось
        except Exception as e:
            print(f"   (Info: {e})")

        print("3. Удаление ограничений (Constraints)...")
        constraints = session.run("SHOW CONSTRAINTS")
        for record in constraints:
            name = record['name']
            try:
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                print(f"   - Удален constraint: {name}")
            except Exception as e:
                print(f"   ! Ошибка удаления {name}: {e}")

        print("4. Удаление обычных индексов...")
        indexes = session.run("SHOW INDEXES")
        for record in indexes:
            name = record['name']
            if record['type'] != 'LOOKUP':
                try:
                    session.run(f"DROP INDEX {name} IF EXISTS")
                    print(f"   - Удален индекс: {name}")
                except Exception as e:
                    print(f"   ! Ошибка удаления {name}: {e}")

    print("\nБАЗА ПОЛНОСТЬЮ ОЧИЩЕНА.")

if __name__ == "__main__":
    nuclear_reset()
    driver.close()