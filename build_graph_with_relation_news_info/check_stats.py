import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)


def analyze_focused():
    print("\n" + "=" * 80)
    print(f"{'📊 ФИНАЛЬНАЯ СТАТИСТИКА ДЛЯ ЗАЩИТЫ НИР (FIXED)':^80}")
    print("=" * 80)

    with driver.session() as session:
        # 1. ОБЩИЙ ОБЪЕМ ГРАФА
        n_total = session.run("MATCH (n) RETURN count(n) as c").single()['c']
        r_total = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']

        # 2. ФИНАНСОВЫЙ СЛОЙ
        r_financial = session.run("""
            MATCH ()-[r:OWNS|INVESTED_IN]->() 
            RETURN count(r) as c
        """).single()['c']

        # 3. ОПЕРАЦИОННЫЙ СЛОЙ (Бизнес-связи)
        # Получаем данные о происхождении каждой связи
        query_biz = """
        MATCH ()-[r:PARTNER_WITH|COMPETES_WITH]->()
        RETURN r.source as prime_source, r.evidence as legacy_ev, r.evidence_log as log
        """
        biz_results = list(session.run(query_biz))
        r_biz_total = len(biz_results)

        stats = {
            "Explicit (Wiki/Text)": 0,
            "Implicit (Vector Competitors)": 0,
            "Implicit (Supply Chain Partners)": 0,
            "Events (Reuters News)": 0,
            "Hybrid (Verified by 2+ sources)": 0
        }

        for record in biz_results:
            sources = set()

            # --- АНАЛИЗ ГЛАВНОГО ИСТОЧНИКА (КТО СОЗДАЛ СВЯЗЬ) ---
            prime_src = str(record['prime_source'])
            legacy_ev = record['legacy_ev']

            # 1. Wiki/GDELT (Скрипт 2)
            if legacy_ev is not None or "Wiki" in prime_src or "GDELT" in prime_src or "LLM Extraction" in prime_src:
                sources.add("Wiki")

            # 2. Vector Inference (Скрипт 3)
            # ВАЖНО: Скрипт 3 пишет "AI Inference" или "Vector Inference"
            if "Vector" in prime_src or "Inference" in prime_src:
                if "Supply Chain" not in prime_src:  # Исключаем Supply Chain, так как у него свой тег
                    sources.add("Vector")

            # 3. Supply Chain (Скрипт 4)
            if "Supply Chain" in prime_src:
                sources.add("SupplyChain")

            # 4. News (Скрипт 5)
            if "Reuters" in prime_src or "News" in prime_src:
                sources.add("News")

            # --- АНАЛИЗ ЛОГОВ (ИСТОРИЯ ОБОГАЩЕНИЯ) ---
            if record['log']:
                for entry_str in record['log']:
                    try:
                        # Пробуем распарсить JSON
                        if isinstance(entry_str, str):
                            entry = json.loads(entry_str)
                        else:
                            entry = entry_str  # Если вдруг драйвер вернул уже dict

                        src = entry.get('source', '')

                        if "Wiki" in src or "GDELT" in src:
                            sources.add("Wiki")

                        # Исправленная проверка для Скрипта 3
                        if "Vector" in src or "Inference" in src:
                            if "Supply Chain" not in src:
                                sources.add("Vector")

                        if "Supply Chain" in src:
                            sources.add("SupplyChain")

                        if "Reuters" in src or "News" in src:
                            sources.add("News")
                    except:
                        pass

            # --- ПОДСЧЕТ ИТОГОВ ДЛЯ ЭТОЙ СВЯЗИ ---
            if len(sources) >= 2:
                stats["Hybrid (Verified by 2+ sources)"] += 1
                # Можно раскомментировать для отладки, чтобы видеть примеры гибридов:
                # print(f"Hybrid Found! Sources: {sources}")
            elif "Vector" in sources:
                stats["Implicit (Vector Competitors)"] += 1
            elif "SupplyChain" in sources:
                stats["Implicit (Supply Chain Partners)"] += 1
            elif "News" in sources:
                stats["Events (Reuters News)"] += 1
            elif "Wiki" in sources:
                stats["Explicit (Wiki/Text)"] += 1

    # --- ВЫВОД В КОНСОЛЬ ---
    print(f"  Всего узлов: {n_total}")
    print(f"  Всего связей: {r_total}")
    print("-" * 80)
    print(f"  1. ФИНАНСОВЫЙ СЛОЙ (Владение, Yahoo): {r_financial}")
    print("-" * 80)
    print(f"  2. ИНТЕЛЛЕКТУАЛЬНЫЙ СЛОЙ (Бизнес-связи): {r_biz_total}")
    print("-" * 80)

    print(f"  {'МЕТОД ОБНАРУЖЕНИЯ':<40} | {'КОЛ-ВО':<10} | {'ДОЛЯ':<10}")
    print("-" * 80)

    ordered_keys = [
        "Implicit (Supply Chain Partners)",
        "Implicit (Vector Competitors)",
        "Explicit (Wiki/Text)",
        "Events (Reuters News)",
        "Hybrid (Verified by 2+ sources)"
    ]

    for key in ordered_keys:
        val = stats.get(key, 0)
        percent = (val / r_biz_total) * 100 if r_biz_total > 0 else 0
        print(f"  {key:<40} | {val:>10} | {percent:>9.1f}%")

    print("=" * 80)

    ai_total = stats["Implicit (Supply Chain Partners)"] + stats["Implicit (Vector Competitors)"] + stats[
        "Hybrid (Verified by 2+ sources)"]

    print(f"\n🏆 ВЫВОДЫ:")
    print(f"1. AI-алгоритмы сгенерировали и подтвердили {ai_total} связей.")
    print(
        f"2. Supply Chain ({stats['Implicit (Supply Chain Partners)']}) и Vectors ({stats['Implicit (Vector Competitors)']}) доминируют в графе.")
    print(
        f"3. {stats['Hybrid (Verified by 2+ sources)']} связей имеют перекрестное подтверждение (Высокая надежность).")


if __name__ == "__main__":
    analyze_focused()
    driver.close()