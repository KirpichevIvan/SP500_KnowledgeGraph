import pandas as pd
import os
import time
from tqdm import tqdm
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv

# --- КОНФИГУРАЦИЯ ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Путь к файлу с новостями
NEWS_CSV_PATH = "../data/classified_reuters_news_mapped.csv"

# Настройки семантического поиска
# Порог для Индустрий (например, "Software" ~ "Software Services")
INDUSTRY_SIMILARITY_THRESHOLD = 0.82 
# Порог для Компаний (строже, чтобы "Apple" не линковалась к новостям про еду)
COMPANY_SIMILARITY_THRESHOLD = 0.75

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# ==========================================
# 1. ОЧИСТКА
# ==========================================
def clean_graph():
    print("\n🧹 ЭТАП 1: Полная очистка новостей...")
    with driver.session() as session:
        # 1. Удаляем все новости и их связи
        session.run("MATCH (n:News) DETACH DELETE n")
        print("   ✅ Все узлы :News удалены.")
        
        # 2. Удаляем ошибочные индустрии (у которых нет связей с Секторами или Компаниями)
        # Это уберет мусор, если он остался от прошлых запусков
        result = session.run("""
            MATCH (i:Industry)
            WHERE NOT (i)--() 
            DELETE i
            RETURN count(i) as count
        """)
        count = result.single()['count']
        print(f"   ✅ Удалено {count} 'осиротевших' индустрий.")

# ==========================================
# 2. ПОДГОТОВКА ВЕКТОРОВ (ДЛЯ ЛИНКОВКИ)
# ==========================================
def prepare_internal_indexes():
    print("\n🧠 ЭТАП 2: Подготовка векторов для внутреннего поиска...")
    
    with driver.session() as session:
        # --- А. Индустрии ---
        print("   🔹 Векторизация Индустрий...")
        # Берем только те, у которых есть имя (существующие в графе)
        result = session.run("MATCH (i:Industry) WHERE i.name IS NOT NULL RETURN i.name as name")
        industries = [r["name"] for r in result]
        
        if industries:
            vectors = embeddings.embed_documents(industries)
            for name, vector in zip(industries, vectors):
                session.run("MATCH (i:Industry {name: $name}) SET i.embedding = $vec", name=name, vec=vector)
            
            # Создаем индекс
            dim = len(vectors[0])
            session.run(f"""
                CREATE VECTOR INDEX industry_name_index IF NOT EXISTS
                FOR (i:Industry) ON (i.embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
            """)
        
        # --- Б. Компании ---
        print("   🔹 Векторизация Компаний...")
        result = session.run("MATCH (c:Company) RETURN c.ticker as ticker, c.name as name")
        companies = [r for r in result]
        
        # Формируем строку: "Company: Microsoft, Ticker: MSFT" для лучшего поиска
        comp_texts = [f"Company: {c['name']}, Ticker: {c['ticker']}" for c in companies]
        
        if comp_texts:
            comp_vectors = embeddings.embed_documents(comp_texts)
            for r, vector in zip(companies, comp_vectors):
                session.run("MATCH (c:Company {ticker: $t}) SET c.company_embedding = $vec", t=r['ticker'], vec=vector)
            
            # Создаем индекс
            dim = len(comp_vectors[0])
            session.run(f"""
                CREATE VECTOR INDEX company_entity_index IF NOT EXISTS
                FOR (c:Company) ON (c.company_embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
            """)
            
    # Даем базе пару секунд на построение индексов
    time.sleep(2)
    print("   ✅ Индексы industry_name_index и company_entity_index готовы.")

# ==========================================
# 3. ЗАГРУЗКА И ПРИВЯЗКА К ИНДУСТРИЯМ
# ==========================================
def ingest_news():
    print("\n📰 ЭТАП 3: Загрузка новостей и семантическая привязка к индустриям...")
    
    # Чтение CSV
    try:
        df = pd.read_csv(NEWS_CSV_PATH)
    except:
        df = pd.read_csv(NEWS_CSV_PATH, sep=';')
    df = df.fillna("")
    
    # ОПТИМИЗАЦИЯ: Сначала найдем соответствия для всех уникальных категорий из CSV
    # Чтобы не дергать векторный поиск на каждой строке
    all_categories_raw = []
    for x in df['GICS_Subsectors_Mapped']:
        if x:
            all_categories_raw.extend([s.strip() for s in str(x).split(';') if s.strip()])
    
    unique_cats = list(set(all_categories_raw))
    print(f"   Найдено {len(unique_cats)} уникальных категорий в CSV. Ищем соответствия в графе...")
    
    cat_mapping = {} # {'CSV Category': 'Graph Industry Name'}
    
    with driver.session() as session:
        # Векторизуем уникальные категории
        if unique_cats:
            cat_vectors = embeddings.embed_documents(unique_cats)
            
            for cat_name, vector in zip(unique_cats, cat_vectors):
                # Ищем ближайшую индустрию
                res = session.run("""
                    CALL db.index.vector.queryNodes('industry_name_index', 1, $vector)
                    YIELD node, score
                    WHERE score >= $thresh
                    RETURN node.name as name
                """, vector=vector, thresh=INDUSTRY_SIMILARITY_THRESHOLD)
                
                match = res.single()
                if match:
                    cat_mapping[cat_name] = match['name']

    print(f"   Сопоставлено {len(cat_mapping)} из {len(unique_cats)} категорий.")

    # Загрузка новостей
    with driver.session() as session:
        count_linked = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
            # Разбиваем строку секторов
            raw_cats = str(row['GICS_Subsectors_Mapped'])
            csv_cats_list = [s.strip() for s in raw_cats.split(';') if s.strip()]
            
            # Превращаем список CSV-категорий в список имен из Графа (через наш маппинг)
            target_industries = [cat_mapping[c] for c in csv_cats_list if c in cat_mapping]
            # Убираем дубликаты
            target_industries = list(set(target_industries))
            
            query = """
            MERGE (n:News {headline: $headline})
            SET n.date = $date, 
                n.description = $desc,
                n.full_text = $headline + '\n' + $desc
            
            WITH n
            UNWIND $targets AS ind_name
            MATCH (i:Industry {name: ind_name})
            MERGE (n)-[:RELATES_TO_INDUSTRY]->(i)
            """
            
            session.run(query, 
                        headline=row['Headlines'], 
                        date=row['Time'], 
                        desc=row['Description'],
                        targets=target_industries)
            
            if target_industries:
                count_linked += 1
                
    print(f"   ✅ Новости загружены. Привязано к индустриям: {count_linked}")

# ==========================================
# 4. СЕМАНТИЧЕСКАЯ ПРИВЯЗКА К КОМПАНИЯМ
# ==========================================
def link_companies_semantic():
    print("\n🔗 ЭТАП 4: Семантическая привязка новостей к компаниям...")
    
    BATCH_SIZE = 100
    
    with driver.session() as session:
        # Получаем ID и заголовки новостей, у которых еще нет связи с компанией
        result = session.run("""
            MATCH (n:News) 
            WHERE NOT (n)-[:MENTIONS]->(:Company)
            RETURN elementId(n) as id, n.headline as headline
        """)
        news_items = [r for r in result]
        
        print(f"   Обработка {len(news_items)} новостей...")
        links_created = 0
        
        for i in tqdm(range(0, len(news_items), BATCH_SIZE), desc="Linking Companies"):
            batch = news_items[i:i+BATCH_SIZE]
            ids = [item['id'] for item in batch]
            headlines = [item['headline'] for item in batch]
            
            # Считаем вектора заголовков
            vectors = embeddings.embed_documents(headlines)
            
            for news_id, vector in zip(ids, vectors):
                # Ищем ближайшую компанию
                # Векторный индекс company_entity_index мы создали на этапе 2
                res = session.run("""
                    CALL db.index.vector.queryNodes('company_entity_index', 1, $vector)
                    YIELD node, score
                    WHERE score >= $thresh
                    RETURN node.ticker as ticker
                """, vector=vector, thresh=COMPANY_SIMILARITY_THRESHOLD)
                
                match = res.single()
                if match:
                    # Создаем связь
                    session.run("""
                        MATCH (n:News), (c:Company {ticker: $ticker})
                        WHERE elementId(n) = $nid
                        MERGE (n)-[:MENTIONS]->(c)
                    """, nid=news_id, ticker=match['ticker'])
                    links_created += 1
                    
    print(f"   ✅ Создано {links_created} связей News -> Company.")

# ==========================================
# 5. ФИНАЛЬНАЯ ПОДГОТОВКА ДЛЯ RAG
# ==========================================
def setup_rag():
    print("\n🚀 ЭТАП 5: Сборка единого индекса для RAG...")
    
    with driver.session() as session:
        # Расставляем метки и формируем текст для поиска
        print("   Обновление свойств search_text...")
        session.run("""
            MATCH (c:Company) SET c:Searchable 
            SET c.search_text = "Company: " + c.name + "\nDescription: " + c.description
        """)
        session.run("""
            MATCH (n:News) SET n:Searchable
            SET n.search_text = "News Date: " + toString(n.date) + "\nHeadline: " + n.headline + "\nContent: " + n.description
        """)
    
    print("   Генерация финального векторного хранилища (unified_knowledge_index)...")
    # Это создаст индекс по метке :Searchable и посчитает эмбеддинги для search_text
    # Может занять время, так как пересчитывает всё
    try:
        Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="unified_knowledge_index",
            node_label="Searchable",
            text_node_properties=["search_text"],
            embedding_node_property="embedding",
        )
        print("   ✅ Unified Index готов к работе!")
    except Exception as e:
        print(f"   ⚠️ Сообщение от индекса (обычно OK, если он обновляется): {e}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    clean_graph()               # 1. Удалить старое
    prepare_internal_indexes()  # 2. Создать индексы для линковки
    ingest_news()               # 3. Загрузить новости и линковать Индустрии
    link_companies_semantic()   # 4. Линковать Компании
    setup_rag()                 # 5. Подготовить для чат-бота
    
    print(f"\n🎉 ВСЕ ГОТОВО! Время выполнения: {(time.time() - start_time):.2f} сек.")