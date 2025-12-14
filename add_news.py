import pandas as pd
import os
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# Настройки
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = ChatOllama(model="qwen3:8b", temperature=0)
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


NEWS_CSV_PATH = "../data/classified_reuters_news_mapped.csv"  # Укажите путь к вашему файлу

def ingest_and_link_news():
    print("📰 Начинаем загрузку новостей...")
    
    # 1. Чтение CSV (обычно новости разделены запятой, но проверим)
    try:
        df = pd.read_csv(NEWS_CSV_PATH)
    except:
        df = pd.read_csv(NEWS_CSV_PATH, sep=';')
    
    df = df.fillna("") # Убираем NaN, чтобы не ломать Cypher
    
    print(f"  Найдено {len(df)} записей. Загрузка в Neo4j...")

    with driver.session() as session:
        # 2. Создаем упрощенные имена компаний для поиска (если их еще нет)
        # Это нужно, чтобы найти "Apple" в заголовке "Apple releases new iPhone",
        # даже если в базе компания называется "Apple Inc."
        print("  🧹 Подготовка имен компаний для поиска...")
        session.run("""
            MATCH (c:Company)
            WHERE c.commonName IS NULL
            WITH c, c.name as original
            
            // Замена LET на WITH ... AS ... для совместимости с Cypher 5
            WITH c, replace(replace(replace(original, ' Inc.', ''), ' Corp.', ''), ' Corporation', '') AS clean
            WITH c, replace(replace(clean, ' Ltd.', ''), ' Group', '') AS clean2
            
            SET c.commonName = trim(clean2)
        """)
        # 3. Загрузка новостей и привязка к Индустрии
        # Мы используем MERGE для новостей, чтобы избежать дублей по заголовку
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting News"):
            query = """
            MERGE (n:News {headline: $headline})
            SET n.date = $date, 
                n.description = $desc,
                n.full_text = $headline + '\n' + $desc  // Поле для будущего эмбеддинга
            
            // Связь с Индустрией (если указана)
            WITH n
            MATCH (i:Industry {name: $industry_name}) 
            // Используем MATCH, чтобы привязаться только к существующим индустриям из вашего графа
            MERGE (n)-[:RELATES_TO_INDUSTRY]->(i)
            """
            
            session.run(query, 
                        headline=row['Headlines'],
                        date=row['Time'],
                        desc=row['Description'],
                        industry_name=row['GICS_Subsectors_Mapped'])

        # 4. Линковка новостей с Компаниями (Эвристика)
        # Ищем упоминание commonName компании в заголовке новости
        print("  🔗 Создание связей News -> Company...")
        link_query = """
            MATCH (n:News)
            WHERE not (n)-[:MENTIONS]->(:Company) // Только необработанные
            MATCH (c:Company)
            WHERE size(c.commonName) > 2 // Игнорируем слишком короткие названия во избежание шума
            
            AND toLower(n.headline) CONTAINS toLower(c.commonName)
            
            MERGE (n)-[:MENTIONS]->(c)
            RETURN count(*) as count
        """
        result = session.run(link_query)
        links = result.single()['count']
        print(f"  ✅ Создано {links} связей между новостями и компаниями.")

def prepare_unified_search_index():
    """
    Чтобы LangChain искал И по компаниям, И по новостям,
    мы добавим им общую метку :Searchable и общее поле text.
    """
    print("🔄 Подготовка единого поискового индекса...")
    with driver.session() as session:
        # Подготовка Компаний
        session.run("""
            MATCH (c:Company)
            SET c:Searchable
            // Формируем текст для поиска: Имя + Описание + Сектор
            SET c.search_text = "Company: " + c.name + "\nDescription: " + c.description
        """)
        
        # Подготовка Новостей
        session.run("""
            MATCH (n:News)
            SET n:Searchable
            // Формируем текст для поиска: Дата + Заголовок + Текст
            SET n.search_text = "News Date: " + toString(n.date) + "\nHeadline: " + n.headline + "\nContent: " + n.description
        """)
    print("✅ Метки :Searchable расставлены.")

# --- ЗАПУСК НОВЫХ ФУНКЦИЙ ---
ingest_and_link_news()
prepare_unified_search_index()

# --- ОБНОВЛЕННЫЙ ВЕКТОРНЫЙ ПОИСК ---
print("⏳ Пересоздание векторного хранилища для поиска по всему графу...")

# Теперь мы ищем по метке Searchable, которая есть и у Компаний, и у Новостей
vector_store = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="unified_knowledge_index", # Даем новое имя, чтобы не конфликтовать со старым
    node_label="Searchable",              # <--- Ищем по общей метке
    text_node_properties=["search_text"], # <--- Общее поле, которое мы создали выше
    embedding_node_property="embedding",
    
    # Кастомный запрос возврата, чтобы понимать, что мы нашли (новость или компанию)
    retrieval_query="""
    RETURN
        node.search_text as text,
        score,
        {
            type: head(labels(node)), 
            name: coalesce(node.name, node.headline),
            date: node.date
        } AS metadata
    """
)

print("🎉 Все готово! Граф содержит компании, индустрии, секторы и новости.")