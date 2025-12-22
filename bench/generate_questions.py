import pandas as pd
import json
import random
import re

# Настройки
INPUT_FILE = '../data/sp500_graph_ready.csv'  # путь к файлу
OUTPUT_FILE = 'benchmark_qa.csv'
NUM_QUESTIONS = 100

def load_data(filepath):
    """Загрузка данных и первичная обработка."""
    try:
        df = pd.read_csv(filepath, sep=';')
    except:
        df = pd.read_csv(filepath, sep=',')
    return df

def safe_json_parse(json_str):
    """Безопасный парсинг JSON полей."""
    if pd.isna(json_str) or json_str == "":
        return None
    try:
        # Иногда в CSV JSON сохраняется с одинарными кавычками, что не валидно для json.loads
        # Это грубая замена, лучше иметь валидный JSON
        if "'" in json_str and '"' not in json_str:
             json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# --- Генераторы вопросов ---
# Каждая функция принимает строку (row) и возвращает кортеж (question, answer) или None

def gen_ticker_lookup(row):
    return (
        f"What is the stock ticker symbol for {row['Name']}?",
        row['Ticker']
    )

def gen_sector_lookup(row):
    return (
        f"In which sector does {row['Name']} operate?",
        row['Sector']
    )

def gen_industry_lookup(row):
    return (
        f"What industry is {row['Name']} associated with?",
        row['Industry']
    )

def gen_employees_count(row):
    if pd.isna(row['Employees']): return None
    return (
        f"How many full-time employees work for {row['Name']}?",
        str(row['Employees'])
    )

def gen_market_cap(row):
    if pd.isna(row['Market Cap']): return None
    return (
        f"What is the current market capitalization of {row['Ticker']}?",
        str(row['Market Cap'])
    )

def gen_ceo_lookup(row):
    """Ищет CEO в JSON офицеров"""
    officers = safe_json_parse(row.get('Officers_JSON'))
    if not officers: return None
    
    # Предполагаем структуру списка словарей: [{'title': '...', 'name': '...'}]
    # Адаптируйте ключи под ваш реальный JSON
    target_role = "CEO"
    ceo_name = None
    
    # Поиск по списку (упрощенная логика)
    if isinstance(officers, list):
        for officer in officers:
            title = officer.get('title', '').lower()
            if 'chief executive officer' in title or 'ceo' in title:
                ceo_name = officer.get('name')
                break
    
    if ceo_name:
        return (f"Who is the CEO of {row['Name']}?", ceo_name)
    return None

def gen_hq_location(row):
    """Ищет город/штат в Address JSON"""
    address = safe_json_parse(row.get('Address_JSON'))
    if not address: return None
    
    # Предполагаем структуру {'city': '...', 'state': '...'}
    city = address.get('city')
    state = address.get('state')
    
    if city and state:
        return (f"Where is the headquarters of {row['Name']} located?", f"{city}, {state}")
    elif city:
        return (f"In which city is {row['Name']} based?", city)
    return None

def gen_website_lookup(row):
    if pd.isna(row['Website']): return None
    return (
        f"What is the official website for {row['Name']}?",
        row['Website']
    )

def gen_description_check(row):
    """Вопрос на проверку описания"""
    if pd.isna(row['Description']): return None
    # Берем первое предложение для контекста
    desc_preview = row['Description'].split('.')[0]
    return (
        f"Provide a brief description of what {row['Name']} does.",
        row['Description'] # Ответ - полное описание
    )

# --- Основной цикл ---

def main():
    df = load_data(INPUT_FILE)
    print(f"Загружено {len(df)} компаний.")

    # Список функций-генераторов
    generators = [
        gen_ticker_lookup,
        gen_sector_lookup,
        gen_industry_lookup,
        gen_employees_count,
        gen_market_cap,
        gen_hq_location,
        gen_website_lookup,
        gen_description_check
    ]

    benchmark_data = []
    
    # Генерируем вопросы пока не наберем 100
    attempts = 0
    while len(benchmark_data) < NUM_QUESTIONS and attempts < NUM_QUESTIONS * 10:
        attempts += 1
        
        # Случайная строка
        row = df.sample(1).iloc[0]
        
        # Случайный тип вопроса
        gen_func = random.choice(generators)
        
        result = gen_func(row)
        
        if result:
            question, answer = result
            
            # Проверка на дубликаты вопросов (опционально)
            if any(q['question'] == question for q in benchmark_data):
                continue
                
            # Формируем контекст для RAG (вся строка в текстовом виде)
            # Это поможет оценить, нашла ли LLM нужную информацию
            context = f"Company: {row['Name']}, Ticker: {row['Ticker']}, Sector: {row['Sector']}, Description: {row['Description']}"
            
            benchmark_data.append({
                'question': question,
                'ground_truth': answer,
                'question_type': gen_func.__name__,
                'ticker': row['Ticker'],
                'context': context # Полезно для дебага RAG
            })

    # Сохранение
    result_df = pd.DataFrame(benchmark_data)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Бенчмарк создан! Сохранено {len(result_df)} вопросов в '{OUTPUT_FILE}'.")
    
    # Пример вывода
    print("\nПримеры вопросов:")
    print(result_df[['question', 'ground_truth']].head())

if __name__ == "__main__":
    main()