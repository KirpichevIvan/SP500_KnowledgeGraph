import pandas as pd
import yfinance as yf
import requests
import io
import time
import json
import os

if not os.path.exists('data'):
    os.makedirs('data')


def get_mega_dataset():
    print("Сбор данных с Википедии и Yahoo...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0'}

    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))

        sp500_df = None
        for t in tables:
            if 'Symbol' in t.columns and 'Security' in t.columns:
                sp500_df = t
                break

        if sp500_df is None:
            print("Не нашли таблицу в Википедии")
            return

    except Exception as e:
        print(f"Ошибка Википедии: {e}")
        return

    sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)
    tickers_list = sp500_df.to_dict('records')

    tickers_list = tickers_list[:]

    results = []
    print(f"Найдено {len(tickers_list)} компаний. Начинаем парсинг...")

    for i, item in enumerate(tickers_list):
        ticker = item['Symbol']
        wiki_name = item['Security']
        wiki_sector = item.get('GICS Sector', 'N/A')
        wiki_industry = item.get('GICS Sub-Industry', 'N/A')

        print(f"[{i + 1}/{len(tickers_list)}] {ticker}...", end=" ")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            officers = info.get('companyOfficers', [])
            clean_officers = []
            for p in officers:
                clean_officers.append({
                    'name': p.get('name'),
                    'title': p.get('title'),
                    'age': p.get('age')
                })
            officers_json = json.dumps(clean_officers, ensure_ascii=False)

            holders_json = "[]"
            try:
                holders_df = stock.institutional_holders
                if holders_df is not None and not holders_df.empty:
                    if 'Date Reported' in holders_df.columns:
                        holders_df['Date Reported'] = holders_df['Date Reported'].astype(str)

                    holders_json = json.dumps(holders_df.head(5).to_dict(orient='records'), ensure_ascii=False)
            except:
                pass

            address_data = {
                'city': info.get('city', 'N/A'),
                'state': info.get('state', 'N/A'),
                'zip': info.get('zip', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            address_json = json.dumps(address_data, ensure_ascii=False)

            data = {
                'Ticker': ticker,
                'Name': info.get('shortName', wiki_name),
                'Sector': wiki_sector,
                'Industry': wiki_industry,
                'Market Cap': info.get('marketCap', 'N/A'),
                'Employees': info.get('fullTimeEmployees', 'N/A'),
                'Website': info.get('website', 'N/A'),
                'Description': info.get('longBusinessSummary', 'Description not found'),
                'Officers_JSON': officers_json,
                'Holders_JSON': holders_json,
                'Address_JSON': address_json
            }
            results.append(data)
            print("OK")

        except Exception as e:
            print(f"Error: {e}")
            results.append({'Ticker': ticker, 'Name': wiki_name, 'Description': str(e)})

        time.sleep(0.3)

    df = pd.DataFrame(results)
    df.to_excel('data/sp500_graph_ready.xlsx', index=False)
    print("\nГотово! Полный файл сохранен в data/sp500_graph_ready.xlsx")


if __name__ == '__main__':
    get_mega_dataset()