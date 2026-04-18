import os
import json
import time
import random
import requests
import datetime
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

POLZA_KEY = os.getenv("POLZA_API_KEY")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
EXCEL_PATH = "../data/sp500_graph_ready.xlsx"

TEST_PAIRS = [
    ("Microsoft", "NVIDIA"),
    ("Visa", "Mastercard"),
    ("Apple", "Alphabet"),
    ("Amazon", "FedEx"),
    ("Chevron", "Exxon Mobil")
]

SUPPORTED_LABELS = ["Company", "Fund", "City", "Sector", "Product", "Resource"]


class GdeltAuditAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=POLZA_KEY,
            base_url="https://api.polza.ai/api/v1"
        )
        self.company_map = self._load_company_map()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _load_company_map(self):
        try:
            df = pd.read_excel(EXCEL_PATH)
            return pd.Series(df.Ticker.values, index=df.Name).to_dict()
        except Exception as e:
            print(f"Ошибка Excel: {e}")
            return {}

    def fetch_gdelt_joint(self, c1, c2):
        """Максимально консервативный запрос к GDELT"""
        query = f'"{c1}" "{c2}"'
        print(f"   GDELT Search: {query} ... ", end="", flush=True)

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "timespan": "6m",
            "maxrecords": 5
        }

        wait_time = random.uniform(12, 18)
        time.sleep(wait_time)

        try:
            resp = self.session.get(GDELT_URL, params=params, timeout=30)

            if resp.status_code == 429:
                print("429 (Заблокировано).")
                return []

            if resp.status_code != 200:
                print(f"Ошибка {resp.status_code}")
                return []

            data = resp.json()
            articles = data.get("articles", [])
            print(f"Найдено: {len(articles)}")
            return articles

        except Exception as e:
            print(f"Ошибка запроса: {e}")
            return []

    def analyze_llm(self, headline, snippet):
        prompt = f"""Analyze this news snippet about two S&P 500 companies.
        Headline: {headline}
        Snippet: {snippet}

        Extract JSON:
        {{
          "summary": "How exactly are these companies linked in this news?",
          "entities": [ {{"name": "...", "type": "Company/Product/Fund/City/Sector/Resource"}} ]
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model="qwen/qwen-2.5-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            txt = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            if "{" in txt:
                txt = txt[txt.find("{"):txt.rfind("}") + 1]
            return json.loads(txt)
        except:
            return {"summary": "Analysis failed", "entities": []}

    def try_match_company(self, name):
        """Проверка по Excel списку"""
        if name in self.company_map:
            return f"MATCH: {self.company_map[name]}"

        best_name = None
        best_score = -1
        for c_name, ticker in self.company_map.items():
            score = fuzz.token_sort_ratio(name.lower(), c_name.lower())
            if score > best_score:
                best_score = score
                best_name = ticker

        if best_score >= 88:
            return f"FUZZY: {best_name} ({best_score}%)"
        return "NEW"

    def run(self):
        final_results = {}

        for c1, c2 in TEST_PAIRS:
            print(f"\nПАРА: {c1} & {c2}")
            articles = self.fetch_gdelt_joint(c1, c2)

            pair_news_data = []
            for art in articles:
                title = art.get('title', '')
                snippet = art.get('snippet', '')
                print(f"      Analyzing: {title[:50]}...")

                analysis = self.analyze_llm(title, snippet)

                for ent in analysis.get('entities', []):
                    if ent['type'] == 'Company':
                        ent['match_status'] = self.try_match_company(ent['name'])

                pair_news_data.append({
                    "title": title,
                    "url": art.get('url'),
                    "summary": analysis.get('summary'),
                    "entities": analysis.get('entities')
                })

            final_results[f"{c1} + {c2}"] = pair_news_data

        with open("gdelt_audit_report.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 50)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 50)
        for pair, news in final_results.items():
            print(f"\n{pair}: {'✅' if news else 'нет новостей'}")
            for n in news:
                print(f"   - {n['title']}")
                print(f"     {n['url']}")
                print(f"     {n['summary']}")


if __name__ == "__main__":
    agent = GdeltAuditAgent()
    agent.run()