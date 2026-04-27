import argparse
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse

import pandas as pd


@dataclass
class CleanConfig:
    allowed_languages: Tuple[str, ...] = ("english",)
    min_title_len: int = 18
    min_text_len: int = 30
    similarity_threshold: float = 0.92
    dedup_url_scope: str = "entity"  # entity | global
    strip_tracking_query_params: bool = True


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "ref", "source"
}


def canonical_url(url: str, strip_tracking_query_params: bool = True) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    try:
        p = urlparse(url)
        netloc = p.netloc.lower().replace("www.", "")
        path = re.sub(r"/+", "/", p.path or "/").rstrip("/")
        query = ""
        if p.query:
            pairs = parse_qsl(p.query, keep_blank_values=True)
            if strip_tracking_query_params:
                pairs = [(k, v) for k, v in pairs if k.lower() not in TRACKING_PARAMS]
            if pairs:
                pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
                query = urlencode(pairs, doseq=True)
        return f"{netloc}{path}?{query}" if query else f"{netloc}{path}"
    except Exception:
        return url.lower()


def row_quality_ok(row: pd.Series, cfg: CleanConfig) -> bool:
    lang = str(row.get("language", "")).strip().lower()
    if cfg.allowed_languages and lang not in cfg.allowed_languages:
        return False

    title = str(row.get("news_title", "") or "").strip()
    snippet = str(row.get("news_snippet", "") or "").strip()
    url = str(row.get("news_url", "") or "").strip()

    if not url.startswith("http"):
        return False
    if len(title) < cfg.min_title_len:
        return False
    if len(f"{title} {snippet}".strip()) < cfg.min_text_len:
        return False
    return True


def is_near_duplicate(text_a: str, text_b: str, threshold: float) -> bool:
    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return True
    seq_ratio = SequenceMatcher(None, text_a, text_b).ratio()
    if seq_ratio >= threshold:
        return True

    # token-overlap помогает ловить варианты с разными хвостами/источниками
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return False
    jacc = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    return jacc >= 0.72


def clean_news(first_csv: str, second_csv: str, output_csv: str, cfg: CleanConfig) -> Dict[str, int]:
    frames: List[pd.DataFrame] = []
    for i, path in enumerate([first_csv, second_csv], start=1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input CSV not found: {path}")
        df = pd.read_csv(path)
        df["_source_order"] = i
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    required_cols = ["entity_type", "entity_name", "news_title", "news_snippet", "news_url", "language"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Missing required column '{col}' in inputs")

    df_all["_canonical_url"] = df_all["news_url"].map(
        lambda u: canonical_url(u, strip_tracking_query_params=cfg.strip_tracking_query_params)
    )
    df_all["_title_norm"] = df_all["news_title"].map(normalize_text)
    df_all["_text_norm"] = (df_all["news_title"].fillna("") + " " + df_all["news_snippet"].fillna(""))\
        .map(normalize_text)

    kept_rows = []
    seen_urls = set()
    seen_urls_by_entity: Dict[Tuple[str, str], set] = {}
    # для near-duplicate храним только по одной сущности, чтобы не терять новости разных сущностей
    seen_text_by_entity: Dict[Tuple[str, str], List[str]] = {}

    dropped_non_english = 0
    dropped_low_quality = 0
    dropped_dup_url = 0
    dropped_dup_text = 0

    # сначала строки из первого CSV, затем из второго
    df_all = df_all.sort_values(by=["_source_order"], kind="stable")

    for _, row in df_all.iterrows():
        lang = str(row.get("language", "")).strip().lower()
        if cfg.allowed_languages and lang not in cfg.allowed_languages:
            dropped_non_english += 1
            continue

        if not row_quality_ok(row, cfg):
            dropped_low_quality += 1
            continue

        entity_key = (str(row.get("entity_type", "")), str(row.get("entity_name", "")))
        cu = row.get("_canonical_url", "")
        if cfg.dedup_url_scope == "global":
            is_dup_url = cu in seen_urls
        else:
            entity_seen = seen_urls_by_entity.setdefault(entity_key, set())
            is_dup_url = cu in entity_seen

        if is_dup_url:
            dropped_dup_url += 1
            continue

        text_norm = row.get("_text_norm", "")
        bucket = seen_text_by_entity.setdefault(entity_key, [])

        duplicate_found = False
        for old in bucket:
            if is_near_duplicate(text_norm, old, cfg.similarity_threshold):
                duplicate_found = True
                break

        if duplicate_found:
            dropped_dup_text += 1
            continue

        kept_rows.append(row)
        if cfg.dedup_url_scope == "global":
            seen_urls.add(cu)
        else:
            seen_urls_by_entity.setdefault(entity_key, set()).add(cu)
        bucket.append(text_norm)

    out_df = pd.DataFrame(kept_rows).drop(columns=[c for c in ["_source_order", "_canonical_url", "_title_norm", "_text_norm"] if c in df_all.columns], errors="ignore")
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    return {
        "input_rows": int(len(df_all)),
        "kept_rows": int(len(out_df)),
        "dropped_non_english": dropped_non_english,
        "dropped_low_quality": dropped_low_quality,
        "dropped_dup_url": dropped_dup_url,
        "dropped_dup_text": dropped_dup_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and clean two GDELT CSV files into a third one.")
    parser.add_argument("--first-csv", default="gdelt_entity_news.csv")
    parser.add_argument("--second-csv", default="gdelt_entity_news_resume.csv")
    parser.add_argument("--output-csv", default="gdelt_entity_news_cleaned.csv")
    parser.add_argument("--languages", default="english", help="Comma-separated allowed languages. Example: english")
    parser.add_argument("--min-title-len", type=int, default=18)
    parser.add_argument("--min-text-len", type=int, default=30)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--dedup-url-scope", choices=["entity", "global"], default="entity",
                        help="URL dedup scope. 'entity' is safer and keeps same URL for different entities.")
    parser.add_argument("--keep-query", action="store_true",
                        help="Keep full query params in URL canonicalization (less aggressive URL dedup).")
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    def resolve_input_path(path: str) -> str:
        """Resolve input path robustly for PyCharm/cwd/script-dir execution."""
        if os.path.isabs(path):
            return path

        candidates = [
            os.path.abspath(path),  # относительно cwd
            os.path.abspath(os.path.join(project_dir, path)),  # относительно корня проекта
            os.path.abspath(os.path.join(base_dir, path)),  # относительно папки скрипта
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # fallback для понятной ошибки в clean_news()
        return candidates[0]

    def resolve_output_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        # output по умолчанию в корень проекта, а не рядом со скриптом
        return os.path.abspath(os.path.join(project_dir, path))

    cfg = CleanConfig(
        allowed_languages=tuple(x.strip().lower() for x in args.languages.split(",") if x.strip()),
        min_title_len=args.min_title_len,
        min_text_len=args.min_text_len,
        similarity_threshold=args.sim_threshold,
        dedup_url_scope=args.dedup_url_scope,
        strip_tracking_query_params=not args.keep_query,
    )

    stats = clean_news(
        first_csv=resolve_input_path(args.first_csv),
        second_csv=resolve_input_path(args.second_csv),
        output_csv=resolve_output_path(args.output_csv),
        cfg=cfg,
    )

    print("Cleaning finished:")
    print(f"  dedup_url_scope: {cfg.dedup_url_scope}")
    print(f"  strip_tracking_query_params: {cfg.strip_tracking_query_params}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  output_csv: {resolve_output_path(args.output_csv)}")


if __name__ == "__main__":
    main()