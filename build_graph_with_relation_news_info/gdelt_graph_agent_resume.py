import argparse
import json
import os
from typing import Dict, List, Tuple

from gdelt_graph_agent import ENTITY_ORDER, GdeltEntityNewsAgent


class GdeltEntityNewsResumeRunner:
    def __init__(self, agent: GdeltEntityNewsAgent):
        self.agent = agent

    @staticmethod
    def _slice_entities(
        entities: Dict[str, List[str]],
        start_type: str,
        start_entity: str,
        start_after: bool,
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        if start_type not in ENTITY_ORDER:
            raise ValueError(f"Unknown start_type={start_type}. Allowed: {ENTITY_ORDER}")

        sliced = {k: [] for k in ENTITY_ORDER}
        type_started = False

        for etype in ENTITY_ORDER:
            names = entities.get(etype, [])
            if not type_started:
                if etype != start_type:
                    continue
                type_started = True

                if not start_entity:
                    sliced[etype] = names
                    continue

                idx = GdeltEntityNewsResumeRunner._find_start_index(names, start_entity)
                sliced[etype] = names[idx + 1 :] if start_after else names[idx:]
            else:
                sliced[etype] = names

        counts = {k: len(v) for k, v in sliced.items()}
        return sliced, counts

    @staticmethod
    def _find_start_index(names: List[str], start_entity: str) -> int:
        if not names:
            raise ValueError("Entity list is empty for start type.")

        normalized = start_entity.strip().strip('"').lower()
        for i, name in enumerate(names):
            if name.strip().lower() == normalized:
                return i

        for i, name in enumerate(names):
            cand = name.strip().strip('"').lower()
            if normalized in cand or cand in normalized:
                print(f"[WARN] start_entity exact match not found, using nearest match: '{name}'")
                return i

        # fallback: найти ближайший по лексикографическому порядку, чтобы можно было продолжить
        lower_names = [n.lower() for n in names]
        for i, n in enumerate(lower_names):
            if n > normalized:
                print(f"[WARN] start_entity not found, using next entity in order: '{names[i]}'")
                return i

        raise ValueError(
            f"start_entity='{start_entity}' not found and no later entity exists. "
            f"Use --include-start with exact value from source."
        )

    def run_from(
        self,
        start_type: str,
        start_entity: str,
        start_after: bool = True,
    ):
        print("[1/4] Loading entities...")
        entities = self.agent.load_entities()

        print("[2/4] Slicing entity queue from checkpoint...")
        sliced_entities, sliced_counts = self._slice_entities(
            entities=entities,
            start_type=start_type,
            start_entity=start_entity,
            start_after=start_after,
        )

        total = sum(sliced_counts.values())
        print(f"[INFO] Remaining entities to process: {total}")
        for t in ENTITY_ORDER:
            if sliced_counts[t] > 0:
                print(f"   - {t}: {sliced_counts[t]}")

        print("[3/4] Fetching GDELT news for remaining entities...")
        self.agent._init_output_csv()
        summary = {
            "started_from": {
                "start_type": start_type,
                "start_entity": start_entity,
                "start_after": start_after,
            },
            "totals": {k: {"entities": len(v), "with_news": 0, "articles": 0} for k, v in sliced_entities.items()},
        }

        for entity_type in ENTITY_ORDER:
            for entity_name in sliced_entities[entity_type]:
                try:
                    rows = self.agent.fetch_news_for_entity(entity_name, entity_type)
                    self.agent._append_rows_csv(rows)
                    if rows:
                        summary["totals"][entity_type]["with_news"] += 1
                        summary["totals"][entity_type]["articles"] += len(rows)
                except Exception as e:
                    print(f"      [!] entity failed: {entity_type}/{entity_name}: {e}")

        print("[4/4] Writing summary JSON...")
        with open(self.agent.out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("Done.")
        print(f"CSV: {self.agent.out_csv_path}")
        print(f"Summary: {self.agent.out_json_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume GDELT entity news collection from checkpoint.")
    parser.add_argument("--start-type", default="Resource", choices=ENTITY_ORDER)
    parser.add_argument("--start-entity", default="installation materials")
    parser.add_argument("--include-start", action="store_true", help="Include checkpoint entity itself.")
    parser.add_argument("--entity-source", default=os.getenv("GDELT_ENTITY_SOURCE", "neo4j"), choices=["neo4j", "excel", "neo4j_first"])
    parser.add_argument("--require-neo4j", action="store_true")
    parser.add_argument("--out-csv", default="gdelt_entity_news_resume.csv")
    parser.add_argument("--out-json", default="gdelt_entity_news_resume_summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Only print the sliced queue, do not query GDELT.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    agent = GdeltEntityNewsAgent()
    agent.entity_source = args.entity_source
    agent.require_neo4j = args.require_neo4j

    if not os.path.isabs(args.out_csv):
        args.out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_csv))
    if not os.path.isabs(args.out_json):
        args.out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_json))

    agent.out_csv_path = args.out_csv
    agent.out_json_path = args.out_json

    runner = GdeltEntityNewsResumeRunner(agent)

    if args.dry_run:
        entities = agent.load_entities()
        _, counts = runner._slice_entities(
            entities=entities,
            start_type=args.start_type,
            start_entity=args.start_entity,
            start_after=not args.include_start,
        )
        print("Dry-run sliced queue:")
        for t in ENTITY_ORDER:
            if counts[t] > 0:
                print(f"  {t}: {counts[t]}")
        return

    runner.run_from(
        start_type=args.start_type,
        start_entity=args.start_entity,
        start_after=not args.include_start,
    )


if __name__ == "__main__":
    main()