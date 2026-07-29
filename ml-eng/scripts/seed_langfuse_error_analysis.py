#!/usr/bin/env python3
"""Seed diverse rag.query traces for Langfuse error analysis.

Usage (from ml-eng/):
  PYTHONPATH=. python scripts/seed_langfuse_error_analysis.py
  PYTHONPATH=. python scripts/seed_langfuse_error_analysis.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.rag.local_env import load_rag_dotenv
from ml.rag.observability import flush_langfuse, rag_trace_context

# Stratified seed set (meta / product / full_rag / geo / compare / free-tier tone)
SEED_CASES: list[dict] = [
    {"query": "Who are you?", "plan_type": "Free", "category": "Farmers", "tag": "meta"},
    {"query": "What is OpenTrace?", "plan_type": "Free", "category": "Government", "tag": "product"},
    {"query": "What is Ask ADZA?", "plan_type": "Integrated", "category": "NGOs", "tag": "product"},
    {"query": "Maize yields in Kenya 2020", "plan_type": "Government", "category": "Government", "tag": "full_rag"},
    {"query": "Rice production trends in Nigeria since 2015", "plan_type": "Government", "category": "Government", "tag": "full_rag"},
    {"query": "Rainfall patterns in Ghana for cocoa farmers", "plan_type": "Farmers", "category": "Farmers", "tag": "full_rag"},
    {"query": "Compare maize production in Kenya and Tanzania 2018-2022", "plan_type": "Agribusinesses", "category": "Agribusinesses", "tag": "compare"},
    {"query": "Food security indicators in Ethiopia last five years", "plan_type": "NGOs", "category": "NGOs", "tag": "full_rag"},
    {"query": "Sorghum prices volatility in West Africa", "plan_type": "Agribusinesses", "category": "Agribusinesses", "tag": "full_rag"},
    {"query": "What crops grow best in northern Uganda?", "plan_type": "Farmers", "category": "Farmers", "tag": "full_rag"},
    {"query": "Cassava production in Côte d'Ivoire 2019", "plan_type": "Integrated", "category": "Government", "tag": "full_rag"},
    {"query": "Climate risks for wheat in Morocco", "plan_type": "NGOs", "category": "NGOs", "tag": "full_rag"},
    {"query": "Hello", "plan_type": "Free", "category": "Farmers", "tag": "meta"},
    {"query": "Summarize OpenTrace data sources", "plan_type": "Integrated", "category": "Agribusinesses", "tag": "product"},
    {"query": "Bean yields in Rwanda 2021 versus 2022", "plan_type": "Government", "category": "Government", "tag": "full_rag"},
    {"query": "Livestock market trends in Senegal", "plan_type": "Agribusinesses", "category": "Agribusinesses", "tag": "full_rag"},
    {"query": "Irrigation and maize in Zambia", "plan_type": "Farmers", "category": "Farmers", "tag": "full_rag"},
    {"query": "Compare coffee exports Kenya vs Ethiopia", "plan_type": "Integrated", "category": "Agribusinesses", "tag": "compare"},
    {"query": "Nutrition outcomes and drought in Niger", "plan_type": "NGOs", "category": "NGOs", "tag": "full_rag"},
    {"query": "Groundnut production Malawi 2020", "plan_type": "Free", "category": "Farmers", "tag": "full_rag"},
    {"query": "Who built Ask ADZA?", "plan_type": "Free", "category": "Government", "tag": "meta"},
    {"query": "Fertilizer subsidy impact in Nigeria", "plan_type": "Government", "category": "Government", "tag": "full_rag"},
    {"query": "Palm oil supply risk Ghana and Côte d'Ivoire", "plan_type": "Agribusinesses", "category": "Agribusinesses", "tag": "compare"},
    {"query": "Tea yields in Kenya highlands", "plan_type": "Farmers", "category": "Farmers", "tag": "full_rag"},
    {"query": "Youth employment in agri-value chains East Africa", "plan_type": "NGOs", "category": "NGOs", "tag": "full_rag"},
    {"query": "Maize self-sufficiency South Africa", "plan_type": "Integrated", "category": "Government", "tag": "full_rag"},
    {"query": "Drought early warning Mozambique", "plan_type": "Government", "category": "Government", "tag": "full_rag"},
    {"query": "Sesame export opportunities Sudan", "plan_type": "Agribusinesses", "category": "Agribusinesses", "tag": "full_rag"},
    {"query": "Potato farming tips for Kenya smallholders", "plan_type": "Farmers", "category": "Farmers", "tag": "full_rag"},
    {"query": "How do I use OpenTrace for policy briefs?", "plan_type": "Integrated", "category": "Government", "tag": "product"},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    load_rag_dotenv(_REPO)
    from ml.rag.chatbot.graph import run_rag

    cases = SEED_CASES[args.offset : args.offset + args.limit]
    session = f"ea-seed-{uuid.uuid4().hex[:10]}"
    print(f"Seeding {len(cases)} traces session={session}")

    ok = 0
    for i, case in enumerate(cases, 1):
        q = case["query"]
        tags = ["error_analysis_seed", f"seed:{case['tag']}", f"plan_type:{case['plan_type']}"]
        print(f"[{i}/{len(cases)}] {case['tag']} | {q[:60]!r}")
        t0 = time.perf_counter()
        try:
            with rag_trace_context(
                trace_name="rag.query",
                session_id=f"{session}",
                plan_type=case["plan_type"],
                category=case["category"],
                trace_input={"query": q, "seed_tag": case["tag"]},
                tags=tags,
            ) as handle:
                result = run_rag(
                    q,
                    plan_type=case["plan_type"],
                    category=case["category"],
                    session_id=f"{session}-{i}",
                    trace_tags=tags,
                )
                handle.update_output(result)
            flush_langfuse()
            ms = (time.perf_counter() - t0) * 1000
            route = "meta" if result.get("is_meta_query") else (
                "product" if result.get("is_product_query") else "full_rag"
            )
            err = result.get("error")
            print(f"  ok route={route} ms={ms:.0f} err={err!r} ans_len={len(str(result.get('answer') or ''))}")
            ok += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            flush_langfuse()

    print(f"Done: {ok}/{len(cases)} succeeded")
    flush_langfuse()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
