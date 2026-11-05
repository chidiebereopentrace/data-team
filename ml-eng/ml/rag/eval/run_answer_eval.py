"""End-to-end answer benchmark.

Questions are built from the query-form catalog, filled with country, indicator,
and year combinations the mart holds, plus report-only and multilingual
questions. Each question runs through the full graph.

Usage:
  cd ml-eng && python -m ml.rag.eval.run_answer_eval --limit 12
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.rag.local_env import load_rag_dotenv

load_rag_dotenv(Path(__file__).resolve().parents[3])

from ml.rag.chatbot.agri_measure_ontology import MEASURES
from ml.rag.chatbot.generator import is_usable_structured_bq_row
from ml.rag.chatbot.graph import run_rag
from ml.rag.chatbot.mart_coverage import CoverageEntry, load_coverage
from ml.rag.chatbot.numeric_grounding import unsourced_figures
from ml.rag.chatbot.query_forms import (
    QUERY_FORMS,
    QueryForm,
    indicator_tables,
    named_sample,
)

_REPORTS = Path(__file__).resolve().parent / "reports"

_MULTILINGUAL: tuple[tuple[str, str], ...] = (
    ("fr", "quelle est la production de maïs au {country} en {year}"),
    ("fr", "évolution du prix du riz au {country} de {year} à {year_end}"),
    ("sw", "uzalishaji wa mahindi nchini {country} mwaka {year} ulikuwa kiasi gani"),
    ("sw", "bei ya mchele nchini {country} kutoka {year} hadi {year_end}"),
    ("ha", "nawa ne yawan masarar da aka noma a {country} a shekarar {year}"),
    ("ar", "ما هو إنتاج الذرة في {country} في عام {year}"),
)

_CROP_BY_INDICATOR = {
    "production": "maize",
    "yield": "maize",
    "trade": "maize",
    "market_price": "maize",
    "food_balance": "wheat",
}


@dataclass
class Question:
    question: str
    kind: str
    lang: str
    form_id: str
    indicator: str
    country: str
    held: bool


@dataclass
class Outcome:
    question: str
    kind: str
    lang: str
    form_id: str
    indicator: str
    held: bool
    answered: bool
    warehouse_hit: bool
    cited: int
    unsourced: list[str]
    invented_on_miss: bool
    latency_s: float
    error: str


def _indicator_label(indicator: str) -> str:
    """The first alias a person would type: several words, no underscores."""
    spec = MEASURES.get(indicator)
    aliases = [str(a) for a in (spec.aliases if spec else ()) if "_" not in str(a)]
    worded = [a for a in aliases if " " in a]
    if worded:
        return worded[0]
    if aliases:
        return aliases[0]
    return indicator.replace("_", " ")


def _fill(form: QueryForm, entry: CoverageEntry, indicator: str) -> str:
    sample = named_sample(entry.table_id) if form.named_discriminator else None
    return form.render(
        country=entry.country,
        year=entry.max_year if form.year_shape == "single_year" else max(entry.min_year, entry.max_year - 4),
        year_end=entry.max_year,
        indicator=_indicator_label(indicator),
        crop=_CROP_BY_INDICATOR.get(indicator, "maize"),
        discriminator=sample[1].replace("_", " ") if sample else "",
    ).replace("  ", " ").strip()


def build_questions(limit: int) -> list[Question]:
    coverage = load_coverage()
    by_table = {
        entry.table_id: entry
        for entry in sorted(coverage.entries, key=lambda e: (e.max_year, e.max_year - e.min_year))
        if entry.iso3
    }
    if not by_table:
        raise SystemExit("No mart coverage artifact. Run: python -m ml.rag.chatbot.mart_coverage --build")
    pairs = [(ind, tid) for ind, tid in indicator_tables() if tid in by_table]
    questions: list[Question] = []
    sql_forms = [form for form in QUERY_FORMS if form.shape != "report"]
    report_forms = [form for form in QUERY_FORMS if form.shape == "report"]
    for index, form in enumerate(sql_forms):
        if not pairs:
            break
        indicator, table_id = pairs[index % len(pairs)]
        entry = by_table[table_id]
        questions.append(
            Question(
                question=_fill(form, entry, indicator),
                kind=form.shape,
                lang="en",
                form_id=form.form_id,
                indicator=indicator,
                country=entry.country,
                held=True,
            )
        )
    for index, form in enumerate(report_forms):
        if not pairs:
            break
        indicator, table_id = pairs[index % len(pairs)]
        entry = by_table[table_id]
        questions.append(
            Question(
                question=_fill(form, entry, indicator),
                kind="report",
                lang="en",
                form_id=form.form_id,
                indicator=indicator,
                country=entry.country,
                held=False,
            )
        )
    production_tables = {t.split(".")[-1].lower() for t in MEASURES["production"].candidate_tables}
    production = next(
        (e for e in sorted(coverage.entries, key=lambda e: -e.max_year) if e.table_id in production_tables and e.iso3),
        None,
    )
    if production is not None:
        for index, (lang, template) in enumerate(_MULTILINGUAL):
            questions.append(
                Question(
                    question=template.format(
                        country=production.country,
                        year=production.max_year if "{year_end}" not in template
                        else max(production.min_year, production.max_year - 4),
                        year_end=production.max_year,
                    ),
                    kind="multilingual",
                    lang=lang,
                    form_id=f"ml-{index:02d}",
                    indicator="production",
                    country=production.country,
                    held=True,
                )
            )
    if limit > 0:
        multilingual = [q for q in questions if q.lang != "en"]
        english = [q for q in questions if q.lang == "en"]
        ml_take = min(len(multilingual), max(2, limit // 6))
        en_take = limit - ml_take
        stride = max(1, len(english) // max(1, en_take))
        questions = english[::stride][:en_take] + multilingual[:ml_take]
    return questions


def _evidence_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    items = [i for i in (result.get("reranked_context") or []) if isinstance(i, dict)]
    items.extend(i for i in (result.get("bq_results") or []) if isinstance(i, dict))
    return items


def run_one(question: Question) -> Outcome:
    t0 = time.perf_counter()
    error = ""
    try:
        result = run_rag(question.question)
    except Exception as exc:
        result = {}
        error = str(exc)[:300]
    latency = time.perf_counter() - t0
    answer = str(result.get("answer") or "")
    citations = [c for c in (result.get("citations") or []) if isinstance(c, dict)]
    rows = [r for r in (result.get("bq_results") or []) if isinstance(r, dict)]
    warehouse_hit = any(is_usable_structured_bq_row(r) for r in rows)
    answered = bool(answer.strip()) and (warehouse_hit or bool(citations))
    unsourced = unsourced_figures(answer, _evidence_items(result)) if answer else []
    invented_on_miss = not warehouse_hit and not citations and bool(unsourced_figures(answer, [{"content": question.question}]))
    return Outcome(
        question=question.question,
        kind=question.kind,
        lang=question.lang,
        form_id=question.form_id,
        indicator=question.indicator,
        held=question.held,
        answered=answered,
        warehouse_hit=warehouse_hit,
        cited=len(citations),
        unsourced=unsourced,
        invented_on_miss=invented_on_miss,
        latency_s=round(latency, 2),
        error=error,
    )


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    held = [o for o in outcomes if o.held]
    english = [o for o in outcomes if o.lang == "en"]
    multilingual = [o for o in outcomes if o.lang != "en"]
    latencies = sorted(o.latency_s for o in outcomes)
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    return {
        "questions": len(outcomes),
        "answer_rate_held": _rate([o.answered for o in held]),
        "warehouse_hit_rate_held": _rate([o.warehouse_hit for o in held if o.kind != "report"]),
        "grounded_rate": _rate([not o.unsourced for o in outcomes]),
        "invented_on_miss": sum(o.invented_on_miss for o in outcomes),
        "answer_rate_en": _rate([o.answered for o in english]),
        "answer_rate_multilingual": _rate([o.answered for o in multilingual]),
        "latency_p95_s": p95,
        "latency_median_s": statistics.median(latencies) if latencies else 0.0,
        "errors": sum(bool(o.error) for o in outcomes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end RAG answer benchmark")
    parser.add_argument("--limit", type=int, default=0, help="sample this many questions (0 = all)")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    questions = build_questions(args.limit)
    outcomes: list[Outcome] = []
    for index, question in enumerate(questions, 1):
        outcome = run_one(question)
        outcomes.append(outcome)
        print(
            f"[{index}/{len(questions)}] {'ANS' if outcome.answered else 'GAP'} "
            f"bq={int(outcome.warehouse_hit)} cited={outcome.cited} {outcome.latency_s}s :: {question.question[:70]}",
            flush=True,
        )
    summary = summarize(outcomes)
    _REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = _REPORTS / f"answer_eval_{args.label}_{stamp}.json"
    path.write_text(
        json.dumps({"summary": summary, "outcomes": [asdict(o) for o in outcomes]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
