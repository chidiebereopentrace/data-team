"""Tests for export graph node and plan policy gate."""
from __future__ import annotations

from ml.rag.chatbot.export_intent import EXPORT_UPGRADE_MESSAGE
from ml.rag.chatbot.graph import node_export
from ml.rag.chatbot.plan_policy import allows_export


def test_allows_export_only_premium_plans() -> None:
    assert allows_export("Agribusinesses")
    assert allows_export("Integrated")
    assert not allows_export("Free")
    assert not allows_export("Government")


def test_node_export_upgrade_when_disabled() -> None:
    out = node_export(
        {
            "export_intent": "csv",
            "export_enabled": False,
            "answer": "Maize yields rose in 2022.",
        }
    )
    assert out["artifacts"] == []
    assert EXPORT_UPGRADE_MESSAGE in out["answer"]


def test_node_export_no_intent() -> None:
    out = node_export({"export_enabled": True, "answer": "Hello"})
    assert out["artifacts"] == []
    assert "answer" not in out


def test_node_export_skips_csv_when_bq_prep_fails() -> None:
    answer = "no accurate CSV export can be generated from OpenTrace sources"
    out = node_export(
        {
            "export_intent": "csv",
            "export_enabled": True,
            "plan_type": "Agribusinesses",
            "query": (
                "Compare maize production in Kenya and Nigeria over the last five years, "
                "and give me a CSV export of the figures."
            ),
            "answer": answer,
            "bq_results": [
                {
                    "source": "bigquery",
                    "content": (
                        "[BQ no_valid_sql: All SQL attempts failed validation or execution]"
                    ),
                    "metadata": {
                        "status": "no_valid_sql",
                        "prep_error": (
                            "All SQL attempts failed validation or execution; "
                            "model=deepseek/deepseek-v4-flash-0731"
                        ),
                        "validation_failed": True,
                    },
                }
            ],
            "citations": [],
        }
    )
    assert out["artifacts"] == []
    merged = out.get("answer") or answer
    assert "Downloadable files" not in merged
    assert "prep_error" not in merged
