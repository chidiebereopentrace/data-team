"""GeoJSON exports preserve observations and never invent locations."""
import json
from datetime import date
from decimal import Decimal

import pytest

from ml.rag.chatbot.export_intent import detect_export_intent
from ml.rag.chatbot.exports.geojson_builder import build_geojson


def test_geojson_preserves_real_points_and_unlocated_records():
    data, name = build_geojson([
        {"latitude": -1.2921, "longitude": 36.8219, "value": Decimal("12.50"), "unit": "t/ha", "date": date(2020, 1, 1)},
        {"country_name": "Kenya", "value": 2, "unit": "t/ha"},
    ], filename="study.geojson")
    result = json.loads(data)
    assert name == "study.geojson" and result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2
    first, second = result["features"]
    assert first["geometry"] == {"type": "Point", "coordinates": [36.8219, -1.2921]}
    assert first["properties"]["unit"] == "t/ha" and first["properties"]["date"] == "2020-01-01"
    assert Decimal(str(first["properties"]["value"])) == Decimal("12.50")
    assert second["geometry"] is None and second["properties"]["country_name"] == "Kenya"


def test_existing_polygon_is_preserved():
    polygon = {"type": "Polygon", "coordinates": [[[36, -1], [37, -1], [37, 0], [36, -1]]]}
    data, _ = build_geojson([{"geometry": polygon, "value": 2, "source": "verified boundary"}])
    assert json.loads(data)["features"][0]["geometry"] == polygon


@pytest.mark.parametrize("row", [
    {"latitude": 91, "longitude": 20}, {"latitude": 0, "longitude": 181},
    {"latitude": float("nan"), "longitude": 20}, {"latitude": 1},
    {"geometry": {"type": "Point", "coordinates": [36, 200]}},
    {"latitude": True, "longitude": False},
    {"geometry": {"type": "Point", "coordinates": [False, True]}},
    {"geometry": {"type": "GeometryCollection", "geometries": [
        {"type": "Point", "coordinates": [1, 2], "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}},
    ]}},
])
def test_invalid_or_incomplete_geometry_fails_explicitly(row):
    with pytest.raises(ValueError):
        build_geojson([{**row, "value": 1}])


def test_geojson_intent_is_recognized_and_combines_with_csv():
    assert detect_export_intent("Export soil observations as GeoJSON") == "geojson"
    assert detect_export_intent("Download the data as CSV and GeoJSON") == "multi"


@pytest.mark.parametrize("plan", ["Research", "Integrated", "Agribusinesses"])
def test_shared_geojson_export_stores_and_refreshes_download(monkeypatch, tmp_path, plan):
    from pathlib import Path
    from urllib.parse import unquote, urlsplit
    from ml.rag.chatbot.artifact_storage import refresh_artifact_url
    from ml.rag.chatbot.export_runner import run_exports

    for name in ("AWS_S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "RAG_ARTIFACT_GCS_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_ARTIFACT_LOCAL_DIR", str(tmp_path))
    artifacts = run_exports(export_kind="geojson", query="Export observations as GeoJSON", answer="Verified data.",
        bq_results=[{"source": "bigquery", "metadata": {"raw_row": {"country_name": "Kenya", "value": 2, "unit": "t/ha"}}}],
        citations=[{"id": 1}], state={}, export_enabled=True, plan_type=plan)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["kind"] == "geojson" and artifact["mime_type"] == "application/geo+json"
    assert artifact["citation_ids"] == [1] and "null" in artifact["summary"]
    local_file = Path(unquote(urlsplit(artifact["url"]).path).lstrip("/"))
    assert json.loads(local_file.read_bytes())["features"][0]["geometry"] is None
    assert refresh_artifact_url(artifact["id"], artifact["filename"])["url"] == artifact["url"]


def test_research_perspective_cannot_bypass_export_gates():
    from ml.rag.chatbot.graph import node_export

    result = node_export({"export_intent": "geojson", "plan_type": "Free", "category": "Research",
                          "export_enabled": False, "answer": "Verified data."})
    assert result["artifacts"] == [] and "export" in result["answer"].lower()


def test_geojson_is_data_export_mode():
    from ml.rag.chatbot.task_mode import is_data_export_only_query

    assert is_data_export_only_query("Export soil observations as GeoJSON")


def test_report_package_keeps_explicit_geojson_format():
    from ml.rag.chatbot.export_runner import _expand_export_kinds

    query = "Export a full report and GeoJSON"
    intent = detect_export_intent(query)
    assert intent == "multi"
    assert "geojson" in _expand_export_kinds(intent, rows=[], analytical=False, query=query)


def test_geojson_dependency_is_available_to_each_deployment_image():
    from pathlib import Path

    ml_root = Path(__file__).resolve().parents[4]
    for manifest in ("requirements.txt", "requirements.railway.txt", "ml/rag/requirements.txt"):
        assert "geojson-pydantic>=2.1,<3" in (ml_root / manifest).read_text(encoding="utf-8")
