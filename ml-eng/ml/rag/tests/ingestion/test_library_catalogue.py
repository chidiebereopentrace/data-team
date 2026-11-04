from __future__ import annotations

from pathlib import Path

from ml.rag.chatbot.corpus_catalog import formation_doc_kinds
from ml.rag.ingestion.library_catalogue import (
    CatalogueRecord,
    join_key,
    load_catalogue,
    route_library_pdf,
    split_catalogue_countries,
    stamp_library_chunk,
)
from ml.rag.text_processors.chunk_contract import document_id_from_path
from ml.rag.text_processors.preprocess.models import ChunkOutput

_CATALOGUE_PATH = "OpenTrace Agricultural Library/Production/Crops/maize/Guide (IITA, 2025).pdf"
_RELATIVE = "Production/Crops/maize/Guide (IITA, 2025).pdf"


def _record(**overrides: str | bool) -> CatalogueRecord:
    data: dict[str, str | bool] = {
        "path": _CATALOGUE_PATH,
        "title": "Guide for sustainable maize production in Ghana",
        "publisher": "IITA",
        "year": "2025",
        "language": "en",
        "url": "https://hdl.handle.net/10568/179060",
        "country": "Ghana",
        "doc_type": "manual_guide",
        "is_academic": False,
    }
    data.update(overrides)
    return CatalogueRecord(
        path=str(data["path"]),
        title=str(data["title"]),
        publisher=str(data["publisher"]),
        year=str(data["year"]),
        language=str(data["language"]),
        url=str(data["url"]),
        country=str(data["country"]),
        doc_type=str(data["doc_type"]),
        is_academic=bool(data["is_academic"]),
    )


def test_join_key_strips_library_root() -> None:
    assert join_key(_CATALOGUE_PATH) == join_key(_RELATIVE)


def test_country_split_keeps_commas_inside_one_name() -> None:
    assert split_catalogue_countries("Ghana; Nigeria") == ["Ghana", "Nigeria"]
    assert split_catalogue_countries("Congo, Democratic Republic of") == [
        "Democratic Republic of the Congo"
    ]


def test_load_catalogue_matches_cache_relative_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "OpenTrace Agricultural Library - Catalogue.csv"
    csv_path.write_text(
        "category,path,title,source_publisher,year,language,original_url,country,doc_type,is_academic\n"
        'Production,"OpenTrace Agricultural Library/Production/Crops/maize/Guide (IITA, 2025).pdf",'
        "Guide,IITA,2025,en,https://example.test,Ghana,manual_guide,N\n",
        encoding="utf-8",
    )
    loaded = load_catalogue(csv_path)
    assert join_key(_RELATIVE) in loaded
    assert loaded[join_key(_RELATIVE)].title == "Guide"


def test_routes_practice_textbook_and_academic() -> None:
    practice = route_library_pdf(_RELATIVE, _record())
    assert practice.target == "formation"
    assert practice.doc_kind == "agricultural_practise"
    assert practice.payload["commodity"] == "maize"
    assert practice.payload["value_chain_stage"] == "production"
    assert practice.payload["audience"] == "farmer"

    textbook = route_library_pdf(
        "Textbooks/Primary/General Agriculture/Science P1.pdf",
        None,
    )
    assert textbook.target == "formation"
    assert textbook.doc_kind == "textbook"
    assert textbook.payload["education_level"] == "primary"
    assert textbook.payload["audience"] == "student"
    assert textbook.payload["textbook_subject"] == "general_agriculture"

    tertiary = route_library_pdf("Textbooks/Tertiary/Soil Science/Manual.pdf", None)
    assert tertiary.payload["audience"] == "trainee_professional"

    academic = route_library_pdf(
        "Academic Publications/Training/Crops/General/ToT.pdf",
        _record(path="OpenTrace Agricultural Library/Academic Publications/Training/Crops/General/ToT.pdf", is_academic=True),
    )
    assert academic.target == "academic"
    assert academic.doc_kind == "academic_article"

    flagged = route_library_pdf(_RELATIVE, _record(is_academic=True))
    assert flagged.target == "academic"


def test_training_path_and_title_set_extension_audience() -> None:
    training = route_library_pdf("Training/Crops/maize/Facilitator guide.pdf", None)
    assert training.payload["audience"] == "extension_agent"
    assert training.payload["value_chain_stage"] == "training"

    titled = route_library_pdf(
        _RELATIVE,
        _record(title="Training of Trainers manual for maize"),
    )
    assert titled.payload["audience"] == "extension_agent"


def test_french_folder_sets_language_and_skips_topic() -> None:
    route = route_library_pdf(
        "Training/General/french/Agrodok 02.pdf",
        _record(
            path="OpenTrace Agricultural Library/Training/General/french/Agrodok 02.pdf",
            language="en",
            title="Agrodok 02",
            country="",
            year="n.d.",
        ),
    )
    assert route.payload["language"] == "fr"
    assert "training_topic" not in route.payload
    assert route.payload["commodity"] == "general"

    topic = route_library_pdf("Training/General/soil_water/Soil health.pdf", None)
    assert topic.payload["training_topic"] == "soil_water"
    assert topic.payload["audience"] == "extension_agent"


def test_stamp_recomputes_acf_provenance_from_catalogue() -> None:
    chunk = ChunkOutput(
        id="old",
        text="Maize yield increased by 12% in the trial.",
        metadata={
            "chunk_index": 0,
            "hierarchy_path": "body",
            "tier": 1,
            "data_level": "global",
            "as_of_date": "1999-01-01",
            "region": "old region",
            "source_id": "old-source",
            "finding": "keep this finding",
            "direction": "increasing",
            "published_at": "1999-01-01",
            "geo_countries": "Kenya",
            "geo_country_primary": "Kenya",
            "geo_scope": "country",
            "publication_year": "1999",
        },
    )
    _route, stamped = stamp_library_chunk(chunk, relative_path=_RELATIVE, record=_record())
    meta = stamped.metadata
    document_id = document_id_from_path(_CATALOGUE_PATH)
    assert meta["document_id"] == document_id
    assert stamped.id
    assert meta["tier"] == 2
    assert meta["data_level"] == "national"
    assert meta["as_of_date"] == "2025-01-01"
    assert meta["source_id"] == document_id
    assert meta["geo_country_primary"] == "Ghana"
    assert meta["finding"] == "keep this finding"
    assert meta["direction"] == "increasing"
    assert "published_at" not in meta


def test_formation_doc_kinds_add_textbooks_for_curriculum_questions() -> None:
    assert formation_doc_kinds("how do farmers grow maize") == ["agricultural_practise"]
    assert formation_doc_kinds("Show the grade 12 agriculture textbook") == [
        "agricultural_practise",
        "textbook",
    ]
