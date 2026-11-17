import json

from ml.rag.text_processors.news_collection_preprocessor import preprocess_news_collection
from ml.rag.text_processors.preprocess.engines import news


def test_snapshot_includes_news_already_in_ingest_manifest(tmp_path, monkeypatch):
    source = tmp_path / "news"
    source.mkdir()
    article = "Kenyan maize cooperatives expanded grain storage to reduce losses after harvesting."
    (source / "harvest.txt").write_text((article + " ") * 20, encoding="utf-8")
    manifest = {}
    monkeypatch.setattr(news, "load_manifest", lambda: manifest)
    monkeypatch.setattr(news, "save_manifest", lambda _manifest: None)
    indexed_chunks = news.preprocess_folder(source)
    assert indexed_chunks
    assert not news.preprocess_folder(source)

    output = tmp_path / "staged.jsonl"
    assert preprocess_news_collection(input_dir=source, output_path=output) == len(indexed_chunks)
    ids = [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()]
    assert ids == [chunk.id for chunk in indexed_chunks]
