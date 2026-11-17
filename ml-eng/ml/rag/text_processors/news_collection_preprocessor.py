"""
Preprocess News corpus (.txt with YAML front matter, .docx) into chunk JSONL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.rag.paths import preprocessed_jsonl_for_corpus
from ml.rag.text_processors.preprocess.engines.news import preprocess_folder
from ml.rag.text_processors.preprocess.write_jsonl import write_chunks_jsonl

DEFAULT_OUTPUT_PATH = preprocessed_jsonl_for_corpus("news")


def preprocess_news_collection(
    *,
    input_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    max_files: int | None = None,
) -> int:
    _ = chunk_chars, overlap_chars
    # This writes a complete snapshot, including documents indexed by earlier builds.
    chunks = preprocess_folder(input_dir, max_files=max_files, use_manifest=False)
    return write_chunks_jsonl(chunks, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess news files into JSONL chunks.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--max-files", type=int, default=None)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    n = preprocess_news_collection(input_dir=args.input_dir, output_path=args.output, max_files=args.max_files)
    print(f"Wrote {n} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
