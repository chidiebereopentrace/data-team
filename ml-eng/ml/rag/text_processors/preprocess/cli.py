from __future__ import annotations

import argparse
from pathlib import Path

from ml.rag.paths import preprocessed_jsonl_for_corpus
from ml.rag.text_processors.preprocess.validate import validate_jsonl
from ml.rag.text_processors.preprocess.write_jsonl import write_chunks_jsonl


def _run_preprocess(corpus: str, input_dir: Path, output: Path) -> int:
    if corpus == "news":
        from ml.rag.text_processors.preprocess.engines.news import preprocess_folder

        chunks = preprocess_folder(input_dir)
    elif corpus == "research":
        from ml.rag.text_processors.preprocess.engines.research import preprocess_folder

        chunks = preprocess_folder(input_dir)
    elif corpus == "ota":
        from ml.rag.text_processors.preprocess.engines.ota import preprocess_folder

        chunks = preprocess_folder(input_dir)
    else:
        raise SystemExit(f"Unknown corpus: {corpus}")
    n = write_chunks_jsonl(chunks, output)
    print(f"Wrote {n} chunks to {output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess documents into chunk JSONL.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run corpus preprocessor")
    run.add_argument("--corpus", required=True, choices=("news", "research", "ota"))
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, default=None)

    val = sub.add_parser("validate", help="Validate chunk JSONL schema and token stats")
    val.add_argument("--jsonl", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "run":
        out = args.output or preprocessed_jsonl_for_corpus(args.corpus)
        return _run_preprocess(args.corpus, args.input_dir, out)
    if args.cmd == "validate":
        stats = validate_jsonl(args.jsonl)
        print(stats)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
