#!/usr/bin/env python3
"""Build or refresh the ChromaDB collection from the catalog pickle."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from style_finder.config import settings
from style_finder.ingest import ingest_catalog
from style_finder.vectorstore import FashionVectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest fashion embeddings into ChromaDB")
    parser.add_argument("--dataset", default=str(settings.dataset_path))
    parser.add_argument("--chroma-path", default=str(settings.chroma_path))
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--log-level", default=settings.log_level)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    store = FashionVectorStore(
        persist_directory=args.chroma_path,
        collection_name=args.collection,
    )
    count = ingest_catalog(store, args.dataset, force=args.force)
    logging.getLogger("ingest").info("Chroma collection ready with %d SKUs", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
