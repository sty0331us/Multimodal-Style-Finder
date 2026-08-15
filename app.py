#!/usr/bin/env python3
"""CLI entry point for the Multimodal Style Finder Gradio app."""

from __future__ import annotations

import argparse
import logging
import sys

from style_finder.config import settings
from style_finder.pipeline import StyleFinderApp
from style_finder.ui import create_gradio_interface
from style_finder.vectorstore import FashionVectorStore


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Multimodal Style Finder UI")
    parser.add_argument(
        "--dataset",
        default=str(settings.dataset_path),
        help="Path to the catalog pickle used to seed ChromaDB",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(settings.chroma_path),
        help="Directory for the persistent ChromaDB store",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Drop and rebuild the Chroma collection from the pickle",
    )
    parser.add_argument("--host", default=settings.server_name, help="Bind address")
    parser.add_argument("--port", type=int, default=settings.server_port, help="Bind port")
    parser.add_argument(
        "--share",
        action="store_true",
        default=settings.share,
        help="Create a public Gradio share link (off by default)",
    )
    parser.add_argument("--log-level", default=settings.log_level)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    logger = logging.getLogger("style_finder")

    try:
        store = FashionVectorStore(
            persist_directory=args.chroma_path,
            collection_name=settings.chroma_collection,
        )
        app = StyleFinderApp(
            args.dataset,
            vector_store=store,
            force_ingest=args.force_ingest,
        )
        demo = create_gradio_interface(app)
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Failed to start Multimodal Style Finder")
        return 1


if __name__ == "__main__":
    sys.exit(main())
