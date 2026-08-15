"""Load the pickle catalog into the ChromaDB collection."""

from __future__ import annotations

import logging
import os

from style_finder.catalog import load_catalog
from style_finder.vectorstore import FashionVectorStore

logger = logging.getLogger(__name__)


def ingest_catalog(
    store: FashionVectorStore,
    dataset_path: str | os.PathLike[str],
    force: bool = False,
) -> int:
    """
    Upsert pickle rows into Chroma.

    Skips work when the collection already has documents unless `force` is True.
    """
    if not store.is_empty() and not force:
        count = store.count()
        logger.info("Chroma already holds %d SKUs; skipping ingest", count)
        return count

    if force and not store.is_empty():
        store.reset()

    catalog = load_catalog(dataset_path)
    return store.upsert_catalog(catalog)
