"""ChromaDB persistence for fashion outfit embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd

from style_finder.models.retriever import RetrievalError

logger = logging.getLogger(__name__)

METADATA_TO_COLUMNS = {
    "item_name": "Item Name",
    "price": "Price",
    "link": "Link",
    "image_url": "Image URL",
}


class FashionVectorStore:
    """Cosine nearest-neighbor search over catalog SKUs stored in ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "fashion_outfits",
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            if persist_directory is None:
                raise ValueError("persist_directory is required unless a Chroma client is provided")
            path = Path(persist_directory)
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))

        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def count(self) -> int:
        return int(self.collection.count())

    def is_empty(self) -> bool:
        return self.count() == 0

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Reset Chroma collection '%s'", self.collection_name)

    def upsert_catalog(self, dataset: pd.DataFrame, batch_size: int = 100) -> int:
        """Index catalog rows that have embeddings. Returns the number of upserted SKUs."""
        if "Embedding" not in dataset.columns:
            raise RetrievalError("Dataset is missing the required 'Embedding' column")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str]] = []
        documents: list[str] = []

        for index, row in dataset.iterrows():
            vector = row.get("Embedding")
            if vector is None or (isinstance(vector, float) and np.isnan(vector)):
                continue
            array = np.asarray(vector, dtype=float).flatten()
            if array.size == 0 or np.isnan(array).any():
                continue
            ids.append(f"sku-{index}")
            embeddings.append(array.tolist())
            metadatas.append(_row_to_metadata(row))
            documents.append(_row_to_document(row))

        if not ids:
            raise RetrievalError("Dataset contains no usable embeddings")

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end],
            )

        logger.info("Upserted %d SKUs into Chroma collection '%s'", len(ids), self.collection_name)
        return len(ids)

    def query_closest(self, user_vector: np.ndarray, n_results: int = 1) -> tuple[pd.Series, float]:
        """Return the nearest catalog row and cosine similarity."""
        if self.is_empty():
            raise RetrievalError("Chroma collection is empty. Run scripts/ingest_chromadb.py first.")

        query = np.asarray(user_vector, dtype=float).flatten().tolist()
        try:
            results = self.collection.query(
                query_embeddings=[query],
                n_results=max(1, n_results),
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:
            logger.exception("Chroma query failed")
            raise RetrievalError(f"Unable to find a catalog match: {exc}") from exc

        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not metadatas:
            raise RetrievalError("Chroma query returned no neighbors")

        similarity = _cosine_similarity_from_distance(float(distances[0]))
        closest_row = metadata_to_row(metadatas[0])
        logger.info(
            "Nearest catalog item: %s (cosine=%.4f)",
            closest_row.get("Item Name", "unknown"),
            similarity,
        )
        return closest_row, similarity

    def get_items_for_image(self, image_url: str) -> pd.DataFrame:
        """Return every indexed SKU that belongs to the same outfit photo."""
        results = self.collection.get(
            where={"image_url": image_url},
            include=["metadatas", "documents"],
        )
        metadatas = results.get("metadatas") or []
        rows = [metadata_to_row(meta) for meta in metadatas if meta]
        if not rows:
            return pd.DataFrame(columns=list(METADATA_TO_COLUMNS.values()))
        frame = pd.DataFrame(rows)
        logger.info("Found %d items for image URL %s", len(frame), image_url)
        return frame

    def _get_or_create_collection(self) -> Any:
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )


def metadata_to_row(metadata: dict[str, Any]) -> pd.Series:
    return pd.Series(
        {
            column: metadata.get(key, "")
            for key, column in METADATA_TO_COLUMNS.items()
        }
    )


def _row_to_metadata(row: pd.Series) -> dict[str, str]:
    return {
        "item_name": str(row.get("Item Name") or ""),
        "price": str(row.get("Price") or ""),
        "link": str(row.get("Link") or ""),
        "image_url": str(row.get("Image URL") or ""),
    }


def _row_to_document(row: pd.Series) -> str:
    name = row.get("Item Name") or "Unknown item"
    price = row.get("Price") or "n/a"
    return f"{name} (${price})"


def _cosine_similarity_from_distance(distance: float) -> float:
    """Chroma cosine space stores distance = 1 - cosine similarity."""
    return float(1.0 - distance)
