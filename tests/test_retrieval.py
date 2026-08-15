"""Tests for ChromaDB retrieval without loading ResNet50."""

import uuid

import numpy as np
import pandas as pd
import pytest

chromadb = pytest.importorskip("chromadb")

from style_finder.models.retriever import RetrievalError
from style_finder.vectorstore import FashionVectorStore


def _store() -> FashionVectorStore:
    return FashionVectorStore(
        collection_name=f"test_fashion_{uuid.uuid4().hex[:8]}",
        client=chromadb.EphemeralClient(),
    )


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item Name": ["noise", "target", "other", "target-bag"],
            "Price": ["1", "10", "2", "20"],
            "Link": ["a", "b", "c", "d"],
            "Image URL": ["noise.jpg", "look.jpg", "other.jpg", "look.jpg"],
            "Embedding": [
                None,
                np.array([1.0, 0.0], dtype=float),
                np.array([0.0, 1.0], dtype=float),
                np.array([0.9, 0.1], dtype=float),
            ],
        }
    )


def test_query_closest_skips_missing_embeddings() -> None:
    store = _store()
    store.upsert_catalog(_catalog())
    row, score = store.query_closest(np.array([1.0, 0.0], dtype=float))
    assert row["Item Name"] == "target"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_get_items_for_image_returns_related_skus() -> None:
    store = _store()
    store.upsert_catalog(_catalog())
    items = store.get_items_for_image("look.jpg")
    assert sorted(items["Item Name"].tolist()) == ["target", "target-bag"]


def test_upsert_rejects_catalog_without_embeddings() -> None:
    store = _store()
    with pytest.raises(RetrievalError, match="Embedding"):
        store.upsert_catalog(pd.DataFrame({"Item Name": ["Jacket"]}))


def test_upsert_rejects_empty_catalog() -> None:
    store = _store()
    with pytest.raises(RetrievalError, match="no usable embeddings"):
        store.upsert_catalog(
            pd.DataFrame(
                {
                    "Item Name": ["Jacket"],
                    "Price": ["1"],
                    "Link": ["x"],
                    "Image URL": ["a.jpg"],
                    "Embedding": [None],
                }
            )
        )


def test_query_on_empty_collection_raises() -> None:
    store = _store()
    with pytest.raises(RetrievalError, match="empty"):
        store.query_closest(np.array([1.0, 0.0], dtype=float))
