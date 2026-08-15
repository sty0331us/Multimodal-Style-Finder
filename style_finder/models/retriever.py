"""Retrieval errors shared by the Chroma vector store."""

from __future__ import annotations


class RetrievalError(RuntimeError):
    """Raised when the catalog cannot be searched."""
