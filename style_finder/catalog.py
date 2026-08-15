"""Catalog loading and schema checks."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"Embedding", "Image URL", "Item Name", "Price", "Link"}
)


def validate_dataset(data: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def load_catalog(dataset_path: str | os.PathLike[str]) -> pd.DataFrame:
    path = os.fspath(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Run scripts/download_dataset.sh first."
        )

    data = pd.read_pickle(path)
    if data.empty:
        raise ValueError("The loaded dataset is empty")

    validate_dataset(data)
    logger.info("Loaded catalog with %d rows from %s", len(data), path)
    return data
