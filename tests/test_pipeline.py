"""Tests for dataset validation in the catalog loader."""

import pandas as pd
import pytest

from style_finder.catalog import validate_dataset


def test_validate_dataset_accepts_required_columns() -> None:
    data = pd.DataFrame(columns=["Embedding", "Image URL", "Item Name", "Price", "Link"])
    validate_dataset(data)


def test_validate_dataset_rejects_missing_columns() -> None:
    data = pd.DataFrame(columns=["Item Name"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(data)
