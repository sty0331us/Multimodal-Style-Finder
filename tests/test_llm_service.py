"""Tests for RAG prompt assembly without calling watsonx."""

import pandas as pd

from style_finder.models.prompts import (
    build_fashion_prompt,
    ensure_catalog_section,
    fallback_response,
    format_item_list,
)


def test_format_item_list() -> None:
    items = pd.DataFrame(
        {
            "Item Name": ["Silk blouse"],
            "Price": ["89.00"],
            "Link": ["https://shop.example/blouse"],
        }
    )
    listing = format_item_list(items)
    assert "Silk blouse ($89.00): https://shop.example/blouse" in listing


def test_fallback_response_uses_similar_header() -> None:
    text = fallback_response("- Jacket", is_exact=False)
    assert "SIMILAR ITEMS:" in text
    assert text.startswith("# Fashion Analysis")


def test_build_fashion_prompt_exact_match_includes_item_details() -> None:
    prompt = build_fashion_prompt("- Coat ($10): https://x", is_exact=True)
    assert "ITEM DETAILS" in prompt
    assert "Coat ($10)" in prompt


def test_ensure_catalog_section_appends_when_missing() -> None:
    result = ensure_catalog_section("A navy look.", "- Coat", is_exact=True)
    assert "ITEM DETAILS:" in result
    assert "- Coat" in result
