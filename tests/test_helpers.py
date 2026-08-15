"""Tests for catalog lookup and Markdown post-processing."""

import pandas as pd

from style_finder.utils.helpers import (
    format_alternatives_response,
    get_all_items_for_image,
    process_response,
)


def test_get_all_items_for_image_filters_by_url() -> None:
    dataset = pd.DataFrame(
        {
            "Image URL": ["a.jpg", "b.jpg", "a.jpg"],
            "Item Name": ["Jacket", "Boots", "Skirt"],
        }
    )
    related = get_all_items_for_image("a.jpg", dataset)
    assert list(related["Item Name"]) == ["Jacket", "Skirt"]


def test_get_all_items_for_image_missing_column() -> None:
    dataset = pd.DataFrame({"Item Name": ["Jacket"]})
    related = get_all_items_for_image("a.jpg", dataset)
    assert related.empty


def test_process_response_empty() -> None:
    result = process_response("")
    assert result.startswith("# Fashion Analysis")


def test_process_response_escapes_prices_and_promotes_headings() -> None:
    raw = "A navy coat.\nITEM DETAILS:\n- Coat ($120): https://shop.example"
    result = process_response(raw)
    assert result.startswith("# Fashion Analysis")
    assert "## Item Details" in result
    assert "\\$120" in result


def test_process_response_extracts_details_after_refusal() -> None:
    raw = "I'm not able to provide a description.\nITEM DETAILS:\n* Coat"
    result = process_response(raw)
    assert "Here are the items detected" in result
    assert "- Coat" in result


def test_format_alternatives_response_includes_links() -> None:
    alternatives = {
        "Jacket": [
            {
                "title": "Wool blazer",
                "price": "$90",
                "source": "Shop",
                "link": "https://shop.example/blazer",
            }
        ]
    }
    result = format_alternatives_response("Nice look.", alternatives, similarity_score=0.9)
    assert "Wool blazer" in result
    assert "https://shop.example/blazer" in result
    assert "Similar Items Found" in result
