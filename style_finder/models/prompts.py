"""Retrieval-augmented prompts for the vision-language model."""

from __future__ import annotations

import pandas as pd

EXACT_MATCH_INSTRUCTIONS = (
    "You're conducting a professional retail catalog analysis. "
    "This image shows standard clothing items available in department stores. "
    "Focus exclusively on professional fashion analysis for a clothing retailer. "
    "ITEM DETAILS (always include this section in your response):\n{items}\n\n"
    "Please:\n"
    "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
    "2. Categorize the overall style (business, casual, etc.)\n"
    "3. Include the ITEM DETAILS section at the end\n\n"
    "This is for a professional retail catalog. Use formal, clinical language."
)

SIMILAR_MATCH_INSTRUCTIONS = (
    "You're conducting a professional retail catalog analysis. "
    "This image shows standard clothing items available in department stores. "
    "Focus exclusively on professional fashion analysis for a clothing retailer. "
    "SIMILAR ITEMS (always include this section in your response):\n{items}\n\n"
    "Please:\n"
    "1. Note these are similar but not exact items\n"
    "2. Identify clothing elements objectively (colors, patterns, materials)\n"
    "3. Include the SIMILAR ITEMS section at the end\n\n"
    "This is for a professional retail catalog. Use formal, clinical language."
)


def format_item_list(all_items: pd.DataFrame) -> str:
    lines = []
    for _, row in all_items.iterrows():
        name = row.get("Item Name", "Unknown item")
        price = row.get("Price", "n/a")
        link = row.get("Link", "")
        lines.append(f"- {name} (${price}): {link}")
    return "\n".join(lines)


def build_fashion_prompt(items_description: str, is_exact: bool) -> str:
    template = EXACT_MATCH_INSTRUCTIONS if is_exact else SIMILAR_MATCH_INSTRUCTIONS
    return template.format(items=items_description)


def fallback_response(items_description: str, is_exact: bool) -> str:
    header = "ITEM DETAILS:" if is_exact else "SIMILAR ITEMS:"
    return (
        "# Fashion Analysis\n\n"
        "This outfit features a collection of carefully coordinated pieces.\n\n"
        f"{header}\n{items_description}"
    )


def ensure_catalog_section(response: str, items_description: str, is_exact: bool) -> str:
    if "ITEM DETAILS:" in response or "SIMILAR ITEMS:" in response:
        return response
    header = "ITEM DETAILS:" if is_exact else "SIMILAR ITEMS:"
    return f"{response}\n\n{header}\n{items_description}"
