"""Utility functions for catalog lookup and markdown cleanup."""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

REJECTION_PHRASES = (
    "I'm not able to provide",
    "I cannot provide",
    "I cannot",
    "I apologize, but I cannot",
    "I apologize, but",
    "I don't feel comfortable",
    "violated our content policy",
)


def get_all_items_for_image(image_url: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """Return every catalog SKU that belongs to the same outfit image."""
    if "Image URL" not in dataset.columns:
        logger.error("Dataset is missing the 'Image URL' column")
        return dataset.iloc[0:0]

    related_items = dataset[dataset["Image URL"] == image_url]
    logger.info("Found %d items for image URL %s", len(related_items), image_url)
    return related_items


def format_alternatives_response(
    user_response: str,
    alternatives: dict,
    similarity_score: float,
    threshold: float = 0.8,
) -> str:
    """Append optional shopping alternatives under the model writeup."""
    if not user_response or any(phrase in user_response for phrase in REJECTION_PHRASES):
        user_response = "## Fashion Analysis Results\n\nHere are the items detected in your image:"

    heading = (
        "Here are some similar items we found:"
        if similarity_score >= threshold
        else "Here are some visually similar items:"
    )
    enhanced_response = f"{user_response}\n\n## Similar Items Found\n\n{heading}\n"

    items_added = 0
    max_items = 10
    for item, alts in alternatives.items():
        enhanced_response += f"\n### {item}:\n"
        if not alts:
            enhanced_response += "- No alternatives found.\n"
            continue
        for alt in alts[:3]:
            if items_added >= max_items:
                break
            enhanced_response += (
                f"- {alt['title']} for {alt['price']} from {alt['source']} "
                f"([Buy it here]({alt['link']}))\n"
            )
            items_added += 1

    return enhanced_response


def process_response(response: str | None) -> str:
    """Normalize model output into stable Markdown for the Gradio panel."""
    if not response:
        logger.warning("Empty response received")
        return (
            "# Fashion Analysis\n\n"
            "No detailed analysis was generated. Please refer to the item details below."
        )

    if any(phrase in response for phrase in REJECTION_PHRASES):
        logger.warning("Model rejected the request; extracting item details")
        items_section = _extract_items_section(response)
        if items_section:
            formatted_items = re.sub(r"^\* ", "- ", items_section, flags=re.MULTILINE)
            return (
                "# Fashion Analysis\n\n"
                "Here are the items detected in your image:\n\n"
                f"{formatted_items}"
            )
        return response.replace("$", "\\$")

    processed = response.replace("$", "\\$")
    processed = processed.replace("ITEM DETAILS:", "## Item Details")
    processed = processed.replace("SIMILAR ITEMS:", "## Similar Items")

    if not processed.startswith("#"):
        processed = f"# Fashion Analysis\n\n{processed}"

    return re.sub(r"^\* ", "- ", processed, flags=re.MULTILINE)


def _extract_items_section(response: str) -> str | None:
    if "ITEM DETAILS:" in response:
        return "## Item Details\n\n" + response.split("ITEM DETAILS:", 1)[1].strip()
    if "SIMILAR ITEMS:" in response:
        return "## Similar Items\n\n" + response.split("SIMILAR ITEMS:", 1)[1].strip()
    return None
