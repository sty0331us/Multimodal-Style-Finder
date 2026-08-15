"""
Watsonx client for Llama 4 Vision Instruct.

The model sees both the user photo and retrieved catalog context (names,
prices, and purchase links), which is the generation stage of multimodal RAG.
"""

from __future__ import annotations

import logging

import pandas as pd
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from style_finder.models.prompts import (
    build_fashion_prompt,
    ensure_catalog_section,
    fallback_response,
    format_item_list,
)

logger = logging.getLogger(__name__)


class LLMGenerationError(RuntimeError):
    """Raised when the vision-language model cannot produce a response."""


class LlamaVisionService:
    """Generate fashion analysis from an image plus retrieved catalog context."""

    def __init__(
        self,
        model_id: str,
        project_id: str,
        region: str = "us-south",
        temperature: float = 0.2,
        top_p: float = 0.6,
        api_key: str | None = None,
        max_tokens: int = 2000,
    ) -> None:
        credentials = Credentials(
            url=f"https://{region}.ml.cloud.ibm.com",
            api_key=api_key,
        )
        self.client = APIClient(credentials)
        params = TextChatParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params,
        )
        logger.info("Initialized Llama vision client: %s (%s)", model_id, region)

    def generate_response(self, encoded_image: str, prompt: str) -> str:
        """Send a multimodal chat request (text prompt + JPEG)."""
        logger.info("Sending multimodal request (prompt_chars=%d)", len(prompt))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]
        try:
            response = self.model.chat(messages=messages)
            content = response["choices"][0]["message"]["content"]
            logger.info("Received model response (chars=%d)", len(content))
            if len(content) >= 7900:
                logger.warning("Response may be truncated (chars=%d)", len(content))
            return content
        except Exception as exc:
            logger.exception("Vision LLM request failed")
            raise LLMGenerationError(f"Error generating response: {exc}") from exc

    def generate_fashion_response(
        self,
        user_image_base64: str,
        matched_row: pd.Series,
        all_items: pd.DataFrame,
        similarity_score: float,
        threshold: float = 0.8,
    ) -> str:
        """
        Build a retrieval-augmented prompt and generate a catalog-style writeup.

        Exact matches (score >= threshold) ask for ITEM DETAILS.
        Near matches ask for SIMILAR ITEMS so the user is not oversold a match.
        """
        del matched_row  # reserved for future prompt personalization
        items_description = format_item_list(all_items)
        is_exact = similarity_score >= threshold
        prompt = build_fashion_prompt(items_description, is_exact)

        try:
            response = self.generate_response(user_image_base64, prompt)
        except LLMGenerationError:
            logger.warning("Falling back to catalog-only response")
            return fallback_response(items_description, is_exact)

        if len(response) < 100:
            logger.info("Model response was too short; using catalog fallback")
            return fallback_response(items_description, is_exact)

        return ensure_catalog_section(response, items_description, is_exact)
