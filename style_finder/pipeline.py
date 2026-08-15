"""Orchestrates encode → Chroma retrieve → generate for a single user photo."""

from __future__ import annotations

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any

from PIL import Image

from style_finder.config import Settings, settings
from style_finder.ingest import ingest_catalog
from style_finder.models.image_processor import ImageEncodingError, ImageProcessor
from style_finder.models.llm_service import LlamaVisionService
from style_finder.models.retriever import RetrievalError
from style_finder.utils.helpers import process_response
from style_finder.vectorstore import FashionVectorStore

logger = logging.getLogger(__name__)


class StyleFinderApp:
    """Multimodal RAG pipeline for fashion style analysis."""

    def __init__(
        self,
        dataset_path: str | os.PathLike[str],
        app_settings: Settings | None = None,
        vector_store: FashionVectorStore | None = None,
        force_ingest: bool = False,
    ) -> None:
        self.settings = app_settings or settings
        self.vector_store = vector_store or FashionVectorStore(
            persist_directory=self.settings.chroma_path,
            collection_name=self.settings.chroma_collection,
        )
        ingested = ingest_catalog(self.vector_store, dataset_path, force=force_ingest)
        logger.info("Chroma collection ready with %d SKUs", ingested)

        self.image_processor = ImageProcessor(
            image_size=self.settings.image_size,
            norm_mean=self.settings.normalization_mean,
            norm_std=self.settings.normalization_std,
        )
        self.llm_service = LlamaVisionService(
            model_id=self.settings.llama_model_id,
            project_id=self.settings.watsonx_project_id,
            region=self.settings.watsonx_region,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            api_key=self.settings.require_api_key(),
            max_tokens=self.settings.llm_max_tokens,
        )

    def process_image(self, image: Any) -> str:
        """Run the full multimodal RAG pipeline and return Markdown."""
        if image is None:
            return "Please upload a fashion image first."

        image_path, cleanup_path = self._resolve_image_path(image)
        try:
            user_encoding = self.image_processor.encode_image(image_path, is_url=False)
            if user_encoding["vector"] is None:
                return "Error: Unable to process the image. Please try another image."

            closest_row, similarity_score = self.vector_store.query_closest(
                user_encoding["vector"]
            )
            logger.info(
                "Closest match: %s (score=%.2f)",
                closest_row["Item Name"],
                similarity_score,
            )

            all_items = self.vector_store.get_items_for_image(closest_row["Image URL"])
            if all_items.empty:
                return "Error: No items found for the matched image."

            bot_response = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding["base64"],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score,
                threshold=self.settings.similarity_threshold,
            )
            return process_response(bot_response)
        except ImageEncodingError:
            return "Error: Unable to process the image. Please try another image."
        except RetrievalError:
            return "Error: Unable to find a match. Please try another image."
        except Exception:
            logger.exception("Unhandled pipeline failure")
            return "Error: Style analysis failed unexpectedly. Check the server logs."
        finally:
            if cleanup_path:
                try:
                    os.unlink(cleanup_path)
                except OSError:
                    logger.debug("Could not delete temp file %s", cleanup_path)

    @staticmethod
    def _resolve_image_path(image: Any) -> tuple[str, str | None]:
        if isinstance(image, str):
            return image, None

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.convert("RGB").save(tmp.name, format="JPEG")
            return tmp.name, tmp.name
