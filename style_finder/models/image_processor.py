"""
Image encoding for multimodal RAG.

ResNet50 produces the same 1000-d embedding used to build the catalog pickle,
so runtime queries stay comparable to vectors stored in ChromaDB.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


class ImageEncodingError(RuntimeError):
    """Raised when an image cannot be loaded or encoded."""


class ImageProcessor:
    """Encode fashion images to JPEG Base64 and a ResNet50 feature vector."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        norm_mean: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
        norm_std: tuple[float, float, float] | list[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=weights).to(self.device)
        self.model.eval()
        logger.info("ResNet50 encoder ready on %s", self.device)

        # Keep the original preprocessing so embeddings match the published pickle.
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(norm_mean), std=list(norm_std)),
            ]
        )

    def encode_image(self, image_input: str, is_url: bool = True) -> dict[str, Any]:
        """
        Return a JPEG base64 string and a flattened ResNet50 feature vector.

        Args:
            image_input: URL or local filesystem path.
            is_url: Treat `image_input` as a remote URL when True.
        """
        try:
            image = self._load_image(image_input, is_url=is_url)
            base64_string = self._to_base64_jpeg(image)
            feature_vector = self._embed(image)
            return {"base64": base64_string, "vector": feature_vector}
        except Exception as exc:
            logger.exception("Failed to encode image from %s", image_input)
            raise ImageEncodingError(f"Unable to encode image: {exc}") from exc

    def _load_image(self, image_input: str, is_url: bool) -> Image.Image:
        if is_url:
            response = requests.get(image_input, timeout=20)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_input).convert("RGB")

    @staticmethod
    def _to_base64_jpeg(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _embed(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)
        return features.cpu().numpy().flatten()
