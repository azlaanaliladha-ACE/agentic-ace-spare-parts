"""Real image embedding client for visual instance retrieval (MVID
proof-of-concept - see src/rag/mvid_ingest.py and
src/rag/visual_retriever.py).

Deliberately separate from src/vision/perception.py's VisionClient
abstraction: that module asks a vision-LLM to *describe* an image in
language (structured JSON); this module encodes an image into a fixed-size
*embedding vector* for nearest-neighbour visual search. The two are
complementary signals, not interchangeable - see AGENT_ARCHITECTURE notes
in src/agents/graph.py.

Model choice: facebook/dinov2-base (Apache 2.0, via `transformers`).
DINOv2 is a self-supervised ViT trained specifically to produce strong
instance/fine-grained visual features (not a language-aligned model like
CLIP/SigLIP) - exactly the property needed to tell visually similar-but-
different industrial parts apart, which is also why the real IRISP paper
(Liu et al., Bosch Research, CVPR 2026 Workshop FGVC13) benchmarked DINOv2
as one of its three candidate encoders for this same problem. dinov2-base
(~86M params, ~330MB) runs comfortably on Apple Silicon via PyTorch's MPS
backend or on CPU, needs no extra heavy dependency (transformers/torch are
already project requirements), and needed no separate embedding-quality
research once its real-world provenance for this exact task was confirmed.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DINOV2_MODEL_NAME = "facebook/dinov2-base"
EMBEDDING_DIM = 768


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ImageEmbeddingClient:
    """Loads facebook/dinov2-base once and embeds images as L2-normalized
    768-dim vectors (the CLS token) suitable for cosine similarity search.
    Construction is real and can fail loudly (network/model unavailable) -
    there is no mock fallback here, consistent with src/vision/perception.py
    ::get_vision_client()'s no-silent-fallback policy.
    """

    def __init__(self, model_name: str = DINOV2_MODEL_NAME) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = _select_device()
        self._processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self.device)
        logger.info("Loaded %s on device=%s", model_name, self.device)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Returns an (N, 768) float32 array of L2-normalized CLS-token
        embeddings, one row per input PIL image, in input order.
        """
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]  # CLS token
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_embeddings.append(cls.cpu().numpy().astype("float32"))
        return np.concatenate(all_embeddings, axis=0)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]


_client_lock = threading.Lock()
_client: ImageEmbeddingClient | None = None


def get_embedding_client() -> ImageEmbeddingClient:
    """Process-wide singleton so the ~330MB model loads once, not per call."""
    global _client
    with _client_lock:
        if _client is None:
            _client = ImageEmbeddingClient()
        return _client
