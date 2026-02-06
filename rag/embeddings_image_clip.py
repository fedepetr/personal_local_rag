## INFO DOCUMENTO
# Embedding per immagini (e testo) usando CLIP via sentence-transformers.

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from PIL import Image
from sentence_transformers import SentenceTransformer

from rag.settings import IMAGE_EMBED_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(IMAGE_EMBED_MODEL)


def embed_images(images: Iterable[Image.Image]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(list(images), normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_texts_for_images(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]
