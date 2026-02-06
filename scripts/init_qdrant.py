import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag.settings import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_IMAGE_COLLECTION,
    OLLAMA_URL,
    OLLAMA_TEXT_EMBED_MODEL,
)
from rag.embeddings_image_clip import embed_texts_for_images

def guess_dim() -> int:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_TEXT_EMBED_MODEL, "input": ["dim probe"]},
        timeout=60,
    )
    r.raise_for_status()
    return len(r.json()["embeddings"][0])

def guess_image_dim() -> int:
    return len(embed_texts_for_images(["dim probe"])[0])

def main():
    c = QdrantClient(url=QDRANT_URL)
    if not c.collection_exists(QDRANT_COLLECTION):
        dim = guess_dim()
        c.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"text": VectorParams(size=dim, distance=Distance.COSINE)},
        )
        print(f"Created '{QDRANT_COLLECTION}' with text dim={dim}")
    else:
        print(f"Collection '{QDRANT_COLLECTION}' already exists.")

    if not c.collection_exists(QDRANT_IMAGE_COLLECTION):
        img_dim = guess_image_dim()
        c.create_collection(
            collection_name=QDRANT_IMAGE_COLLECTION,
            vectors_config={"image": VectorParams(size=img_dim, distance=Distance.COSINE)},
        )
        print(f"Created '{QDRANT_IMAGE_COLLECTION}' with image dim={img_dim}")
    else:
        print(f"Collection '{QDRANT_IMAGE_COLLECTION}' already exists.")

if __name__ == "__main__":
    main()
