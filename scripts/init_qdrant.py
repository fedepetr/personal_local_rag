import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag.settings import QDRANT_URL, QDRANT_COLLECTION, OLLAMA_URL, OLLAMA_TEXT_EMBED_MODEL

def guess_dim() -> int:
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_TEXT_EMBED_MODEL, "input": ["dim probe"]},
        timeout=60,
    )
    r.raise_for_status()
    return len(r.json()["embeddings"][0])

def main():
    c = QdrantClient(url=QDRANT_URL)
    if c.collection_exists(QDRANT_COLLECTION):
        print(f"Collection '{QDRANT_COLLECTION}' already exists.")
        return

    dim = guess_dim()
    c.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={"text": VectorParams(size=dim, distance=Distance.COSINE)},
    )
    print(f"Created '{QDRANT_COLLECTION}' with text dim={dim}")

if __name__ == "__main__":
    main()
