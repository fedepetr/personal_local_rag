## INFO DOCUMENTO 
# Questo script riporta le funzioni per inserire nella collezione i vettori embeddati dal modello di OLLAMA

from qdrant_client import QdrantClient
from rag.settings import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY

def client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY  # ← QUI PASSIAMO LA KEY
    )

def upsert(points):
    client().upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )
