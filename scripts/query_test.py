from qdrant_client import QdrantClient
from rag.embeddings_text_ollama import embed_text_batch
from rag.settings import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY

query = "come accedo ai servizi VDI?"

vec = embed_text_batch([query])[0]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

hits = client.query_points(
    collection_name=QDRANT_COLLECTION,
    query=vec,          # ← nuovo parametro
    using="text",       # ← nome del vector (named vector)
    limit=3,
)

for h in hits.points:
    print(f"\nscore={h.score:.4f}")
    print(h.payload["text"][:500])
    print("-" * 60)
