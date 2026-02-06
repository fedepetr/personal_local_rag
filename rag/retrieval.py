from qdrant_client import QdrantClient
from rag.settings import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY
from rag.embeddings_text_ollama import embed_text_batch
from qdrant_client.models import Filter, FieldCondition, MatchValue

class Retriever:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

    # -------------------------
    # 1. search raw
    # -------------------------
    def search(self, query: str, top_k: int = 5, domain: str | None = None):
        vec = embed_text_batch([query])[0]

        flt = None
        if domain:
            flt = Filter(
                must=[
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain)
                    )
                ]
            )

        hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            using="text",
            limit=top_k,
            query_filter=flt,
        )

        return hits.points

    # -------------------------
    # 2. formatted chunks
    # -------------------------
    def retrieve_chunks(self, query: str, top_k: int = 5, domain: str | None = None):
        hits = self.search(query, top_k, domain)

        results = []
        for h in hits:
            results.append({
                "score": h.score,
                "text": h.payload.get("text"),
                "doc_id": h.payload.get("doc_id"),
                "chunk_id": h.payload.get("chunk_id"),
                "source_path": h.payload.get("source_path"),
                "domain": h.payload.get("domain"),
                "version": h.payload.get("version"),
                "update_date": h.payload.get("update_date"),
            })

        return results

    # -------------------------
    # 3. ready-for-LLM context
    # -------------------------
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        chunks = self.retrieve_chunks(query, top_k)

        context = "\n\n---\n\n".join(
            c["text"] for c in chunks
        )

        return context
