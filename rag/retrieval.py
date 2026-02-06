from qdrant_client import QdrantClient
from rag.settings import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_IMAGE_COLLECTION,
    QDRANT_API_KEY,
)
from rag.embeddings_text_ollama import embed_text_batch
from rag.embeddings_image_clip import embed_texts_for_images
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

    # -------------------------
    # 4. image search (semantic + doc filter)
    # -------------------------
    def retrieve_images_for_docs(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int = 5,
    ):
        if not doc_ids:
            return []

        vec = embed_texts_for_images([query])[0]

        flt = Filter(
            should=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
                for doc_id in doc_ids
            ]
        )

        hits = self.client.query_points(
            collection_name=QDRANT_IMAGE_COLLECTION,
            query=vec,
            using="image",
            limit=top_k,
            query_filter=flt,
        )

        results = []
        for h in hits.points:
            results.append(
                {
                    "score": h.score,
                    "doc_id": h.payload.get("doc_id"),
                    "image_id": h.payload.get("image_id"),
                    "image_path": h.payload.get("image_path"),
                    "caption": h.payload.get("caption"),
                    "page_no": h.payload.get("page_no"),
                    "source_path": h.payload.get("source_path"),
                }
            )

        return results

    def retrieve_context_with_images(self, query: str, top_k_text: int = 5, top_k_images: int = 3) -> dict:
        chunks = self.retrieve_chunks(query, top_k_text)
        doc_ids = list({c["doc_id"] for c in chunks if c.get("doc_id")})

        text_context = "\n\n---\n\n".join(c["text"] for c in chunks if c.get("text"))
        image_query = text_context if text_context else query
        images = self.retrieve_images_for_docs(image_query, doc_ids, top_k_images)

        return {
            "text_context": text_context,
            "chunks": chunks,
            "images": images,
        }
