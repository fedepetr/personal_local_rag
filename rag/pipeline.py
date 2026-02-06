from pathlib import Path
import uuid
from qdrant_client.models import PointStruct

from rag.settings import DOCS_DIR
from rag.docling_ingest import extract_text_chunks
from rag.embeddings_text_ollama import embed_text_batch
from rag.qdrant_store import upsert
from datetime import datetime

# funzione che definisce il dominio provando a prenderlo dal path (ora non serve)
def infer_domain_from_path(path: str) -> str:
    p = path.lower()
    if "hr" in p:
        return "HR"
    if "elicotteri" in p:
        return "Elicotteri"
    if "aerei" in p:
        return "Aerei"
    if "logistica" in p:
        return "Logistica"
    return "General"

# Namespace costante: serve per generare UUID deterministici
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

def point_id(doc_id: str, chunk_id: int) -> str:
    # UUID deterministico: stesso doc_id+chunk_id => stesso ID
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{chunk_id}"))

def ingest_file(file_path: str) -> dict:
    data = extract_text_chunks(file_path)
    texts = [c["text"] for c in data["chunks"]]
    vecs = embed_text_batch(texts)

    points = []
    for c, v in zip(data["chunks"], vecs):
        pid = point_id(data["doc_id"], c["chunk_id"])
        points.append(
            PointStruct(
                id=pid,
                vector={"dense": v},
                payload={
                    "doc_id": data["doc_id"],
                    "chunk_id": c["chunk_id"],

                    # 🔹 nuovi campi schema Qdrant
                    "version": "v1",
                    "domain": infer_domain_from_path(data["source_path"]),
                    "update_date": datetime.utcnow().isoformat() + "Z",
                    "source_path": data["source_path"],

                    # 🔹 utile per LLM (facoltativo)
                    "text": c["text"],
                },
            )
        )

    upsert(points)
    return {"doc_id": data["doc_id"], "chunks": len(points), "file": file_path}

def ingest_folder(folder: str = DOCS_DIR) -> list[dict]:
    p = Path(folder)
    out = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".pdf", ".docx", ".pptx", ".md", ".txt", ".html"}:
            out.append(ingest_file(str(fp)))
    return out
