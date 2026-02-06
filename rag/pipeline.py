from pathlib import Path
import uuid
from qdrant_client.models import PointStruct

from rag.settings import DOCS_DIR, QDRANT_IMAGE_COLLECTION
from rag.docling_ingest import extract_text_and_images
from rag.embeddings_text_ollama import embed_text_batch
from rag.embeddings_image_clip import embed_images
from rag.qdrant_store import upsert, upsert_to_collection
from datetime import datetime
from PIL import Image

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

def image_point_id(doc_id: str, image_id: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:img:{image_id}"))

def ingest_file(file_path: str) -> dict:
    data = extract_text_and_images(file_path)
    texts = [c["text"] for c in data["chunks"]]
    vecs = embed_text_batch(texts)

    points = []
    for c, v in zip(data["chunks"], vecs):
        pid = point_id(data["doc_id"], c["chunk_id"])
        points.append(
            PointStruct(
                id=pid,
                vector={"text": v},
                payload={
                    "doc_id": data["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "type": "text",

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
    image_points = []
    images = data.get("images", [])
    if images:
        pil_images = []
        image_payloads = []
        for img in images:
            try:
                with Image.open(img["image_path"]) as im:
                    pil = im.convert("RGB").copy()
            except Exception:
                continue
            pil_images.append(pil)
            image_payloads.append(img)

        if pil_images:
            image_vecs = embed_images(pil_images)
            for img, v in zip(image_payloads, image_vecs):
                pid = image_point_id(data["doc_id"], img["image_id"])
                image_points.append(
                    PointStruct(
                        id=pid,
                        vector={"image": v},
                        payload={
                            "doc_id": data["doc_id"],
                            "image_id": img["image_id"],
                            "type": "image",
                            "source_path": data["source_path"],
                            "image_path": img["image_path"],
                            "caption": img.get("caption"),
                            "page_no": img.get("page_no"),
                            "version": "v1",
                            "update_date": datetime.utcnow().isoformat() + "Z",
                        },
                    )
                )

    if image_points:
        upsert_to_collection(image_points, QDRANT_IMAGE_COLLECTION)

    return {"doc_id": data["doc_id"], "chunks": len(points), "file": file_path}

def ingest_folder(folder: str = DOCS_DIR) -> list[dict]:
    p = Path(folder)
    out = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".pdf", ".docx", ".pptx", ".md", ".txt", ".html"}:
            out.append(ingest_file(str(fp)))
    return out
