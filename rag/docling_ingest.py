### INFO DOCUMENTO 
## questo script prende il documento in input ed estrae in chunk di demnsione regolabile. 
## Serve per distinguere il testo dalle immagini, in modo da performare l'embedding nello stesso db vettoriale mappando insieme il testo e le immagini

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib

from rag.settings import IMAGES_DIR

def doc_id_for(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]

def extract_text_chunks(file_path: str, chunk_size=2000, chunk_overlap=300) -> dict:
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    # testo strutturato (ottimo per RAG)
    md = doc.export_to_markdown()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(md)

    return {
        "doc_id": doc_id_for(file_path),
        "source_path": file_path,
        "chunks": [{"chunk_id": i, "text": t} for i, t in enumerate(chunks)],
    }


def extract_text_and_images(
    file_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 300,
    images_dir: str = IMAGES_DIR,
) -> dict:
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    md = doc.export_to_markdown()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(md)

    doc_id = doc_id_for(file_path)
    images_out = []
    images_root = Path(images_dir)
    images_root.mkdir(parents=True, exist_ok=True)

    pictures = getattr(doc, "pictures", [])
    for idx, pic in enumerate(pictures):
        try:
            img = pic.get_image(doc)
        except Exception:
            img = None
        if img is None:
            continue

        image_name = f"{doc_id}_img_{idx:04d}.png"
        image_path = images_root / image_name
        try:
            img.save(image_path)
        except Exception:
            continue

        caption = None
        try:
            caption = pic.caption_text(doc).strip() or None
        except Exception:
            caption = None

        page_no = None
        try:
            if getattr(pic, "prov", None):
                page_no = pic.prov[0].page_no
        except Exception:
            page_no = None

        images_out.append(
            {
                "image_id": idx,
                "image_path": str(image_path),
                "caption": caption,
                "page_no": page_no,
            }
        )

    return {
        "doc_id": doc_id,
        "source_path": file_path,
        "chunks": [{"chunk_id": i, "text": t} for i, t in enumerate(chunks)],
        "images": images_out,
    }

