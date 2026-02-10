from pathlib import Path
import hashlib

import fitz  # PyMuPDF

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.settings import IMAGES_DIR


def doc_id_for(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def extract_text_chunks_docling(file_path: str, chunk_size=2000, chunk_overlap=300) -> dict:
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

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


def extract_images_pymupdf(file_path: str, images_dir: str = IMAGES_DIR) -> list[dict]:
    p = Path(file_path)
    doc_id = doc_id_for(str(p))

    out_dir = Path(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(str(p))
    images_out = []

    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        img_list = page.get_images(full=True)  # lista immagini nella pagina

        for img_idx, img in enumerate(img_list):
            xref = img[0]
            base = pdf.extract_image(xref)
            img_bytes = base["image"]
            ext = base.get("ext", "png")

            image_name = f"{doc_id}_p{page_idx+1:03d}_img{img_idx:03d}.{ext}"
            image_path = out_dir / image_name

            with open(image_path, "wb") as f:
                f.write(img_bytes)

            images_out.append(
                {
                    "image_id": f"{page_idx}_{img_idx}",
                    "image_path": str(image_path),
                    "page_no": page_idx + 1,
                    "caption": None,  # PyMuPDF non estrae caption in modo affidabile
                }
            )

    pdf.close()
    return images_out


def extract_text_and_images(file_path: str, chunk_size=2000, chunk_overlap=300) -> dict:

    # testo con Docling (light, ti funziona già)
    text_data = extract_text_chunks_docling(file_path, chunk_size, chunk_overlap)

    # immagini con PyMuPDF (zero HuggingFace)
    images = extract_images_pymupdf(file_path)

    return {
        "doc_id": text_data["doc_id"],
        "source_path": file_path,
        "chunks": text_data["chunks"],
        "images": images,
    }


#def main():
#    from PIL import Image
#
#    project_root = Path(__file__).resolve().parents[1]
#    pdf_path = project_root / "docs" / "manuale-operativo-firma-digitale.pdf"
#
#    data = extract_text_and_images(str(pdf_path), chunk_size=2000, chunk_overlap=300)
#
#    # apri le prime 3 immagini
#    for img in data["images"][:3]:
#        Image.open(img["image_path"]).show()
#
#
#if __name__ == "__main__":
#    ##main()
