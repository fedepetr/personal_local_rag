### INFO DOCUMENTO 
## questo script prende il documento in input ed estrae in chunk di demnsione regolabile. 
## Serve per distinguere il testo dalle immagini, in modo da performare l'embedding nello stesso db vettoriale mappando insieme il testo e le immagini

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

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

