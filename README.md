# RAG Project (Local)

Questo progetto implementa un RAG (Retrieval-Augmented Generation) interamente in locale, con ingestion, embedding e retrieval su documenti PDF.

**Stack e servizi**
- Qdrant: vector database locale (via Docker) per lo storage degli embedding
- Ollama:
  - `embeddinggemma`: modello per l'embedding del testo
  - `llama3.2`: LLM per il retrieval e la risposta
- Docling: parser che converte PDF in HTML strutturato per preservare la struttura durante l'embedding

**Requisiti**
- Python (virtual environment)
- Docker (per Qdrant)
- Ollama installato e avviato
- Dipendenze Python in `requirements.txt`

**Struttura del progetto**
- `rag/`: librerie e codice per ingestion, embedding e retrieval
- `scripts/`: script di test per ingestion ed esecuzione RAG
- `docs/`: documenti da indicizzare (PDF)

**Setup**
1. Crea e attiva il virtual environment
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Installa le dipendenze
   ```powershell
   pip install -r requirements.txt
   ```

**Uso**
1. Ingestion + embedding
   ```powershell
   python scripts/ingest.py
   ```
2. Retrieval (test di risposta RAG)
   ```powershell
   python scripts/test_rag_answer.py
   ```

**Diagrammi**

Diagramma semplice
```mermaid
flowchart LR
  A[Documenti] --> B[Estrazione testo+immagini]
  B --> C[Embedding testo]
  B --> D[Embedding immagini]
  C --> E[Qdrant: rag_text]
  D --> F[Qdrant: rag_images]
  G[Query] --> H[Search testo]
  H --> I[Doc_id rilevanti]
  G --> J[Search immagini]
  J --> K[Filtro per doc_id]
  K --> L[Immagini rilevanti]
  H --> M[Contesto testuale]
  L --> N[Contesto visivo]
  M --> O[Contesto finale]
  N --> O
  O --> P[LLM]
```

Diagramma tecnico (con moduli reali)
```mermaid
flowchart TD
  A[docs/*] --> B[rag.docling_ingest.extract_text_and_images]
  B --> C[rag.embeddings_text_ollama.embed_text_batch]
  B --> D[rag.embeddings_image_clip.embed_images]

  C --> E[rag.qdrant_store.upsert -> QDRANT_COLLECTION]
  D --> F[rag.qdrant_store.upsert_to_collection -> QDRANT_IMAGE_COLLECTION]

  G[User query] --> H[rag.embeddings_text_ollama.embed_text_batch]
  H --> I[rag.retrieval.Retriever.search -> rag_text]
  I --> J[doc_id list]

  G --> K[rag.embeddings_image_clip.embed_texts_for_images]
  K --> L[rag.retrieval.Retriever.retrieve_images_for_docs -> rag_images]
  L --> M[filter doc_id]

  I --> N[text_context]
  L --> O[images]
  N --> P[final_context]
  O --> P
  P --> Q[LLM response]
```

**Note**
- La dimensione dei chunk per l'embedding e configurabile direttamente negli script.

