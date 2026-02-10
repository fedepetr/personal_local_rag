from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag.retrieval import Retriever
from rag.llm_ollama import generate
from rag.settings import OLLAMA_LLM_MODEL

@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]
    images: list[dict] | None = None
    output_path: str | None = None

SYSTEM_RULES = """Sei un assistente tecnico.
Usa SOLO le informazioni presenti nel CONTEXT.
Se nel context non c'è abbastanza info, di' chiaramente che non lo sai e cosa manca.
Rispondi in italiano, in modo conciso e pratico.
Quando possibile cita le fonti come [doc_id:chunk_id].
"""

def build_prompt(query: str, chunks: list[dict]) -> str:
    ctx_parts = []
    for c in chunks:
        tag = f"[{c.get('doc_id')}:{c.get('chunk_id')}]"
        ctx_parts.append(f"{tag}\n{c['text']}")

    context = "\n\n---\n\n".join(ctx_parts)

    return f"""{SYSTEM_RULES}

CONTEXT:
{context}

DOMANDA:
{query}

RISPOSTA:
"""

def _safe_relpath(path: str, base_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base_dir.resolve()))
    except Exception:
        return str(Path(path).resolve())

def save_answer_report(
    query: str,
    answer: str,
    chunks: list[dict],
    images: list[dict],
    output_path: str | Path,
) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# RAG Answer")
    lines.append("")
    lines.append("## Query")
    lines.append(query)
    lines.append("")
    lines.append("## Answer")
    lines.append(answer)
    lines.append("")
    lines.append("## Sources")
    for c in chunks:
        tag = f"[{c.get('doc_id')}:{c.get('chunk_id')}]"
        src = c.get("source_path")
        lines.append(f"- {tag} {src}")
    lines.append("")
    lines.append("## Images")
    if images:
        for img in images:
            img_path = img.get("image_path")
            if not img_path:
                continue
            rel = _safe_relpath(img_path, out_path.parent)
            caption = img.get("caption") or ""
            meta = f"(doc_id={img.get('doc_id')}, page={img.get('page_no')})"
            lines.append(f"![{caption}]({rel}) {meta}")
    else:
        lines.append("Nessuna immagine trovata.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)

class RAGAgent:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, query: str, top_k: int = 5, temperature: float = 0.2) -> RAGAnswer:
        chunks = self.retriever.retrieve_chunks(query, top_k=top_k)
        prompt = build_prompt(query, chunks)
        out = generate(model=OLLAMA_LLM_MODEL, prompt=prompt, temperature=temperature)
        return RAGAnswer(answer=out.strip(), sources=chunks)

    def answer_with_images(
        self,
        query: str,
        top_k_text: int = 5,
        top_k_images: int = 3,
        temperature: float = 0.2,
        output_path: str | None = None,
    ) -> RAGAnswer:
        data = self.retriever.retrieve_context_with_images(
            query,
            top_k_text=top_k_text,
            top_k_images=top_k_images,
        )
        chunks = data["chunks"]
        images = data["images"]
        prompt = build_prompt(query, chunks)
        out = generate(model=OLLAMA_LLM_MODEL, prompt=prompt, temperature=temperature)

        if output_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = Path("logs") / f"rag_answer_{ts}.md"

        saved_path = save_answer_report(query, out.strip(), chunks, images, output_path)
        return RAGAnswer(answer=out.strip(), sources=chunks, images=images, output_path=saved_path)
