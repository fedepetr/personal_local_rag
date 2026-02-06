from dataclasses import dataclass
from rag.retrieval import Retriever
from rag.llm_ollama import generate
from rag.settings import OLLAMA_LLM_MODEL

@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]

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

class RAGAgent:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, query: str, top_k: int = 5, temperature: float = 0.2) -> RAGAnswer:
        chunks = self.retriever.retrieve_chunks(query, top_k=top_k)
        prompt = build_prompt(query, chunks)
        out = generate(model=OLLAMA_LLM_MODEL, prompt=prompt, temperature=temperature)
        return RAGAnswer(answer=out.strip(), sources=chunks)
