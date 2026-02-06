from rag.rag_agent import RAGAgent

agent = RAGAgent()

q = "sai dirmi quali sono i Limiti d’uso e di valore citati nel documento?"
res = agent.answer(q, top_k=5)

print("\n=== ANSWER ===\n")
print(res.answer)

print("\n=== SOURCES ===\n")
for s in res.sources:
    print(f"- score={s['score']:.3f} [{s['doc_id']}:{s['chunk_id']}] {s['source_path']}")
