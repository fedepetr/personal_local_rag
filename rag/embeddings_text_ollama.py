## INFO DOCUMENTO
# Questo codice performa l'embedding dei chunk di testo che gli vengono passati effettuando una chiamata 
# api verso OLLAMA che al momento caricato il modello embedding Gemma

from __future__ import annotations
import time
import requests

from rag.settings import OLLAMA_URL, OLLAMA_TEXT_EMBED_MODEL

def embed_text_batch(
    texts: list[str],
    batch_size: int = 16,
    timeout_s: int = 300,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> list[list[float]]:
    """
    Embeds a list of texts using Ollama /api/embed with batching + retry.
    Returns embeddings in the same order as input texts.
    """
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": OLLAMA_TEXT_EMBED_MODEL, "input": batch},
                    timeout=timeout_s,
                )
                r.raise_for_status()
                all_vecs.extend(r.json()["embeddings"])
                last_err = None
                break
            except requests.exceptions.ReadTimeout as e:
                last_err = e
                time.sleep(backoff_s * (attempt + 1))
            except requests.exceptions.RequestException as e:
                # altri errori HTTP/connessione
                last_err = e
                time.sleep(backoff_s * (attempt + 1))

        if last_err is not None:
            raise last_err

    return all_vecs
