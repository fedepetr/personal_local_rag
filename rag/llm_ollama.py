import requests
from rag.settings import OLLAMA_URL

def generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]
