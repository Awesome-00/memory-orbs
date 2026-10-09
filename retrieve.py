import ollama
import numpy as np
from store import get_active_memories

def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return np.array(
        response["embeddings"][0],
        dtype=np.float32
    )

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )