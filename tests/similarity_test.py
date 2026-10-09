import ollama
import numpy as np


def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return np.array(response["embeddings"][0])

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

memories = [
    "User uses Arch Linux",
    "User likes pizza",
    "Project deadline is November 2",
    "User is building a Next.js app"
]


query = "What Linux distribution does the user use?"

query_embedding = get_embedding(query)

for memory in memories:
    memory_embedding = get_embedding(memory)

    score = cosine_similarity(
        query_embedding,
        memory_embedding
    )

    print(f"{score:.4f} | {memory}")