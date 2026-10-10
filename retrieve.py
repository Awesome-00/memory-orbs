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
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))



def retrieve_memories(query, top_k=5, threshold=0.65):
    memories = get_active_memories()
    query_embedding = get_embedding(query)
    results = []
    for memory in memories:
        memory_id = memory[0]
        memory_text = memory[1]
        memory_type = memory[2]
        source_message = memory[3]
        embedding_blob = memory[4]
        created_at = memory[7]
        if embedding_blob is None:
            continue
        memory_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        if memory_embedding.shape != query_embedding.shape:
            continue
        score = cosine_similarity(query_embedding, memory_embedding)

        if score < threshold:
            continue
        results.append({
            "id": memory_id,
            "text": memory_text,
            "type": memory_type,
            "source_message": source_message,
            "score": score,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
        
    
    