import ollama
import numpy as np

from store import init_db, add_memory
response = ollama.embed(
    model="nomic-embed-text",
    input="User uses Arch Linux"
)

embedding = np.array(
    response["embeddings"][0],
    dtype=np.float32
)
embedding_blob = embedding.tobytes()
init_db()

memory_id = add_memory(
    text="User uses Arch Linux",
    memory_type="fact",
    source_message="I use Arch Linux.",
    embedding=embedding_blob
)

print(memory_id)
from store import get_all_memories
rows = get_all_memories()
rows = get_all_memories()

print(rows[-1])

last_memory = rows[-1]

blob = last_memory[4]

restored_embedding = np.frombuffer(
    blob,
    dtype=np.float32
)

print(type(restored_embedding))
print(len(restored_embedding))
print(restored_embedding[:5])
print(rows[-1])
restored_embedding = np.frombuffer(
    blob,
    dtype=np.float32
)

print(type(restored_embedding))
print(len(restored_embedding))
print(restored_embedding[:5])
print(len(restored_embedding))