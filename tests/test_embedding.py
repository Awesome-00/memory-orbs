import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="User uses Arch Linux"
)

embedding = response["embeddings"][0]

print("Length:", len(embedding))
print("First 5 values:", embedding[:5])