import ollama
import json
from retrieve import get_embedding, cosine_similarity
from store import add_memory, get_active_memories, supersede_memory
import numpy as np


def extract_memories(message):
    system_prompt = """
    Extract meaningful, potentially useful long-term memories from the
    user's message.

    Valid memory types:
    - fact
    - preference
    - project
    - event
    - goal
    
    fact: Information about the user or their circumstances, including current tools, operating system, job, or location.
    preference: Something the user likes, dislikes, or prefers.
    project: Something the user is building or working on.
    event: Something that happened at a particular time.
    goal: Something the user wants to achieve.

    Rules:
    - Write each memory as a self-contained statement.
    - Preserve the meaning of the original message.
    - Do not invent or assume information.
    - Ignore small talk and messages without useful memories.
    - Return an empty JSON array if nothing is worth remembering.

    Return ONLY a valid JSON array. Each element must be an object
    with exactly two fields: "text" and "type".

    Example:
    [
        {
            "text": "User prefers C++ for competitive programming",
            "type": "preference"
        }
    ]

    If nothing is worth remembering, return:
    []
    """
    memory_schema = {
        "type": "array",
        "items": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string"
            },
            "type": {
                "type": "string",
                "enum": ["fact", "preference", "project", "event", "goal"]
            }
        },
        "required": ["text", "type"],
        "additionalProperties": False
    }
    }

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": message
        }
    ]

    response = ollama.chat(
        model="gemma3:4b",
        messages=chat_messages,
        format=memory_schema
    )

    response_text = response.message.content
    extracted_memories = json.loads(response_text)
    for memory in extracted_memories:
        if not isinstance(memory, dict):
            raise ValueError("Each memory must be a dictionary")
        
        required_fields = {"text", "type"}
        if not required_fields.issubset(memory.keys()):
            raise ValueError("all required fields are necessary")
        
        if not isinstance(memory["text"], str) or not memory["text"].strip():
            raise ValueError("Memory text must be a non-empty string")
        
        valid_types = {"fact", "preference", "project", "event", "goal"}
        if memory["type"] not in valid_types:
            raise ValueError("Invalid memory type")

    return extracted_memories

def ingest_message(message):
    if not message.strip():
        return []

    extracted_memories = extract_memories(message)
    memory_ids = []

    for memory in extracted_memories:
        active_memories = get_active_memories()
        
        active_memories = [
        row for row in get_active_memories()
        if row[0] not in memory_ids
        ]

        superseded_ids = find_superseded_memories(
            memory, active_memories
        )

        active_ids = {row[0] for row in active_memories}
        superseded_ids = [
            old_id for old_id in superseded_ids
            if isinstance(old_id, int) and old_id in active_ids
        ]

        embedding = get_embedding(memory["text"])
        embedding_bytes = embedding.tobytes()

        memory_id = add_memory(
            memory["text"],
            memory["type"],
            message,
            embedding_bytes
        )

        for old_id in superseded_ids:
            supersede_memory(old_id, memory_id)

        memory_ids.append(memory_id)

    return memory_ids

SUPERSEDABLE_TYPES = {"fact", "preference"} 

def find_superseded_memories(new_memory, active_memories):
    if not active_memories:
        return []
    if new_memory["type"] not in SUPERSEDABLE_TYPES:
        return []
    if not active_memories:
        return []

    same_type = [
        row for row in active_memories
        if row[2] == new_memory["type"] and row[4] is not None
    ]

    if not same_type:
        return []

    new_embedding = get_embedding(new_memory["text"])
    candidates = []

    for row in same_type:
        old_embedding = np.frombuffer(row[4], dtype=np.float32)

        if old_embedding.shape != new_embedding.shape:
            continue

        score = cosine_similarity(new_embedding, old_embedding)

        if score >= 0.75:
            candidates.append({
                "id": row[0],
                "text": row[1],
                "type": row[2],
                "similarity": round(score, 3),
            })

    if not candidates:
        return []

    chat_messages = [
        {
            "role": "system",
            "content": """
            You identify memories that are directly replaced or contradicted
            by a new memory.

            Return ONLY a JSON array of integer IDs.

            A memory may be superseded only if it describes the SAME specific,
            changeable fact or preference and the new memory explicitly updates
            or contradicts it.

            Examples:
            - "My laptop has 16 GB RAM" -> "My laptop has 32 GB RAM":
              supersede the old memory.
            - "I use Python" -> "I use C++ for competitive programming":
              do not supersede; both can be true.
            - "I use Zorin OS" -> "My laptop has 32 GB RAM":
              do not supersede.
            - "I want to learn networking" -> "I am considering CCNA":
              do not supersede.

            Similarity or a shared type/topic is NOT sufficient.
            If uncertain, return [].
            """
        },
                {
                    "role": "user",
                    "content": f"""
        New memory:
        {json.dumps(new_memory)}

        Candidate existing memories:
        {json.dumps(candidates)}

        Return the IDs of memories genuinely superseded by the new memory.
        """
                }
    ]

    response = ollama.chat(
        model="gemma3:4b",
        messages=chat_messages,
        format={
            "type": "array",
            "items": {"type": "integer"}
        }
    )

    try:
        superseded_ids = json.loads(response.message.content)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(superseded_ids, list):
        return []

    valid_ids = {memory["id"] for memory in candidates}

    return [
        memory_id
        for memory_id in superseded_ids
        if isinstance(memory_id, int)
        and not isinstance(memory_id, bool)
        and memory_id in valid_ids
    ]
    
    
        

if __name__ == "__main__":
    message = "I prefer C++ for competitive programming"
    saved_ids = ingest_message(message)
    print("Saved memory IDs:", saved_ids)
        
       