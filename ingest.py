import ollama
import json
from retrieve import get_embedding
from store import add_memory, get_active_memories, supersede_memory


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

def find_superseded_memories(new_memory, active_memories):
    if not active_memories:
        return []
    chat_messages = [
        {
            "role": "system",
            "content": """
            You identify existing memories that are directly replaced
            or contradicted by a new memory.

            Return ONLY a JSON array of the IDs of memories that are
            genuinely superseded.
            If no existing memory is replaced, return [].

            Do not select memories merely because they share a type
            or topic. Preserve unrelated facts and preferences.
            
            - Only return IDs of memories that express the same changeable factor preference as the new memory.
            - A new memory must actually contradict or explicitly update an old one.
            - Similar topics are not enough.
            - Never select a memory just because it is about the same operating system.
            - If uncertain, return an empty array.
            """
        },
        {
            "role": "user",
            "content": f"""
            New memory:
            {json.dumps(new_memory)}

            Existing active memories:
            {json.dumps(active_memories)}

            Which existing memory IDs are superseded?
            Return a JSON array of integer IDs.
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
    response_text = response.message.content
    print("LLM response:", repr(response_text))

    superseded_ids = json.loads(response_text)
    print("Parsed response:", repr(superseded_ids))
    print("Parsed type:", type(superseded_ids))
    return superseded_ids
    
    
        

if __name__ == "__main__":
    message = "I prefer C++ for competitive programming"
    saved_ids = ingest_message(message)
    print("Saved memory IDs:", saved_ids)
        
       