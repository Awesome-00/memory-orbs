from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import ingest_message
from retrieve import retrieve_memories
from store import init_db, get_all_memories, delete_memory

app = FastAPI(title="Memory Orbs")

init_db()

class MemoryRequest(BaseModel):
    message: str

class RecallRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/memory")
def create_memory(request: MemoryRequest):
    try:
        ids = ingest_message(request.message)
        return {"saved_ids": ids, "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recall")
def recall(request: RecallRequest):
    try:
        memories = retrieve_memories(request.query, top_k=request.top_k)
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories")
def list_memories():
    rows = get_all_memories()

    memories = [
        {
            "id": row[0],
            "text": row[1],
            "type": row[2],
            "source_message": row[3],
            "status": row[5],
            "superseded_by": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]

    return {"memories": memories, "count": len(memories)}

@app.delete("/memory/{memory_id}")
def remove_memory(memory_id: int):
    deleted = delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted_id": memory_id}
