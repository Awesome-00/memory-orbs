import sqlite3

def init_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    type TEXT NOT NULL,
    source_message TEXT NOT NULL,
    embedding BLOB,
    status TEXT DEFAULT 'active',
    superseded_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def add_memory(
    text,
    memory_type,
    source_message,
    embedding=None #fix later
    ):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO memories
        (text, type, source_message, embedding)
        VALUES (?, ?, ?, ?);
        """,
        (text, memory_type, source_message, embedding)
    )

    memory_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return memory_id

def get_all_memories():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM memories;
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

def get_active_memories():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute(
    """
    SELECT id, text, type
    FROM memories
    WHERE status = 'active';
    """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

def delete_memory(memory_id):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM memories
        WHERE id = ?;
        """,
        (memory_id,)
    )

    conn.commit()
    conn.close()

def supersede_memory(old_id, new_id):
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE memories
        SET status = 'superseded',
            superseded_by = ?
        WHERE id = ?;
        """,
        (new_id, old_id)
    )

    conn.commit()
    conn.close()
