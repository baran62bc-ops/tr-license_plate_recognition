import sqlite3

def init_db():
    conn = sqlite3.connect("plates.db")
    conn.execute("DROP TABLE IF EXISTS plates")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            confidence_score REAL,
            image_path TEXT,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()
