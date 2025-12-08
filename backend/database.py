import sqlite3
import json
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "fashion.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # User Taste Vector (serialized numpy array)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_taste (
            user_id TEXT PRIMARY KEY,
            vector BLOB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Interaction Log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            item_id TEXT,
            action TEXT, -- 'like', 'dislike', 'superlike', 'save'
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize on import
init_db()
