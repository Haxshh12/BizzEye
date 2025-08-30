# app/auth/models.py
import sqlite3
import os

# Use a stable absolute path (project-root/database/users.db)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database"))
os.makedirs(BASE_DIR, exist_ok=True)

DB_NAME = os.path.join(BASE_DIR, "users.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # users (existing)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user',
            subscription TEXT DEFAULT 'free'
        )
    """)

    # uploads (if you already added earlier, keep it; otherwise include)
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # NEW: subscriptions
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan TEXT NOT NULL,                 -- e.g., 'premium'
            amount REAL DEFAULT 0,              -- e.g., 499.0
            status TEXT DEFAULT 'active',       -- 'success','active','cancelled','failed'
            transaction_id TEXT,                -- gateway / mock txn id
            gateway TEXT DEFAULT 'mock',        -- 'mock','stripe','razorpay'
            start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_date TIMESTAMP,                 -- expiry (e.g., +30 days)
            raw_payload TEXT,                   -- JSON string (gateway/meta)
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
