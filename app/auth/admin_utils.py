# app/auth/admin_utils.py
import sqlite3
from .models import DB_NAME
from typing import List, Dict
from datetime import datetime, timedelta

def list_users() -> List[Dict]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, email, role, subscription FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    users = []
    for r in rows:
        users.append({"id": r[0], "username": r[1], "email": r[2], "role": r[3], "subscription": r[4]})
    return users

def get_user(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, email, role, subscription FROM users WHERE id=?", (user_id,))
    r = c.fetchone()
    conn.close()
    if not r:
        return None
    return {"id": r[0], "username": r[1], "email": r[2], "role": r[3], "subscription": r[4]}

def change_role(user_id: int, new_role: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()
    conn.close()

def change_subscription(user_id: int, plan: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET subscription=? WHERE id=?", (plan, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # delete uploads for that user first (optional safety)
    c.execute("DELETE FROM uploads WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def log_upload(user_id: int, filename: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (user_id, filename))
    conn.commit()
    conn.close()

def list_uploads(limit: int = 200):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT uploads.id, uploads.user_id, users.username, uploads.filename, uploads.uploaded_at
        FROM uploads 
        LEFT JOIN users ON uploads.user_id = users.id
        ORDER BY uploads.uploaded_at DESC
        LIMIT ?
    """, (limit,))
    
    rows = c.fetchall()
    uploads = []
    for row in rows:
        utc_time = datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")
        ist_time = utc_time + timedelta(hours=5, minutes=30)
        uploads.append((row[0], row[1], row[2], row[3], ist_time.strftime("%Y-%m-%d %H:%M:%S")))
    
    return uploads