# app/payment/management.py
import os
import uuid
import json
import sqlite3
from datetime import datetime, timedelta
from auth.models import DB_NAME

# Backwards compatibility (your old code used this)
# PAYMENTS_MODE = "live"
def set_subscription(identifier: str, plan: str):
    """
    Legacy shortcut: directly set user's subscription (email or username).
    Prefer using start_checkout(...) for payments.
    """
    user = _get_user(identifier)
    if not user:
        raise ValueError("User not found")
    _update_user_subscription(user["id"], plan)

# ---- Internals ----
def _connect():
    return sqlite3.connect(DB_NAME)

def _get_user(identifier: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT id, username, email, role, subscription
        FROM users WHERE email = ? OR username = ?
    """, (identifier, identifier))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "email": row[2], "role": row[3], "subscription": row[4]}

def _update_user_subscription(user_id: int, plan: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET subscription = ? WHERE id = ?", (plan, user_id))
    conn.commit()
    conn.close()

def _insert_subscription_record(user_id: int, plan: str, amount: float, status: str,
                                transaction_id: str, gateway: str = "mock",
                                meta: dict | None = None, months: int = 1):
    conn = _connect()
    c = conn.cursor()
    started_at = datetime.utcnow()
    expires_at = started_at + timedelta(days=30 * max(1, months))
    payload = json.dumps(meta or {}, ensure_ascii=False)
    c.execute("""
        INSERT INTO subscriptions (user_id, plan, amount, status, transaction_id, gateway, start_date, end_date, raw_payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, plan, amount, status, transaction_id, gateway,
          started_at.isoformat(), expires_at.isoformat(), payload))
    conn.commit()
    conn.close()

# ---- Public API ----
def start_checkout(identifier: str, plan: str = "premium", amount: float = 499.0,
                   currency: str = "INR", months: int = 1) -> dict:
    """
    Start a checkout flow. In MOCK mode: instantly 'succeeds' and updates DB.
    Later, for LIVE mode, return a payment URL and wait for webhook.
    """
    mode = os.environ.get("PAYMENTS_MODE", "mock").lower()
    user = _get_user(identifier)
    if not user:
        raise ValueError("User not found")

    if mode == "mock":
        # Simulate success
        tx = f"mock_{uuid.uuid4().hex[:12]}"
        _update_user_subscription(user["id"], plan)
        _insert_subscription_record(
            user_id=user["id"], plan=plan, amount=amount, status="success",
            transaction_id=tx, gateway="mock", meta={"currency": currency, "months": months}, months=months
        )
        return {
            "status": "success",
            "transaction_id": tx,
            "payment_url": None,
            "mode": "mock"
        }
    # Placeholder for real gateway integration (Stripe/Razorpay)
    raise NotImplementedError("Live payments not implemented yet. Set PAYMENTS_MODE=mock for dev.")


def list_recent_subscriptions(limit: int = 200):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT s.id, u.username, u.email, s.plan, s.amount, s.status,
               s.transaction_id, s.gateway, s.start_date, s.end_date
        FROM subscriptions s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.start_date DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "username": r[1], "email": r[2], "plan": r[3], "amount": r[4],
            "status": r[5], "transaction_id": r[6], "gateway": r[7],
            "start_date": r[8], "end_date": r[9]
        })
    return out

def handle_webhook(payload: dict):
    """
    Reserved for LIVE mode. For mock mode you don't need this.
    """
    return {"ok": True, "note": "Webhook handler not implemented for live mode yet."}
