import streamlit as st
from .utils import add_user, get_user, verify_password
from .models import init_db

ADMIN_PASSKEY = "youVSyou"  # secure later

def register_user():
    st.subheader("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    role_choice = st.radio("Register as", ["User", "Admin"])
    role = "user"

    if role_choice == "Admin":
        passkey = st.text_input("Enter Admin Passkey", type="password")
        if passkey == ADMIN_PASSKEY:
            role = "admin"
        elif passkey != "":
            st.warning("⚠️ Wrong passkey! You will be registered as User.")

    if st.button("Register"):
        if username and email and password:
            try:
                add_user(username, email, password, role)
                st.success(f"✅ Registered as {role}. Please login.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please fill all fields")

def login_user():
    st.subheader("Login")
    email = st.text_input("Email (for login)")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(email)
        if user and verify_password(password, user[3]):  # user[3] = hashed password
            st.success(f"✅ Welcome {user[1]}! You are logged in as {user[4]}")
            st.session_state["user"] = {"id": user[0], "username": user[1], "role": user[4], "subscription": user[5]}
        else:
            st.error("Invalid email or password")
