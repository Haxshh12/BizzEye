import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from auth.models import init_db
from auth.routes import register_user, login_user
from payment.management import list_recent_subscriptions #new
from payment.management import set_subscription
from data_processing.validation import validate_columns
from data_processing.transformations import normalize_dataframe
from data_processing.kpi_calculations import basic_kpis
from auth.admin_utils import list_users, list_uploads, change_role, change_subscription, delete_user, get_user, log_upload
from visualizations.basic_charts import radar_chart_from_category, gauge_chart_from_target, histogram_sales, pie_by_category
from visualizations.advanced_charts import stacked_bar_category_region,heatmap_region_product,boxplot_category_sales,trend_and_forecast,gantt_chart,cohort_retention_heatmap,rfm_segmentation_charts,clv_estimation,advanced_forecast_xgboost,ai_generated_insights,ai_insights_gemini
# ---- Page config ----
st.set_page_config(page_title="BizzEye", page_icon="👁️", layout="wide")

# ---- Free User Dashboard ----
def user_dashboard(df: pd.DataFrame):
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # Charts
    st.subheader("📈 Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(radar_chart_from_category(df), use_container_width=True)
    with c2:
        st.plotly_chart(gauge_chart_from_target(df, monthly_target=5000000.0), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(histogram_sales(df), use_container_width=True)
    with c4:
        st.plotly_chart(pie_by_category(df), use_container_width=True)

    # Insights
    st.subheader("🧠 Basic Insights")
    kpi = basic_kpis(df)
    cols = st.columns(3)
    cols[0].metric("Total Sales", f"₹{kpi['total_sales']}")
    cols[1].metric("Orders", f".{kpi['total_orders']}")
    cols[2].metric("Avg Order Value", f"₹{kpi['avg_order_value']}")
    c5 = st.columns(3)
    c5[0].write(f"**Top Category:** {kpi['top_category'] or '—'}")
    c5[1].write(f"**Top Product:** {kpi['top_product'] or '—'}")
    c5[2].write(f"**(MonthWise Growth) MoM Growth:** {str(kpi['mom_growth_pct']) + '%' if kpi['mom_growth_pct'] is not None else '—'}")

    # Upgrade CTA for free users
        # Upgrade CTA for free users
    user = st.session_state.get("user")
    if user and user.get("subscription") == "free":
        st.warning("You're on the Free plan. Unlock advanced charts, AI insights, and reports.")
        if st.button("🚀 Upgrade to Premium"):
            from payment.management import start_checkout
            try:
                identifier = user.get("email") or user.get("username")
                res = start_checkout(identifier, plan="premium", amount=499.0, currency="INR", months=1)
                if res.get("status") == "success":
                    # reflect new plan in the session
                    st.session_state["user"]["subscription"] = "premium"
                    st.success(f"Upgraded to Premium! 🎉 Transaction: {res.get('transaction_id')}")
                else:
                    st.error("Payment failed or cancelled.")
            except Exception as e:
                st.error(f"Upgrade failed: {e}")


# ---- Premium Dashboard ----
def premium_dashboard(df: pd.DataFrame):
    st.subheader("🚀 Premium Analytics Suite           (Advanced Insights..)")
    st.subheader('')

    # --- Advanced Charts ---
    st.markdown(" ")
    if {"category", "region", "sales"}.issubset(df.columns):
        st.plotly_chart(stacked_bar_category_region(df), use_container_width=True)
    else:
        st.info("ℹ️ Stacked bar (Category × Region) unavailable – missing columns.")

    if {"region", "product", "sales"}.issubset(df.columns):
        st.plotly_chart(heatmap_region_product(df), use_container_width=True)
    else:
        st.info("ℹ️ Heatmap (Region × Product) unavailable – missing columns.")

    if {"category", "sales"}.issubset(df.columns):
        st.plotly_chart(boxplot_category_sales(df), use_container_width=True)
    else:
        st.info("ℹ️ Boxplot (Category Sales) unavailable – missing columns.")

    if {"date", "sales"}.issubset(df.columns):
        trend_and_forecast(df)
    else:
        st.info("ℹ️ Trend & Forecast unavailable – missing Date/Sales columns.")

    if {"product", "start_date", "end_date"}.issubset(df.columns):
        st.plotly_chart(gantt_chart(df), use_container_width=True)
    else:
        st.info("ℹ️ Gantt chart unavailable – missing timeline columns.")

    # --- Cohort Analysis ---
    st.markdown("### 🧩 Cohort & Retention Analysis")
    cohort_retention_heatmap(df)
    # --- RFM Segmentation --regency,frequency,monetary
    rfm_segmentation_charts(df)
    # --- Customer Lifetime Value --customer retention cycle
    clv_estimation(df)
    # --- advanced forecasting ---forecasting using XgBoost
    st.markdown("### ⚙️ Advanced Forecasting (XGBoost)")
    advanced_forecast_xgboost(df)
    # --- AI insights ---(detailed in profit margin)
    ai_generated_insights(df)
    # st.markdown("### 🧠 AI-generated summarized insights)
    ai_insights_gemini(df)  #will auto-fallback if key/lib missing


# ---- Admin Dashboard ----
def admin_dashboard():
    st.markdown("___")
    st.header("👑 Admin Dashboard")
    st.markdown("---")

    # ---- Users ----
    st.subheader("👥 Users")
    users = list_users()
    if users:
        df_users = pd.DataFrame(users)
        st.dataframe(df_users, use_container_width=True)

        # selection ui
        st.markdown("----")
        options = [f"{u['id']} | {u['username']} | {u['email']} | {u['role']} | {u['subscription']}" for u in users]
        st.subheader("🕵️Select user")
        sel = st.selectbox("Select user to manage", options, key="admin_user_select")
        uid = int(sel.split("|")[0].strip())
        user = get_user(uid)
        st.write(f"Selected Member: *{user['username']}* ({user['email']}) — role: `{user['role']}`, plan: `{user['subscription']}`")
        st.write("--------------------------")

        # action buttons (with unique keys)
        c1, c2, c3 = st.columns(3)
        if c1.button("Promote to Admin 🎓", key=f"promote_{uid}"):
            change_role(uid, "admin")
            st.success("User promoted to admin")
            # st.experimental_rerun()
        if c1.button("Demote to User 🧢", key=f"demote_{uid}"):
            change_role(uid, "user")
            st.success("User demoted to user")
            # st.experimental_rerun()
        if c2.button("Upgrade to Premium 📈", key=f"upgrade_{uid}"):
            change_subscription(uid, "premium")
            st.success("User upgraded to premium")
            # st.experimental_rerun()
        if c2.button("Downgrade to Free 📉", key=f"downgrade_{uid}"):
            change_subscription(uid, "free")
            st.success("User downgraded to free")
            # st.experimental_rerun()
        # delete flow (confirm)
        if c3.button("Delete User ❌", key=f"delete_{uid}"):
            st.warning("Click Confirm to permanently delete this user and their uploads.")
            st.session_state["admin_confirm_delete"] = uid

        if st.session_state.get("admin_confirm_delete") == uid:
            if st.button("Confirm Delete", key=f"confirm_delete_{uid}"):
                delete_user(uid)
                st.success("User deleted")
                st.session_state.pop("admin_confirm_delete", None)
                # st.experimental_rerun()
    else:
        st.info("No users found in DB.")

    st.markdown("---")

    # ---- Uploads ----
    st.subheader("📃 Recent Uploads")
    uploads = list_uploads(limit=200)
    if uploads:
        df_uploads = pd.DataFrame(uploads)
        st.dataframe(df_uploads, use_container_width=True)
    else:
        st.info("No uploads recorded yet.")

        st.markdown("---")
    st.markdown("### 💳 Subscriptions")
    subs = list_recent_subscriptions(limit=200)
    if subs:
        st.dataframe(pd.DataFrame(subs), use_container_width=True)
    else:
        st.info("No subscriptions found yet.")

# ---- Main ----
def main():
    st.title("👁️BizzEye")
    init_db()

    if "user" in st.session_state and st.session_state["user"]:
        role = st.session_state["user"].get("role", "user")
        plan = st.session_state["user"].get("subscription", "free")
        st.sidebar.write(f"**Role:** {role}  |  **Plan:** {plan}")

        if role == "admin":
            admin_dashboard()
        else:
            st.header("🧾 Upload Sales CSV")
            up = st.file_uploader("Upload CSV", type=["csv"])
            if up is None:
                st.info("Upload a CSV to see your dashboard.")
                return

            try:
                df = pd.read_csv(up)
                validate_columns(df)
                df = normalize_dataframe(df)
                user = st.session_state.get("user")
                if user and user.get("id"):
                    try:
                        # optional: save uploaded file to disk (recommended)
                        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database", "uploads"))
                        os.makedirs(upload_dir, exist_ok=True)
                        safe_name = f"user{user['id']}_{int(pd.Timestamp.now().timestamp())}_{up.name}"
                        save_path = os.path.join(upload_dir, safe_name)
                        with open(save_path, "wb") as f:
                            f.write(up.getbuffer())
                        # log to DB
                        log_upload(user["id"], safe_name)
                    except Exception as e:
                        st.warning(f"Could not persist upload record: {e}")


                user_dashboard(df)

                if plan == "premium":
                    st.markdown("---")
                    premium_dashboard(df)

            except Exception as e:
                st.error(f"Error processing file: {e}")
        return

    # Auth menu
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Register":
        register_user()
    else:
        login_user()

if __name__ == "__main__":
    main()
