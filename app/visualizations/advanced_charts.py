# app/visualizations/advanced_charts.py
"""
Premium visualizations & analytics for Bizz_Eye.
Includes:
- Stacked bar, heatmap, boxplot, trend/forecast, gantt
- Cohort & Retention, RFM Segmentation, CLV
- Market Basket Analysis, Advanced Forecast (XGBoost)
- AI-style Insights
"""
import os,json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional deps
try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLBASKET_OK = True
except Exception:
    MLBASKET_OK = False


# ---------------- Helpers ---------------- #
def _cols(df: pd.DataFrame):
    return {c.lower() for c in df.columns}

def _lc(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

def _require(df: pd.DataFrame, needed: set[str]) -> bool:
    miss = needed - _cols(df)
    if miss:
        st.warning(f"⚠️ Missing columns: {', '.join(miss)}")
        return False
    return True


# ---------------- Basic Advanced Charts ---------------- #
def stacked_bar_category_region(df: pd.DataFrame):
    if not _require(df, {"category", "region", "sales"}): return go.Figure()
    cat, reg, sal = _lc(df, "category"), _lc(df, "region"), _lc(df, "sales")
    agg = df.groupby([cat, reg], as_index=False)[sal].sum()
    return px.bar(agg, x=cat, y=sal, color=reg, barmode="stack", title="📊 Sales by Category & Region")

def heatmap_region_product(df: pd.DataFrame):
    if not _require(df, {"region", "product", "sales"}): return go.Figure()
    reg, prod, sal = _lc(df, "region"), _lc(df, "product"), _lc(df, "sales")
    pivot = df.pivot_table(values=sal, index=reg, columns=prod, aggfunc="sum", fill_value=0)
    return px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="🌍 Sales Heatmap")

def boxplot_category_sales(df: pd.DataFrame):
    if not _require(df, {"category", "sales"}): return go.Figure()
    cat, sal = _lc(df, "category"), _lc(df, "sales")
    return px.box(df, x=cat, y=sal, points="all", title="📦 Sales Distribution by Category")

def trend_and_forecast(df: pd.DataFrame):
    if not _require(df, {"date", "sales"}): return
    dt, sal = _lc(df, "date"), _lc(df, "sales")
    work = df.copy(); work[dt] = pd.to_datetime(work[dt], errors="coerce")
    ts = work.groupby(dt)[sal].sum().reset_index(); ts["t"] = range(len(ts))
    if len(ts) < 2: return st.info("Not enough data for forecast.")
    coeffs = np.polyfit(ts["t"], ts[sal], 1)
    ts["forecast"] = coeffs[0] * ts["t"] + coeffs[1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts[dt], y=ts[sal], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=ts[dt], y=ts["forecast"], mode="lines", name="Forecast", line=dict(dash="dash")))
    fig.update_layout(title="🔮 Sales Trend & Forecast")
    st.plotly_chart(fig, use_container_width=True)

def gantt_chart(df: pd.DataFrame):
    if not _require(df, {"product", "start_date", "end_date"}): return go.Figure()
    prod, start, end = _lc(df, "product"), _lc(df, "start_date"), _lc(df, "end_date")
    work = df.copy(); work[start] = pd.to_datetime(work[start]); work[end] = pd.to_datetime(work[end])
    fig = px.timeline(work, x_start=start, x_end=end, y=prod, color=prod, title="📅 Product Launch Timeline")
    fig.update_yaxes(autorange="reversed"); return fig


# ---------------- Premium Additions ---------------- #
def cohort_retention_heatmap(df: pd.DataFrame):
    if not _require(df, {"customerid", "date"}): 
        return
    
    cust, dt = _lc(df, "customerid"), _lc(df, "date")
    work = df[[cust, dt]].dropna().copy()
    work[dt] = pd.to_datetime(work[dt])
    work["order_month"] = work[dt].values.astype("datetime64[M]")

    # Assign cohort (first purchase month per customer)
    first = work.groupby(cust)["order_month"].min().rename("cohort")
    work = work.merge(first, on=cust)

    # Cohort period (months since first purchase)
    work["period"] = ((work["order_month"].dt.year - work["cohort"].dt.year) * 12 +
                      (work["order_month"].dt.month - work["cohort"].dt.month))

    # Unique active customers per cohort-period
    cohort = work.groupby(["cohort", "period"])[cust].nunique().reset_index()

    # Cohort sizes (number of unique customers in period 0)
    sizes = cohort[cohort["period"] == 0][["cohort", cust]].rename(columns={cust: "size"})
    df_ret = cohort.merge(sizes, on="cohort")
    df_ret["retention"] = df_ret[cust] / df_ret["size"]

    # Pivot for heatmap
    pivot = df_ret.pivot(index="cohort", columns="period", values="retention").fillna(0)

    # Better axis labels
    pivot.index = pivot.index.strftime("%b %Y")  # format cohorts
    pivot.columns = [f"Month {p}" for p in pivot.columns]

    # Heatmap with professional styling
    fig = px.imshow(
        pivot,
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale="Blues",
        title="📊 Cohort Retention Heatmap",
        labels=dict(x="Cohort Period (Months Since First Purchase)", y="Cohort (First Purchase Month)", color="Retention Rate")
    )

    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20, family="Arial", color="black")),
        coloraxis_colorbar=dict(title="Retention", tickformat=".0%"),
        xaxis_title="Cohort Period",
        yaxis_title="Cohort Start Month",
        font=dict(size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Add cohort sizes on y-axis
    cohort_sizes = sizes.set_index("cohort")["size"]
    cohort_sizes.index = cohort_sizes.index.strftime("%b %Y")
    annotations = []
    for i, cohort in enumerate(pivot.index):
        annotations.append(
            dict(x=-0.5, y=i, text=f" (n={cohort_sizes[cohort]})", 
                 showarrow=False, xref="x", yref="y", 
                 xanchor="right", font=dict(size=11, color="gray"))
        )
    fig.update_layout(annotations=annotations)

    # Show
    st.plotly_chart(fig, use_container_width=True)

    # Add short professional insights below chart
    st.markdown("""
    **Interpretation:**  
    - Darker cells indicate stronger customer retention.  
    - Cohort sizes are shown on the right (n = number of new customers in that month).  
    - This view helps track how customer loyalty evolves over time and identifies strong or weak retention months.  
    """)


def rfm_segmentation_charts(df: pd.DataFrame):
    if not _require(df, {"customerid", "date", "sales"}): return
    cust, dt, sal = _lc(df,"customerid"), _lc(df,"date"), _lc(df,"sales")
    work = df[[cust, dt, sal]].dropna().copy(); work[dt]=pd.to_datetime(work[dt])
    max_date = work[dt].max()
    rfm = work.groupby(cust).agg(
        recency=(dt,lambda x:(max_date-x.max()).days),
        frequency=(dt,"count"), monetary=(sal,"sum")).reset_index()
    rfm["R"] = pd.qcut(-rfm["recency"],5,labels=False)+1
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=False)+1
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"),5,labels=False)+1
    rfm["Score"] = rfm[["R","F","M"]].sum(axis=1)
    rfm["Segment"] = pd.cut(rfm["Score"], bins=[2,5,7,10,13,15], labels=["Hibernating","At Risk","Promising","Loyal","VIP"])
    st.subheader("🧭 RFM Segmentation"); st.dataframe(rfm.head(), use_container_width=True)
    seg_counts = rfm["Segment"].value_counts().reset_index()
    st.plotly_chart(px.bar(seg_counts, x="Segment", y="count", text_auto=True, title="Segment Distribution"), use_container_width=True)

def clv_estimation(df: pd.DataFrame):
    if not _require(df, {"customerid", "date", "sales"}):
        return

    cust, dt, sal = _lc(df, "customerid"), _lc(df, "date"), _lc(df, "sales")
    work = df[[cust, dt, sal]].copy()
    work[dt] = pd.to_datetime(work[dt], errors="coerce")
    work["month"] = work[dt].values.astype("datetime64[M]")

    # Monthly revenue per customer
    monthly = work.groupby([cust, "month"])[sal].sum().reset_index()

    # Active customers per month
    active = monthly.groupby("month")[cust].nunique()
    if len(active) < 3:
        return st.info("📊 Insufficient historical data to produce a reliable CLV model.")

    # Retention Rate (average customer survival month-to-month)
    retention = (active / active.shift(1)).dropna().clip(0, 1.2).mean()

    # ARPU (Average Revenue Per User per month)
    arpu = monthly.groupby("month")[sal].sum().mean() / active.mean()

    # Customer Lifetime Value (geometric retention-adjusted ARPU)
    clv = arpu * (retention / (1 + 0.01 - retention))

    # Trend analysis
    monthly_revenue = monthly.groupby("month")[sal].sum()
    revenue_growth = (
        (monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0] * 100
        if len(monthly_revenue) > 1 else 0
    )

    # Presentation block
    st.subheader("💎 Customer Lifetime Value Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lifetime Value (CLV)", f"${clv:,.2f}")
    col2.metric("Retention Rate", f"{retention*100:.1f}%")
    col3.metric("ARPU (Monthly)", f"${arpu:,.2f}")

    # Secondary insights
    st.write("### 📊 Strategic Insights")
    st.write(f"- **Revenue Growth:** {revenue_growth:+.2f}% since first recorded month.")
    st.write(f"- **Retention Sensitivity:** A 5% improvement in retention could lift CLV by ~{(clv*0.05):,.2f}.")
    st.write("- **Business Implication:** Sustained retention is the strongest driver of lifetime value.")

    # Optional visualization: CLV or ARPU over time
    fig = px.line(monthly_revenue.reset_index(), x="month", y=sal, 
                  title="Monthly Revenue Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)


def advanced_forecast_xgboost(df: pd.DataFrame):
    if not _require(df, {"date","sales"}): return
    if not XGB_OK: return st.warning("Install xgboost for advanced forecasting.")
    dt,sal=_lc(df,"date"),_lc(df,"sales")
    work=df[[dt,sal]].dropna().copy(); work[dt]=pd.to_datetime(work[dt])
    ts=work.groupby(pd.Grouper(key=dt,freq="M"))[sal].sum().reset_index()
    if len(ts)<24: return st.info("Need ≥24 months.")
    ts["t"]=np.arange(len(ts)); ts["month"]=ts[dt].dt.month
    ts["lag1"]=ts[sal].shift(1); ts=ts.dropna()
    split=int(len(ts)*0.8); train,test=ts.iloc[:split],ts.iloc[split:]
    model=xgb.XGBRegressor(n_estimators=300); model.fit(train[["t","month","lag1"]],train[sal])
    pred=model.predict(test[["t","month","lag1"]])
    fig=go.Figure([go.Scatter(x=train[dt],y=train[sal],name="Train"),
                   go.Scatter(x=test[dt],y=test[sal],name="Test"),
                   go.Scatter(x=test[dt],y=pred,name="Forecast",line=dict(dash="dash"))])
    fig.update_layout(title="⚙️ XGBoost Forecast"); st.plotly_chart(fig,use_container_width=True)

def ai_generated_insights(df: pd.DataFrame):
    if "sales" not in _cols(df): 
        return

    sal = _lc(df, "sales")
    insights = []

    # Core financials
    total_sales = df[sal].sum()
    avg_sales = df[sal].mean()
    median_sales = df[sal].median()
    sales_volatility = df[sal].std()

    insights.append(f"💰 Total Sales: ${total_sales:,.0f}")
    insights.append(f"📊 Average Sale Value: ${avg_sales:,.2f} (Median: ${median_sales:,.2f})")
    insights.append(f"📉 Sales Volatility: {sales_volatility:,.2f} (Std. Dev)")

    if not {"sales", "profit"}.issubset(set(_cols(df))):
        return

    sal = _lc(df, "sales")
    prof = _lc(df, "profit")
    insights = []

    # Overall profitability
    total_sales = df[sal].sum()
    total_profit = df[prof].sum()
    overall_margin = (total_profit / total_sales * 100) if total_sales else 0
    insights.append(f"💹 Overall Profit Margin: {overall_margin:.2f}%")

    # Category-level profitability
    if "category" in _cols(df):
        cat = _lc(df, "category")
        cat_group = df.groupby(cat).agg({sal: "sum", prof: "sum"})
        cat_group["margin"] = (cat_group[prof] / cat_group[sal] * 100).round(2)

        # Top & bottom margin categories
        top_cat = cat_group["margin"].idxmax()
        bottom_cat = cat_group["margin"].idxmin()

        insights.append(f"🏆 Highest Margin Category: {top_cat} ({cat_group.loc[top_cat,'margin']:.2f}%)")
        insights.append(f"⚠️ Lowest Margin Category: {bottom_cat} ({cat_group.loc[bottom_cat,'margin']:.2f}%)")

        # Optional: average margin across categories
        avg_margin = cat_group["margin"].mean()
        insights.append(f"📊 Average Category Margin: {avg_margin:.2f}%")
    # Category performance
    if "category" in _cols(df):
        cat = _lc(df, "category")
        cat_sales = df.groupby(cat)[sal].sum().sort_values(ascending=False)
        share = (cat_sales / total_sales * 100).round(2)
        top_cat, bottom_cat = cat_sales.idxmax(), cat_sales.idxmin()
        insights.append(f"🏆 {top_cat} leads with {share[top_cat]}% of total revenue")
        insights.append(f"⚠️ {bottom_cat} contributes only {share[bottom_cat]}% — potential underperformer")

    # Customer analytics
    if "customername" in _cols(df):
        cust = _lc(df, "customername")
        cust_sales = df.groupby(cust)[sal].sum().sort_values(ascending=False)
        top_customer = cust_sales.idxmax()
        top_share = cust_sales.iloc[0] / total_sales * 100
        insights.append(f"👤 Top Customer: {top_customer} ({top_share:.2f}% of revenue)")
        if len(cust_sales) > 1:
            top5_share = cust_sales.head(5).sum() / total_sales * 100
            insights.append(f"⭐ Top 5 customers drive {top5_share:.2f}% of revenue")

    # Time-series performance
    if "date" in _cols(df):
        date = _lc(df, "date")
        df[date] = pd.to_datetime(df[date], errors="coerce")
        monthly_sales = df.groupby(df[date].dt.to_period("M"))[sal].sum()
        if len(monthly_sales) > 1:
            growth = (monthly_sales.iloc[-1] - monthly_sales.iloc[0]) / monthly_sales.iloc[0] * 100
            cagr = ((monthly_sales.iloc[-1] / monthly_sales.iloc[0]) ** (1/(len(monthly_sales)-1)) - 1) * 100
            trend = "📈 Uptrend" if growth > 0 else "📉 Downtrend"
            insights.append(f"{trend}: {growth:.2f}% total change | CAGR: {cagr:.2f}%")

    # Profitability (if profit column exists)
    if "profit" in _cols(df):
        prof = _lc(df, "profit")
        total_profit = df[prof].sum()
        margin = total_profit / total_sales * 100
        insights.append(f"💹 Total Profit: ${total_profit:,.0f} | Margin: {margin:.2f}%")

    # Display
    st.subheader("🧠 AI-Driven Insights")
    for i in insights:
        st.write("• " + i)


def trend_and_forecast(df: pd.DataFrame):
    cols = _cols(df)
    if not {"date", "sales"}.issubset(cols):
        st.warning("⚠️ Need columns: date, sales.")
        return go.Figure()

    dt = _lc(df, "date")
    sal = _lc(df, "sales")

    work = df.copy()
    work[dt] = pd.to_datetime(work[dt], errors="coerce")
    ts = work.groupby(dt)[sal].sum().reset_index()
    ts["t"] = range(len(ts))

    if len(ts) < 2:
        st.info("Not enough data points for trend/forecast.")
        return go.Figure()

    coeffs = np.polyfit(ts["t"], ts[sal], 1)
    ts["forecast"] = coeffs[0] * ts["t"] + coeffs[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts[dt], y=ts[sal], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=ts[dt], y=ts["forecast"], mode="lines", name="Forecast", line=dict(dash="dash")))
    fig.update_layout(title="🔮 Sales Trend & Forecast")

    return fig   # ✅ return, don’t just st.plotly_chart

def _build_llm_summary_payload(df: pd.DataFrame) -> dict:
    """Compact summary of sales for LLM (don’t send raw rows)."""
    sal = _lc(df, "sales")
    if sal is None:
        return {"kpi": {}, "notes": ["No 'sales' column found."]}

    def safe_col(name): 
        c = _lc(df, name); return c if c in df.columns else None

    dt = safe_col("date")
    cat = safe_col("category")
    reg = safe_col("region")
    prod = safe_col("product")

    payload = {}

    # KPIs
    total_sales = float(df[sal].sum())
    orders = int(df.shape[0])
    avg_order_value = float(df[sal].mean()) if orders else 0.0
    payload["kpi"] = {
        "total_sales": total_sales,
        "orders": orders,
        "avg_order_value": avg_order_value
    }

    # Monthly trend
    if dt:
        work = df[[dt, sal]].dropna().copy()
        work[dt] = pd.to_datetime(work[dt], errors="coerce")
        m = work.groupby(work[dt].dt.to_period("M"))[sal].sum().to_timestamp().reset_index()
        payload["monthly_sales"] = [
            {"month": str(r[dt].date()), "sales": float(r[sal])} for _, r in m.iterrows()
        ]
    else:
        payload["monthly_sales"] = []

    # Aggregations
    def topn(col, n=8):
        if col:
            s = df.groupby(col)[sal].sum().sort_values(ascending=False).head(n)
            return [{"name": k, "sales": float(v)} for k, v in s.items()]
        return []

    payload["by_category"] = topn(cat, 8)
    payload["by_region"] = topn(reg, 8)
    payload["top_products"] = topn(prod, 12)

    return payload


def ai_insights_gemini(df: pd.DataFrame, model_name: str = "gemini-1.5-flash"):
    """
    If GEMINI_API_KEY is available (st.secrets or env), call Gemini to generate
    narrative insights. Otherwise, fall back to rule-based insights.
    """
    # Try to get key from Streamlit secrets or env
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    # If no key or library missing → fallback
    try:
        import google.generativeai as genai  # lazy import
    except Exception:
        st.info("ℹ️ Gemini client not installed. Using local rule-based insights.")
        return ai_generated_insights(df)

    if not api_key:
        st.info("ℹ️ No GEMINI_API_KEY configured. Using local rule-based insights.")
        return ai_generated_insights(df)

    # Build compact payload for LLM
    payload = _build_llm_summary_payload(df)
    payload_json = json.dumps(payload, ensure_ascii=False)

    prompt = f"""
You are a senior business analyst. Based on the JSON summary of a company's sales,
produce clear, non-generic, actionable insights. Focus on trends, anomalies, and
what to do next. Keep it concise: 5–10 bullets plus a short action plan.

JSON summary:
{payload_json}

Guidelines:
- Use concrete numbers (% and amounts) where possible from the summary.
- If monthly series exists, analyze recent momentum and volatility.
- Call out top/bottom categories, regions, and products with likely reasons.
- Add 3–5 prioritized actions (pricing, promo, inventory, territory focus).
- Write in professional tone. Avoid repeating the raw JSON back.
"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or (
            "\n".join([p.text for p in getattr(resp, "candidates", []) if hasattr(p, "text")]) or ""
        )
        if not text.strip():
            raise RuntimeError("Empty response from model.")
        st.subheader("🧠 AI-Generated Insights (Gemini)")
        st.markdown(text)
    except Exception as e:
        st.warning(f"AI insight generation failed ({e}). Showing local insights instead.")
        return ai_generated_insights(df)
