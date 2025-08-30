import pandas as pd

def basic_kpis(df: pd.DataFrame):
    total_sales = float(df["sales"].sum()) if "sales" in df.columns else 0.0
    total_orders = len(df)
    total_qty = int(df["quantity"].sum()) if "quantity" in df.columns else None
    avg_order_value = (total_sales / total_orders) if total_orders else 0.0

    top_by_category = None
    if "category" in df.columns:
        top_by_category = df.groupby("category")["sales"].sum().idxmax()

    top_product = None
    if "product" in df.columns:
        top_product = df.groupby("product")["sales"].sum().idxmax()

    growth = None
    if "date" in df.columns:
        m = df.copy()
        m["ym"] = m["date"].dt.to_period("M")
        ms = m.groupby("ym")["sales"].sum().sort_index()
        if len(ms) >= 2:
            last, prev = ms.iloc[-1], ms.iloc[-2]
            growth = ((last - prev) / prev * 100.0) if prev != 0 else None

    return {
        "total_sales": round(total_sales, 2),
        "total_orders": total_orders,
        "total_quantity": total_qty,
        "avg_order_value": round(avg_order_value, 2),
        "top_category": top_by_category,
        "top_product": top_product,
        "mom_growth_pct": round(growth, 2) if growth is not None else None
    }
