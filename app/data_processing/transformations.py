import pandas as pd

def normalize_dataframe(df: pd.DataFrame):
    # Standardize col names
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Ensure numeric
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Drop empty rows
    df = df.dropna(subset=[c for c in ["date","sales"] if c in df.columns])
    return df

def category_aggregate(df: pd.DataFrame):
    if "category" in df.columns:
        return df.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    elif "product" in df.columns:
        return df.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    else:
        return pd.DataFrame({"label": [], "sales": []})

def monthly_sales(df: pd.DataFrame):
    if "date" not in df.columns: 
        return None
    m = df.dropna(subset=["date"]).copy()
    m["year_month"] = m["date"].dt.to_period("M").astype(str)
    return m.groupby("year_month", as_index=False)["sales"].sum().sort_values("year_month")
