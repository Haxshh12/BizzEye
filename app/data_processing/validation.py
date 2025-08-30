import pandas as pd

REQUIRED_COLS = [["date","product","category","sales","quantity"],
                 ["date","product","sales"],  # fallback if no category/quantity
                 ["date","sales"]]            # minimal

def validate_columns(df: pd.DataFrame):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    for option in REQUIRED_COLS:
        if all(c in cols for c in option):
            return option
    raise ValueError(
        f"CSV missing required columns. Provide at least one of: {REQUIRED_COLS}"
    )
