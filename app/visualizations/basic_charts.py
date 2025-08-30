import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# --- helper: group top N + "Others" ---
def _topn_with_others(df: pd.DataFrame, label_col: str, value_col: str, topn: int = 8):
    agg = (df.groupby(label_col, as_index=False)[value_col]
             .sum()
             .sort_values(value_col, ascending=False))
    if len(agg) <= topn:
        return agg.rename(columns={label_col: "label"})
    top = agg.iloc[:topn].copy()
    others_sum = agg.iloc[topn:][value_col].sum()
    if others_sum > 0:
        top = pd.concat([top, pd.DataFrame({"label": ["Others"], value_col: [others_sum]})], ignore_index=True)
        top.rename(columns={label_col: "label"}, inplace=True)
    else:
        top.rename(columns={label_col: "label"}, inplace=True)
    return top

# --- RADAR (cleaner scales, markers, top-N focus) ---
def radar_chart_from_category(df: pd.DataFrame, topn: int = 8):
    label_col = "category" if "category" in df.columns else ("product" if "product" in df.columns else None)
    if (label_col is None) or ("sales" not in df.columns) or df.empty:
        return go.Figure()

    agg = _topn_with_others(df, label_col, "sales", topn)
    labels = agg["label"].tolist()
    values = agg["sales"].astype(float).tolist()

    # close the loop for polar
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill='toself',
            mode='lines+markers',
            marker=dict(size=6),
            line=dict(width=2)
        )
    )
    fig.update_layout(
        title=f"Sales Radar by {('Category' if label_col=='category' else 'Product')} (Top {topn})",
        polar=dict(radialaxis=dict(visible=True, tickformat=",", showline=True, gridcolor="lightgray")),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig
def gauge_chart_from_target(df, monthly_target=5000000):
    total_sales = float(df["sales"].sum()) if "sales" in df.columns else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_sales,
        number={'prefix':'₹', 'valueformat':',.0f'},
        title={'text': f"Progress vs Target (₹{monthly_target:,})"},
        delta={'reference': monthly_target, 'increasing': {'color': "green"}, 'decreasing': {'color':'red'}},
        gauge={
            'axis': {'range': [0, monthly_target]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5*monthly_target], 'color':'red'},
                {'range': [0.5*monthly_target, 0.8*monthly_target], 'color':'yellow'},
                {'range': [0.8*monthly_target, monthly_target], 'color':'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': monthly_target
            }
        }
    ))
    return fig

def histogram_sales(df: pd.DataFrame):
    if ("sales" not in df.columns) or df["sales"].dropna().empty:
        return go.Figure()

    s = df["sales"].dropna().astype(float)
    n = len(s)
    iqr = np.subtract(*np.percentile(s, [75, 25]))
    bin_width = 2 * iqr * (n ** (-1/3)) if iqr > 0 else None
    if bin_width and bin_width > 0:
        nbins = max(10, int(np.ceil((s.max() - s.min()) / bin_width)))
    else:
        nbins = 30

    fig = px.histogram(df, x="sales", nbins=nbins, title="Sales Distribution", marginal=None)
    # mean line
    mean_val = s.mean()
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash", annotation_text=f"Mean: {mean_val:,.0f}", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Sales",
        yaxis_title="Count",
        bargap=0.02,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig

# --- PIE (donut style + top-N + %-labels) ---
def pie_by_category(df: pd.DataFrame, topn: int = 8):
    label_col = "category" if "category" in df.columns else ("product" if "product" in df.columns else None)
    if (label_col is None) or ("sales" not in df.columns) or df.empty:
        return go.Figure()

    top = _topn_with_others(df, label_col, "sales", topn)
    fig = px.pie(
        top,
        names="label",
        values="sales",
        hole=0.4,
        title=f"Sales Share by {('Category' if label_col=='category' else 'Product')} (Top {topn} + Others)"
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(l=30, r=30, t=60, b=30))
    return fig