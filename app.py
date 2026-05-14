"""
app.py — Main Streamlit entry-point.

Run:   streamlit run app.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader   import get_rfm, get_transactions, RFM_CACHE, TXN_CACHE
from utils.segmentation  import (
    run_segmentation_pipeline, plot_umap, plot_elbow, plot_silhouette,
    segment_summary, SEGMENT_COLORS
)
from utils.recommender   import (
    run_recommender_pipeline, hybrid_recommend, cb_recommend_by_product
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg:       #0d0f1a;
    --card:     #141624;
    --border:   #252840;
    --accent1:  #00C9A7;
    --accent2:  #845EC2;
    --accent3:  #4B9FEA;
    --text:     #e2e4f0;
    --muted:    #8890aa;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: var(--accent1); font-family: 'Space Mono', monospace; }
.metric-card .label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

.segment-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 4px;
}

.rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rec-rank { font-family: 'Space Mono', monospace; color: var(--accent2); font-weight: 700; width: 30px; }
.rec-name { flex: 1; padding: 0 12px; }
.rec-score { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: var(--accent1); }

h1, h2, h3 { font-family: 'Sora', sans-serif; }

.stTabs [data-baseweb="tab"] { font-family: 'Sora', sans-serif; }
div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Session-state cache helpers ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading & processing dataset …")
def load_all():
    # Auto-generate demo data on Streamlit Cloud (data/ is gitignored)
    data_file = os.path.join(os.path.dirname(__file__), "data", "online_retail_II.xlsx")
    if not os.path.exists(data_file):
        import generate_demo_data  # noqa: runs as side-effect, creates the file
    rfm_raw  = get_rfm()
    txn      = get_transactions()
    seg      = run_segmentation_pipeline(rfm_raw)
    rec      = run_recommender_pipeline(txn, evaluate=False)
    return rfm_raw, txn, seg, rec


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Customer Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "🎯 Customer Lookup",
         "📊 Cluster Explorer",
         "🛍️ Product Similarity"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** Online Retail II (UCI)")
    st.markdown("**Model:** K-Means + DBSCAN + SVD + TF-IDF")
    st.caption("Built for placement portfolio · BTech Final Year")


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    rfm_raw, txn, seg, rec = load_all()
    rfm     = seg["rfm"]
    metrics = seg["metrics"]
    em_metrics = rec.get("eval_metrics", {})
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# Customer Intelligence Hub")
    st.markdown("RFM Segmentation · Hybrid Recommendations · Interactive Analytics")
    st.markdown("---")

    # Top-level KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(rfm):,}",                   "Customers"),
        (f"{seg['optimal_k']}",              "Segments"),
        (f"{metrics['kmeans_silhouette']:.3f}", "Silhouette"),
        (f"{metrics['kmeans_db']:.3f}",      "Davies-Bouldin"),
        (f"£{rfm['Monetary'].sum():,.0f}",   "Total Revenue"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="value">{val}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # UMAP
    st.subheader("📍 UMAP Cluster Projection")
    fig_umap = plot_umap(rfm, seg["embedding"])
    st.plotly_chart(fig_umap, use_container_width=True)

    # Elbow + Silhouette side-by-side
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_elbow(seg["k_results"]), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_silhouette(seg["k_results"]), use_container_width=True)

    # Segment summary table
    st.subheader("📋 Segment Summary")
    summary = segment_summary(rfm)
    st.dataframe(
        summary.style.format({
            "Avg_Recency": "{:.1f} days",
            "Avg_Frequency": "{:.1f}",
            "Avg_Monetary": "£{:.2f}",
            "Total_Revenue": "£{:,.0f}",
            "Revenue_%": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Revenue pie
    fig_pie = go.Figure(go.Pie(
        labels=summary["Segment"],
        values=summary["Total_Revenue"],
        hole=0.45,
        marker=dict(colors=[SEGMENT_COLORS.get(s, "#888") for s in summary["Segment"]]),
    ))
    fig_pie.update_layout(
        title="Revenue Contribution by Segment",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Model metrics
    if em_metrics:
        st.subheader("🎯 Recommender Evaluation")
        mc1, mc2, mc3 = st.columns(3)
        for col, (k, v) in zip([mc1, mc2, mc3], em_metrics.items()):
            with col:
                st.metric(k, str(v))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER LOOKUP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Lookup":
    st.markdown("# 🎯 Customer Lookup")
    st.markdown("Enter a Customer ID to view their segment and personalised recommendations.")
    st.markdown("---")

    all_customers = sorted(rfm["CustomerID"].unique())
    customer_id   = st.selectbox(
        "Select or type Customer ID",
        options=all_customers,
        index=0,
    )

    if customer_id:
        row = rfm[rfm["CustomerID"] == customer_id]
        if row.empty:
            st.warning("Customer not found in RFM table.")
        else:
            row = row.iloc[0]
            seg_name  = row["Segment"]
            seg_color = SEGMENT_COLORS.get(seg_name, "#888")

            # Badge + RFM
            st.markdown(
                f'<div style="margin:20px 0;">'
                f'<span class="segment-badge" style="background:{seg_color}22;'
                f'border:1.5px solid {seg_color};color:{seg_color};">'
                f'⬤ &nbsp;{seg_name}</span></div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Recency",   f"{int(row.Recency)} days")
            col2.metric("Frequency", f"{int(row.Frequency)} orders")
            col3.metric("Monetary",  f"£{row.Monetary:,.2f}")

            st.markdown("---")
            st.subheader("🛒 Top 10 Recommendations")

            alpha = st.slider("CF ↔ CB blend (alpha = CF weight)", 0.0, 1.0, 0.6, 0.05)

            with st.spinner("Generating recommendations …"):
                recs = hybrid_recommend(
                    customer_id,
                    algo=rec["algo"],
                    items=rec["items"],
                    tfidf_matrix=rec["tfidf_matrix"],
                    ui=rec["ui"],
                    alpha=alpha,
                    top_n=10,
                )

            if recs.empty:
                st.info("Not enough purchase history to generate recommendations.")
            else:
                for i, r in recs.iterrows():
                    desc  = str(r.get("Description", "N/A"))[:70]
                    score = float(r["Hybrid_Score"])
                    bar_w = int(score * 100)
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<span class="rec-rank">#{i+1}</span>'
                        f'<span class="rec-name">{desc}</span>'
                        f'<span class="rec-score">{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTER EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Cluster Explorer":
    st.markdown("# 📊 Cluster Explorer")
    st.markdown("---")

    segments    = sorted(rfm["Segment"].unique())
    chosen_segs = st.multiselect("Filter segments", segments, default=segments)

    filtered = rfm[rfm["Segment"].isin(chosen_segs)]
    st.markdown(f"**{len(filtered):,} customers** in selected segments.")

    # Stats
    st.subheader("Segment Statistics")
    summary = segment_summary(filtered)
    st.dataframe(
        summary.style.format({
            "Avg_Recency":   "{:.1f}",
            "Avg_Frequency": "{:.2f}",
            "Avg_Monetary":  "£{:.2f}",
            "Total_Revenue": "£{:,.0f}",
            "Revenue_%":     "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # UMAP filtered
    st.subheader("Cluster Projection (filtered)")
    fig_f = plot_umap(filtered, seg["embedding"][filtered.index])
    st.plotly_chart(fig_f, use_container_width=True)

    # DBSCAN comparison
    st.subheader("DBSCAN Cluster Distribution")
    db_counts = filtered["DBSCAN_Cluster"].value_counts().reset_index()
    db_counts.columns = ["Cluster", "Count"]
    noise = (filtered["DBSCAN_Cluster"] == -1).sum()

    fig_db = go.Figure(go.Bar(
        x=db_counts["Cluster"].astype(str),
        y=db_counts["Count"],
        marker_color="#845EC2",
    ))
    fig_db.update_layout(
        title=f"DBSCAN Clusters (noise points = {noise})",
        xaxis_title="Cluster ID (-1 = noise)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.8)",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig_db, use_container_width=True)

    # Sample customers table
    st.subheader("Sample Customers")
    n_sample = st.slider("Number of rows to show", 10, 200, 50)
    st.dataframe(
        filtered[["CustomerID", "Segment", "Recency", "Frequency", "Monetary",
                  "Cluster", "DBSCAN_Cluster"]]
        .head(n_sample)
        .style.format({"Monetary": "£{:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRODUCT SIMILARITY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🛍️ Product Similarity":
    st.markdown("# 🛍️ Product Similarity Search")
    st.markdown("TF-IDF cosine similarity on product descriptions.")
    st.markdown("---")

    items   = rec["items"]
    tmatrix = rec["tfidf_matrix"]

    # Search by description keyword
    keyword = st.text_input("Search products by keyword", placeholder="e.g. candle, mug, bag …")
    if keyword:
        mask  = items["Description"].str.contains(keyword, case=False, na=False)
        found = items[mask].head(20)
        if found.empty:
            st.warning("No products found.")
        else:
            chosen = st.selectbox(
                "Select a product to find similar items",
                options=found["StockCode"].tolist(),
                format_func=lambda x: f"{x} — {items[items.StockCode==x].iloc[0].Description}",
            )
            top_k = st.slider("Top-K similar products", 5, 20, 10)

            with st.spinner("Computing similarity …"):
                sims = cb_recommend_by_product(chosen, items, tmatrix, top_n=top_k)

            if sims.empty:
                st.info("Not enough data for this product.")
            else:
                st.subheader(f"Top {top_k} similar to: {chosen}")
                for i, r in sims.iterrows():
                    score = float(r["CB_Score"])
                    bar_w = int(score * 200)
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<span class="rec-rank">#{i+1}</span>'
                        f'<span class="rec-name"><b>{r.StockCode}</b> — {str(r.Description)[:70]}</span>'
                        f'<span class="rec-score">{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.info("Type a keyword above to search for products.")

        # Show a random sample of available products
        st.subheader("Browse Products")
        sample = items.sample(min(20, len(items)), random_state=1)
        st.dataframe(sample[["StockCode", "Description"]], use_container_width=True, hide_index=True)