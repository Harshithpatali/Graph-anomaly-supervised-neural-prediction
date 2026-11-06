"""Graph-Based Customer Intelligence dashboard; reads precomputed label-free reports."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "customer_intelligence"
st.set_page_config(page_title="Customer Intelligence", page_icon="◈", layout="wide")

st.markdown("""
<style>
.stApp{background:#0b1220;color:#edf3fb}
[data-testid="stHeader"]{background:rgba(11,18,32,.9)}
[data-testid="stMain"] .block-container{max-width:1480px;padding-top:1.5rem}
h1,h2,h3{color:#f4f7fc!important}
p,li,label{color:#d5e0ed}
section[data-testid="stSidebar"]{background:#0e192a}
[data-testid="stMetric"]{background:linear-gradient(145deg,#142238,#101b2d);border:1px solid #263750;border-radius:13px;padding:1rem}
[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *{color:#f4f8fd!important}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{border:1px solid #263750;border-radius:12px;overflow:hidden}
[data-baseweb="popover"],[role="listbox"],[role="option"]{background:#101d30!important;color:#f4f7fc!important}
.stButton>button,.stDownloadButton>button{background:#173b54;color:#fff!important;border:1px solid #438eaf}
.intel-kicker{color:#54c7ec;text-transform:uppercase;letter-spacing:.16em;font-size:.75rem;font-weight:700}

/* Stable system typography and accessible BaseWeb portal menus. */
.stApp,[data-testid="stMain"],[data-testid="stMarkdownContainer"]{
  font-family="Segoe UI, Arial, sans-serif";
  font-family: "Segoe UI", Arial, sans-serif !important;
}
body > div [data-baseweb="popover"],[data-baseweb="popover"],[data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"],[role="listbox"]{
  background: #111e31 !important;
  background: #101d30 !important;
  color: #f4f7fc !important;
  z-index: 999999 !important;
}
[data-baseweb="popover"] [role="option"],[role="option"]{
  background: #101d30 !important;color:#f4f7fc !important;
}
[data-baseweb="popover"] [role="option"]:hover,[role="option"][data-highlighted="true"]{
  background: #24415d !important;
  background: #294967 !important;
}
</style>
""", unsafe_allow_html=True)

def read_zip(upload):
    out = {}
    with zipfile.ZipFile(upload) as zf:
        for name in ("customer_intelligence.parquet","customer_profiles.parquet","top_transaction_anomalies.parquet"):
            matches=[p for p in zf.namelist() if p.replace("\\","/").endswith(name)]
            if matches:
                with zf.open(matches[0]) as f: out[name]=pd.read_parquet(io.BytesIO(f.read()))
        for name in ("segment_summary.csv","segmentation_metrics.csv","graph_entity_metrics.csv","graph_communities.csv"):
            matches=[p for p in zf.namelist() if p.replace("\\","/").endswith(name)]
            if matches:
                with zf.open(matches[0]) as f: out[name]=pd.read_csv(f)
        matches=[p for p in zf.namelist() if p.replace("\\","/").endswith("run_metadata.json")]
        if matches:
            with zf.open(matches[0]) as f: out["run_metadata.json"]=json.loads(f.read().decode())
    return out

@st.cache_data(show_spinner=False)
def load_local():
    out={}
    for name in ("customer_intelligence.parquet","customer_profiles.parquet","top_transaction_anomalies.parquet"):
        p=REPORT_DIR/name
        if p.exists(): out[name]=pd.read_parquet(p)
    for name in ("segment_summary.csv","segmentation_metrics.csv","graph_entity_metrics.csv","graph_communities.csv"):
        p=REPORT_DIR/name
        if p.exists(): out[name]=pd.read_csv(p)
    p=REPORT_DIR/"run_metadata.json"
    if p.exists(): out["run_metadata.json"]=json.loads(p.read_text())
    return out

def plot(fig, height=350):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#111c2e",plot_bgcolor="#111c2e",
                      font=dict(color="#edf3fb"),margin=dict(l=20,r=20,t=45,b=25),height=height)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

def download(frame,name,label):
    st.download_button(label,frame.to_csv(index=False).encode(),name,"text/csv")

st.sidebar.markdown('<div class="intel-kicker">Customer intelligence platform</div>',unsafe_allow_html=True)
st.sidebar.title("◈ Graph & Behavior")
st.sidebar.caption("Discover customers · Understand behavior · Surface anomalies")
data=load_local()
if "customer_intelligence.parquet" not in data:
    up=st.sidebar.file_uploader("Upload GitHub Actions research ZIP",type=["zip"],
       help="Download the customer-intelligence-research artifact from the GitHub Actions run.")
    if up is not None:
        try: data=read_zip(up); st.sidebar.success("Research bundle loaded for this session.")
        except Exception as exc: st.sidebar.error(f"Unable to read ZIP: {exc}")
pages=["Overview","Customer Explorer","Customer Segments","Graph Intelligence","Anomaly Lab","Methodology"]
page=st.sidebar.radio("Workspace",pages)
st.sidebar.caption("Descriptive signals only. No fraud probability is produced.")
st.markdown('<div class="intel-kicker">Behavioral analytics · Network science · Unsupervised discovery</div>',unsafe_allow_html=True)
st.title("Graph-Based Customer Intelligence")
st.markdown("### Discover customers. Understand behavior. Identify risk. Find hidden anomalies.")
st.caption("A label-free research workspace for customer behavior, segmentation, entity relationships and anomaly prioritization.")

customers=data.get("customer_intelligence.parquet")
if customers is None or customers.empty:
    st.info("The dashboard is ready, but the precomputed research bundle is not included in this deployment. Run the Customer Intelligence Research workflow, download its customer-intelligence-research ZIP artifact, then upload it in the sidebar.")
    st.markdown("Expected files: customer_intelligence.parquet, customer_profiles.parquet, top_transaction_anomalies.parquet, segment_summary.csv, segmentation_metrics.csv, graph_entity_metrics.csv, graph_communities.csv and run_metadata.json.")
    st.stop()
customers=customers.copy()
if "user_id" in customers: customers["user_id"]=customers["user_id"].astype(str)
meta=data.get("run_metadata.json",{})
segments=data.get("segment_summary.csv",pd.DataFrame())
metrics=data.get("segmentation_metrics.csv",pd.DataFrame())
entities=data.get("graph_entity_metrics.csv",pd.DataFrame())
communities=data.get("graph_communities.csv",pd.DataFrame())
transactions=data.get("top_transaction_anomalies.parquet")

if page=="Overview":
    st.subheader("Intelligence overview")
    cols=st.columns(5)
    vals=[("Customers",f"{len(customers):,}"),("Transactions",f"{meta.get('transactions',0):,}"),
          ("Graph nodes",f"{meta.get('graph_nodes',0):,}"),("Graph edges",f"{meta.get('graph_edges',0):,}"),
          ("Segments",str(customers.segment_id.nunique()) if "segment_id" in customers else "—")]
    for col,(label,value) in zip(cols,vals): col.metric(label,value)
    left,right=st.columns(2)
    with left:
        st.markdown("#### Behavioral risk distribution")
        if "behavioral_risk_index" in customers:
            plot(px.histogram(customers,x="behavioral_risk_index",nbins=30,color_discrete_sequence=["#54c7ec"]))
        else: st.info("Risk index unavailable.")
    with right:
        st.markdown("#### Unsupervised anomaly bands")
        if "anomaly_band" in customers:
            counts=customers.anomaly_band.value_counts().rename_axis("band").reset_index(name="customers")
            plot(px.pie(counts,names="band",values="customers",hole=.58))
        else: st.info("Anomaly bands unavailable.")
    st.markdown("#### Segment profile")
    if not segments.empty: st.dataframe(segments,use_container_width=True,hide_index=True)
    st.warning("Behavioral risk is a cohort-relative heuristic. Anomaly scores are unsupervised rankings. Neither is a fraud probability or proof of misconduct.")

elif page=="Customer Explorer":
    st.subheader("Customer Explorer")
    bands=sorted(customers.behavioral_risk_band.dropna().astype(str).unique()) if "behavioral_risk_band" in customers else []
    c1,c2,c3=st.columns(3)
    selected_bands=c1.multiselect("Behavioral risk band",bands,default=bands)
    ids=sorted(customers.segment_id.dropna().unique().tolist()) if "segment_id" in customers else []
    selected_ids=c2.multiselect("Segment",ids,default=ids)
    search=c3.text_input("Search customer ID")
    view=customers.copy()
    if selected_bands and "behavioral_risk_band" in view: view=view[view.behavioral_risk_band.astype(str).isin(selected_bands)]
    if selected_ids and "segment_id" in view: view=view[view.segment_id.isin(selected_ids)]
    if search and "user_id" in view: view=view[view.user_id.str.contains(search,case=False,na=False)]
    sort="anomaly_score" if "anomaly_score" in view else "behavioral_risk_index"
    if sort in view: view=view.sort_values(sort,ascending=False)
    show=[c for c in ["user_id","segment_id","behavioral_risk_index","behavioral_risk_band","anomaly_score","anomaly_band","transaction_count","total_spend","mean_transaction_value","days_since_last_purchase","risk_explanations"] if c in view]
    st.caption(f"{len(view):,} matching customers")
    st.dataframe(view[show].head(1000),use_container_width=True,height=440,hide_index=True)
    download(view[show],"filtered_customer_intelligence.csv","Download filtered customers")
    if not view.empty and "user_id" in view:
        selected=st.selectbox("Inspect customer",view.user_id.head(500).tolist())
        row=view[view.user_id==selected].iloc[0]
        a,b,c,d=st.columns(4)
        a.metric("Transactions",f"{row.get('transaction_count',0):,.0f}")
        b.metric("Lifetime spend",f"{row.get('total_spend',0):,.2f}")
        c.metric("Behavioral risk",f"{row.get('behavioral_risk_index',0):.1f}/100")
        d.metric("Anomaly score",f"{row.get('anomaly_score',0):.3f}")
        st.write("**Why surfaced:**",row.get("risk_explanations","No explanation available."))

elif page=="Customer Segments":
    st.subheader("Customer Segmentation")
    if "segment_id" not in customers: st.warning("Segment labels unavailable."); st.stop()
    counts=customers.segment_id.value_counts().sort_index().rename_axis("segment_id").reset_index(name="customers")
    left,right=st.columns(2)
    with left: plot(px.bar(counts,x="segment_id",y="customers",color="segment_id"))
    with right:
        if not segments.empty and {"median_transactions","median_total_spend","customers"}.issubset(segments.columns):
            plot(px.scatter(segments,x="median_transactions",y="median_total_spend",size="customers",color="segment_id"))
        else: st.info("Segment comparison unavailable.")
    if not segments.empty: st.dataframe(segments,use_container_width=True,hide_index=True)
    if not metrics.empty:
        with st.expander("Clustering diagnostics"):
            st.dataframe(metrics,use_container_width=True,hide_index=True)
            if {"k","silhouette_sample"}.issubset(metrics.columns): plot(px.line(metrics,x="k",y="silhouette_sample",markers=True),300)
    st.caption("Segment IDs are descriptive K-means groupings, not ordinal customer-value tiers.")

elif page=="Graph Intelligence":
    st.subheader("Shared-entity network intelligence")
    st.caption("User-device-IP graph, built from at most the first 50,000 input transactions.")
    if entities.empty: st.warning("Graph metrics are not in this bundle."); st.stop()
    a,b,c=st.columns(3)
    a.metric("Entities",f"{len(entities):,}")
    b.metric("Communities",f"{len(communities):,}" if not communities.empty else "—")
    c.metric("Maximum degree",f"{entities.degree.max():,.0f}" if "degree" in entities else "—")
    if {"node_type","degree"}.issubset(entities.columns):
        left,right=st.columns(2)
        with left:
            types=entities.node_type.value_counts().rename_axis("type").reset_index(name="nodes")
            plot(px.bar(types,x="type",y="nodes",color="type"))
        with right:
            top=entities.sort_values("degree",ascending=False).head(20).copy()
            top["entity_id"]=top.entity_id.astype(str).str.slice(0,35)
            plot(px.bar(top.sort_values("degree"),x="degree",y="entity_id",color="node_type",orientation="h",
                        hover_data=[c for c in ["component_size","betweenness"] if c in top]),520)
    st.markdown("#### Largest communities")
    if not communities.empty: st.dataframe(communities.head(30),use_container_width=True,hide_index=True)
    st.markdown("#### Most connected entities")
    st.dataframe(entities.head(100),use_container_width=True,hide_index=True)
    download(entities,"graph_entity_metrics.csv","Download graph metrics")
    st.caption("Only entity metrics and community summaries are exported; raw graph edges are not included, so no artificial network layout is drawn.")

elif page=="Anomaly Lab":
    st.subheader("Unsupervised Anomaly Lab")
    st.caption("Customer anomaly score = 60% Isolation Forest percentile + 40% LOF percentile. Scores are relative outlier signals, not fraud classifications.")
    if "anomaly_score" in customers:
        n=st.slider("Customers to inspect",10,min(500,len(customers)),min(100,len(customers)),10)
        ranked=customers.sort_values("anomaly_score",ascending=False).head(n)
        show=[c for c in ["user_id","anomaly_score","anomaly_rank","anomaly_band","behavioral_risk_index","transaction_count","total_spend","risk_explanations"] if c in ranked]
        st.dataframe(ranked[show],use_container_width=True,height=420,hide_index=True)
        download(ranked[show],"top_customer_anomalies.csv","Download customer anomaly ranking")
    if transactions is not None and not transactions.empty:
        st.markdown("#### Transaction anomaly candidates")
        st.dataframe(transactions.head(500),use_container_width=True,height=380,hide_index=True)
        download(transactions,"transaction_anomaly_candidates.csv","Download transaction anomalies")
    else: st.info("Transaction anomaly report unavailable.")

elif page=="Methodology":
    st.subheader("Methodology, provenance & limitations")
    st.markdown("""
**Primary objective:** discover behavioral patterns and network structure without a supervised fraud classifier.

**Data handling**
- Historical class labels are dropped before customer feature generation and are not used in clustering, risk ranking, graph construction or anomaly scoring.
- Customer profiles summarize purchase frequency, value, timing, recency and device/IP diversity.
- Segmentation uses robust scaling and K-means; candidate cluster counts use sample silhouette and Davies–Bouldin diagnostics.
- Customer anomaly ranking combines Isolation Forest and Local Outlier Factor percentile scores.
- Graph intelligence links users to device and IP entities and summarizes degree, connected components, betweenness and Louvain communities.

**Interpretation**
- Behavioral risk index is a cohort-relative heuristic (0–100), not a probability.
- Anomaly bands are ranking buckets, not confirmed labels.
- Segment IDs are arbitrary cluster identifiers, not ordinal scores.
""")
    if meta: st.json(meta)
    for item in meta.get("limitations",["Graph limited to first 50,000 input transactions.","Entity frequency features are full-snapshot descriptive counts, not point-in-time predictions."]): st.markdown(f"- {item}")
    st.warning("Do not use these outputs alone to block, accuse or penalize a customer. They are exploratory decision-support signals.")
