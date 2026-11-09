"""Graph-Based Customer Intelligence dashboard; reads precomputed label-free reports."""
from __future__ import annotations
import io, json, zipfile, urllib.request
from pathlib import Path
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "customer_intelligence"
st.set_page_config(page_title="Customer Intelligence | Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0b1220!important;color:#f3f7fc!important}
[data-testid="stHeader"]{background:rgba(11,18,32,.96)!important}
[data-testid="stMain"] .block-container{max-width:1560px;padding:1.6rem 2rem 3rem}
.stApp,.stApp p,.stApp li,.stApp label,.stApp span,.stApp div{color:inherit}
h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:#f3f7fc!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#aebed1!important}
section[data-testid="stSidebar"]{background:#0e192a!important}
section[data-testid="stSidebar"] *{color:#e8f0fa!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#162840,#101d30)!important;border:1px solid #304762;border-radius:14px;padding:1rem}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#f7faff!important}
[data-testid="stMetricDelta"],[data-testid="stMetricDelta"] *{color:#a9c8dd!important}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#111e31;border:1px solid #30445e;border-radius:12px;overflow:hidden}
[data-baseweb="select"] *,[data-baseweb="input"] *,[data-baseweb="textarea"] *,input,textarea{color:#f4f8ff!important}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{background:#14243a!important;border-color:#405773!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:#101d30!important;color:#f4f7fc!important}
[data-baseweb="popover"] *,[role="listbox"] *,[role="option"] *{color:#f4f7fc!important}
[data-baseweb="popover"] [role="option"]:hover,[role="option"][aria-selected="true"]{background:#294967!important}
.stButton>button,.stDownloadButton>button{background:#17405d!important;color:#fff!important;border:1px solid #5798bb!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:#205779!important;border-color:#78c8e9!important}
[data-testid="stDataFrame"] *{color:#eaf2fb!important}
hr{border-color:#30445e!important}
.stApp [data-testid='stWidgetLabel'],.stApp [data-testid='stWidgetLabel'] *,.stApp [data-testid='stMarkdownContainer'] *,.stApp [data-testid='stAlert'] *,.stApp [data-testid='stExpander'] *{color:#edf4fc!important}
.stApp input::placeholder,.stApp textarea::placeholder{color:#a9bbcf!important;opacity:1!important}
.stApp [data-testid='stFileUploader'] section{background:#14243a!important;border:1px dashed #52708f!important}
.stApp [data-testid='stFileUploader'] span,.stApp [data-testid='stFileUploader'] small{color:#dce8f5!important}
.stApp button{color:#fff!important}
.intel-kicker{color:#65d5f4!important;text-transform:uppercase;letter-spacing:.16em;font-size:.75rem;font-weight:800}
.dashboard-note{background:linear-gradient(110deg,#14243a,#10283a);border:1px solid #34536d;border-radius:12px;padding:14px 18px;color:#e4f1fc!important}
.product-subtitle{color:#b7c9dc!important;font-size:1rem}
.section-eyebrow{color:#72d7f2!important;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800}
[data-testid="stDataFrame"] th{background:#1a3049!important;color:#fff!important}
.stApp [data-testid="stAlert"]{border-radius:10px}

/* Stable system typography and accessible BaseWeb portal menus. */
.stApp,[data-testid="stMain"],[data-testid="stMarkdownContainer"]{
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
        for name in ("customer_intelligence.parquet","customer_profiles.parquet","top_transaction_anomalies.parquet","customer_purchase_timeline.parquet"):
            matches=[p for p in zf.namelist() if p.replace("\\","/").endswith(name)]
            if matches:
                with zf.open(matches[0]) as f: out[name]=pd.read_parquet(io.BytesIO(f.read()))
        for name in ("segment_summary.csv","segmentation_metrics.csv","graph_entity_metrics.csv","graph_communities.csv","graph_edges_sample.csv","graph_edges.csv"):
            matches=[p for p in zf.namelist() if p.replace("\\","/").endswith(name)]
            if matches:
                with zf.open(matches[0]) as f: out[name]=pd.read_csv(f)
        matches=[p for p in zf.namelist() if p.replace("\\","/").endswith("run_metadata.json")]
        if matches:
            with zf.open(matches[0]) as f: out["run_metadata.json"]=json.loads(f.read().decode())
    return out

@st.cache_data(show_spinner="Fetching the latest public research bundle…", ttl=3600)
def load_github_bundle():
    """Download the portfolio research bundle from this public GitHub repository."""
    url = "https://raw.githubusercontent.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction/main/public-data/customer-intelligence-research.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "customer-intelligence-streamlit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return read_zip(io.BytesIO(payload))

@st.cache_data(show_spinner=False)
def load_local():
    out={}
    for name in ("customer_intelligence.parquet","customer_profiles.parquet","top_transaction_anomalies.parquet"):
        p=REPORT_DIR/name
        if p.exists(): out[name]=pd.read_parquet(p)
    for name in ("segment_summary.csv","segmentation_metrics.csv","graph_entity_metrics.csv","graph_communities.csv","graph_edges_sample.csv"):
        p=REPORT_DIR/name
        if p.exists(): out[name]=pd.read_csv(p)
    p=REPORT_DIR/"run_metadata.json"
    if p.exists(): out["run_metadata.json"]=json.loads(p.read_text())
    return out

def plot(fig, height=350):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#111c2e",plot_bgcolor="#111c2e",
                      font=dict(family="Arial, sans-serif",size=13,color="#f1f6fd"),
                      title_font=dict(size=17,color="#ffffff"),margin=dict(l=28,r=24,t=55,b=35),height=height,
                      legend=dict(font=dict(color="#e8f1fb")),
                      xaxis=dict(title_font=dict(color="#dce8f5"),tickfont=dict(color="#dce8f5"),gridcolor="#263a52"),
                      yaxis=dict(title_font=dict(color="#dce8f5"),tickfont=dict(color="#dce8f5"),gridcolor="#263a52"))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

def download(frame,name,label):
    st.download_button(label,frame.to_csv(index=False).encode(),name,"text/csv")


def segment_narrative(group: pd.Series, overall: pd.Series) -> list[str]:
    """Describe segment differences using explicit median comparisons."""
    statements=[]
    specs=[("median_transactions","transaction frequency"),("median_total_spend","lifetime spend"),
           ("median_avg_ticket","average transaction value"),("median_recency_days","purchase recency"),
           ("median_devices","device diversity")]
    for key,label in specs:
        if key not in group.index or key not in overall.index: continue
        value=pd.to_numeric(pd.Series([group[key]]),errors="coerce").iloc[0]
        base=pd.to_numeric(pd.Series([overall[key]]),errors="coerce").iloc[0]
        if pd.isna(value) or pd.isna(base): continue
        if base == 0: continue
        ratio=value/base
        if ratio >= 1.25: statements.append(f"Higher {label} than cohort median ({ratio:.1f}×).")
        elif ratio <= .75: statements.append(f"Lower {label} than cohort median ({ratio:.1f}×).")
    return statements or ["No large median differences on the available summary measures."]

def anomaly_evidence(row: pd.Series, cohort: pd.DataFrame) -> pd.DataFrame:
    """Rank feature deviations by absolute robust z-score against the current cohort."""
    excluded={"user_id","segment_id","behavioral_risk_index","anomaly_score","anomaly_rank",
              "isolation_forest_score","lof_score","class"}
    cols=[c for c in cohort.select_dtypes(include="number").columns if c not in excluded]
    rows=[]
    for col in cols:
        values=pd.to_numeric(cohort[col],errors="coerce").replace([float("inf"),-float("inf")],pd.NA).dropna()
        if len(values)<10: continue
        med=values.median(); q1=values.quantile(.25); q3=values.quantile(.75); spread=q3-q1
        value=pd.to_numeric(pd.Series([row.get(col)]),errors="coerce").iloc[0]
        if pd.isna(value) or spread<=0: continue
        rz=(value-med)/(spread/1.349)
        rows.append({"indicator":col,"observed_value":float(value),"cohort_median":float(med),
                     "robust_deviation":float(rz),"absolute_deviation":abs(float(rz))})
    return pd.DataFrame(rows).sort_values("absolute_deviation",ascending=False).head(8) if rows else pd.DataFrame()

def executive_pdf(customers: pd.DataFrame, segments: pd.DataFrame, meta: dict) -> bytes:
    """Build a concise downloadable research report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=16*mm,bottomMargin=16*mm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote",parent=styles["BodyText"],fontSize=8,leading=11,textColor=colors.HexColor("#455468")))
    story=[Paragraph("Customer Intelligence — Executive Research Report",styles["Title"]),
           Paragraph("Descriptive customer analytics · Unsupervised discovery · Graph intelligence",styles["SmallNote"]),Spacer(1,8*mm)]
    total=len(customers)
    high=int((customers["behavioral_risk_band"]=="High").sum()) if "behavioral_risk_band" in customers else 0
    extreme=int((customers["anomaly_band"]=="Extreme").sum()) if "anomaly_band" in customers else 0
    spend=float(customers["total_spend"].sum()) if "total_spend" in customers else 0
    rows=[["Metric","Value"],["Customers",f"{total:,}"],["Source transactions",f"{int(meta.get('transactions',0)):,}"],
          ["Total customer spend",f"{spend:,.2f}"],["High relative-risk profiles",f"{high:,}"],["Extreme anomaly candidates",f"{extreme:,}"]]
    table=Table(rows,colWidths=[85*mm,75*mm])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5e1")),("PADDING",(0,0),(-1,-1),7)]))
    story += [Paragraph("Executive snapshot",styles["Heading2"]),table,Spacer(1,6*mm)]
    story.append(Paragraph("Segment summary",styles["Heading2"]))
    if not segments.empty:
        cols=[c for c in ["segment_id","customers","median_transactions","median_total_spend","median_avg_ticket","median_recency_days"] if c in segments]
        data=[cols]+segments[cols].head(20).astype(str).values.tolist()
        tab=Table(data,repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                 ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cbd5e1")),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),5)]))
        story.append(tab)
    story += [Spacer(1,5*mm),Paragraph("Methodology and interpretation",styles["Heading2"]),
              Paragraph("Customer segmentation uses robust scaling and K-means. Customer anomaly ranking combines Isolation Forest percentile (60%) and Local Outlier Factor percentile (40%). Behavioral risk is a cohort-relative heuristic index (0–100), not a calibrated probability. Anomaly scores are relative unsupervised rankings, not confirmed fraud labels. Graph analysis uses at most the first 50,000 source transactions. Customer purchase histories retain at most 50 transactions per customer.",styles["BodyText"]),
              Spacer(1,3*mm),Paragraph("Decision-use limitation: these exploratory signals should not be used alone to block, accuse or penalize a customer. Validate findings with appropriate human review and additional evidence.",styles["SmallNote"])]
    doc.build(story)
    return buffer.getvalue()

\ndef graph_network_figure(edge_frame: pd.DataFrame, entity_frame: pd.DataFrame):
    """Render a small deterministic sample of the user-device-IP graph."""
    if edge_frame.empty or not {'source', 'target'}.issubset(edge_frame.columns): return None
    graph = nx.from_pandas_edgelist(edge_frame.head(350), source='source', target='target')
    if graph.number_of_nodes() == 0: return None
    positions = nx.spring_layout(graph, seed=42, iterations=35)
    ex, ey = [], []
    for u, v in graph.edges():
        ex.extend([positions[u][0], positions[v][0], None]); ey.extend([positions[u][1], positions[v][1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=.8,color='rgba(148,170,195,.4)'), hoverinfo='none'))
    kinds = {}
    if not entity_frame.empty and {'entity_id','node_type'}.issubset(entity_frame.columns): kinds = entity_frame.drop_duplicates('entity_id').set_index('entity_id').node_type.to_dict()
    nodes = list(graph.nodes()); types = [kinds.get(n,str(n).split('::',1)[0]) for n in nodes]
    palette = {'user':'#38bdf8','device':'#a78bfa','ip':'#34d399'}
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes], mode='markers',
        text=[f'{types[i]} · {str(n)[:45]} · degree {graph.degree(n)}' for i,n in enumerate(nodes)], hoverinfo='text',
        marker=dict(size=[9+min(graph.degree(n),7) for n in nodes], color=[palette.get(t,'#f59e0b') for t in types], line=dict(width=1,color='#e6f0fa'))))
    fig.update_layout(title='Sampled user-device-IP relationship network',showlegend=False,height=600,
        xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=10,r=10,t=50,b=10),paper_bgcolor='#111c2e',plot_bgcolor='#111c2e')
    return fig

st.sidebar.markdown('<div class="intel-kicker">Customer intelligence platform</div>',unsafe_allow_html=True)
st.sidebar.title("◈ Intelligence Hub")
st.sidebar.caption("Discover customers · Understand behavior · Surface anomalies")
data=load_local()
if "customer_intelligence.parquet" not in data:
    try:
        data=load_github_bundle()
        st.sidebar.success("Latest public GitHub research bundle loaded.")
    except Exception as exc:
        st.sidebar.warning(f"Automatic GitHub data load unavailable: {exc}")
if "customer_intelligence.parquet" not in data:
    up=st.sidebar.file_uploader("Upload research ZIP (fallback)",type=["zip"],
       help="Use this only if the public GitHub bundle is unavailable.")
    if up is not None:
        try: data=read_zip(up); st.sidebar.success("Research bundle loaded for this session.")
        except Exception as exc: st.sidebar.error(f"Unable to read ZIP: {exc}")
pages=["Overview","Customer 360","Customer Explorer","Customer Segments","Graph Intelligence","Anomaly Lab","Executive Report","Methodology"]
page=st.sidebar.radio("Workspace",pages)
st.sidebar.caption("Descriptive signals only. No fraud probability is produced.")
st.markdown('<div class="intel-kicker">Behavioral analytics · Network science · Unsupervised discovery</div>',unsafe_allow_html=True)
st.title("Customer Intelligence")
st.markdown('<div class="product-subtitle">Understand customer behavior. Discover meaningful segments. Prioritize unusual activity. Explore hidden relationships.</div>',unsafe_allow_html=True)
st.caption("Decision-support analytics built from precomputed, label-free customer behavior features.")

customers=data.get("customer_intelligence.parquet")
if customers is None or customers.empty:
    st.info("The dashboard could not find its precomputed research bundle. Check that the Customer Intelligence Research workflow completed successfully and published public-data/customer-intelligence-research.zip. You can also upload a research ZIP using the sidebar fallback.")
    st.markdown("Expected files: customer_intelligence.parquet, customer_profiles.parquet, top_transaction_anomalies.parquet, segment_summary.csv, segmentation_metrics.csv, graph_entity_metrics.csv, graph_communities.csv and run_metadata.json.")
    st.stop()
customers=customers.copy()
if "user_id" in customers: customers["user_id"]=customers["user_id"].astype(str)
meta=data.get("run_metadata.json",{})
segments=data.get("segment_summary.csv",pd.DataFrame())
metrics=data.get("segmentation_metrics.csv",pd.DataFrame())
entities=data.get("graph_entity_metrics.csv",pd.DataFrame())
communities=data.get("graph_communities.csv",pd.DataFrame())
transactions=data.get("top_transaction_anomalies.parquet")\ntimeline=data.get("customer_purchase_timeline.parquet",pd.DataFrame())\ngraph_edges=data.get("graph_edges.csv",data.get("graph_edges_sample.csv",pd.DataFrame()))

if page=="Overview":
    st.subheader("Executive overview")
    st.markdown('<div class="dashboard-note">A portfolio-grade view of customer behavior, cohort composition, anomaly signals and shared-entity network structure. All metrics are descriptive and label-free.</div>',unsafe_allow_html=True)
    risk_col="behavioral_risk_index" if "behavioral_risk_index" in customers else None
    spend_col="total_spend" if "total_spend" in customers else None
    txn_col="transaction_count" if "transaction_count" in customers else None
    cols=st.columns(5)
    high_risk=int((customers["behavioral_risk_band"]=="High").sum()) if "behavioral_risk_band" in customers else 0
    extreme=int((customers["anomaly_band"]=="Extreme").sum()) if "anomaly_band" in customers else 0
    vals=[("Customers",f"{len(customers):,}"),("Transactions",f"{meta.get('transactions',0):,}"),
          ("Customer spend",f"{customers[spend_col].sum():,.0f}" if spend_col else "—"),
          ("High relative-risk",f"{high_risk:,}"),("Extreme anomalies",f"{extreme:,}")]
    for col,(label,value) in zip(cols,vals): col.metric(label,value)
    st.markdown("#### Customer behavior")
    left,right=st.columns(2)
    with left:
        st.markdown("**Transaction value distribution across customers**")
        if spend_col: plot(px.histogram(customers,x=spend_col,nbins=35,color_discrete_sequence=["#38bdf8"]))
        else: st.info("Spend data unavailable.")
    with right:
        st.markdown("**Behavioral risk distribution**")
        if risk_col: plot(px.histogram(customers,x=risk_col,nbins=30,color_discrete_sequence=["#a78bfa"]))
        else: st.info("Risk index unavailable.")
    left,right=st.columns(2)
    with left:
        st.markdown("**Customer frequency vs lifetime spend**")
        if txn_col and spend_col:
            sample=customers.sample(min(5000,len(customers)),random_state=42)
            fig=px.scatter(sample,x=txn_col,y=spend_col,color="segment_id" if "segment_id" in sample else None,
                           size=risk_col if risk_col else None,hover_data=[c for c in ["user_id","behavioral_risk_band","anomaly_band"] if c in sample],
                           opacity=.68)
            plot(fig,420)
        else: st.info("Transaction count or spend unavailable.")
    with right:
        st.markdown("**Anomaly cohort composition**")
        if "anomaly_band" in customers:
            counts=customers.anomaly_band.fillna("Unknown").value_counts().rename_axis("band").reset_index(name="customers")
            plot(px.pie(counts,names="band",values="customers",hole=.58,color_discrete_sequence=["#22c55e","#f59e0b","#ef4444","#64748b"]))
        else: st.info("Anomaly bands unavailable.")
    st.markdown("#### Segment performance profile")
    if "segment_id" in customers:
        numeric=[c for c in ["transaction_count","total_spend","mean_transaction_value","behavioral_risk_index","anomaly_score"] if c in customers]
        if numeric:
            segment_profile=customers.groupby("segment_id",dropna=False)[numeric].median().reset_index()
            st.dataframe(segment_profile,use_container_width=True,hide_index=True)
            if "total_spend" in numeric:
                plot(px.bar(segment_profile,x="segment_id",y="total_spend",color="segment_id",title="Median customer spend by segment"),360)
    if not metrics.empty and {"k","silhouette_sample"}.issubset(metrics.columns):
        st.markdown("#### Segmentation model diagnostics")
        plot(px.line(metrics,x="k",y="silhouette_sample",markers=True,title="Silhouette score across candidate cluster counts"),320)
    st.warning("Behavioral risk is a cohort-relative heuristic. Anomaly scores are unsupervised rankings. Neither is a fraud probability or proof of misconduct.")

elif page=="Customer 360":
    st.subheader("Customer 360")
    st.caption("A unified customer profile combining purchase history, behavioral indicators, cohort context and network connections.")
    if "user_id" not in customers: st.warning("Customer identifiers are unavailable."); st.stop()
    ids=customers.user_id.astype(str).tolist()
    selected=st.selectbox("Select customer",ids,key="customer360_id")
    row=customers[customers.user_id.astype(str)==selected].iloc[0]
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Transactions",f"{row.get('transaction_count',0):,.0f}")
    m2.metric("Lifetime spend",f"{row.get('total_spend',0):,.2f}")
    m3.metric("Relative-risk index",f"{row.get('behavioral_risk_index',0):.1f}/100")
    m4.metric("Anomaly rank",f"{int(row.get('anomaly_rank',0)):,}" if pd.notna(row.get('anomaly_rank')) else "—")
    left,right=st.columns([1.1,1])
    with left:
        st.markdown("#### Behavioral indicators")
        indicators=[c for c in ["transaction_count","active_days","transactions_per_active_day","mean_transaction_value","transaction_value_cv","spend_change_late_vs_early","distinct_device_id","distinct_ip_address","days_since_last_purchase"] if c in row.index]
        st.dataframe(pd.DataFrame({"Indicator":indicators,"Observed value":[row[c] for c in indicators]}),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Customer classification")
        st.write("**Segment:**",row.get("segment_id","Unavailable"))
        st.write("**Relative-risk band:**",row.get("behavioral_risk_band","Unavailable"))
        st.write("**Anomaly band:**",row.get("anomaly_band","Unavailable"))
        st.write("**Behavioral explanation:**",row.get("risk_explanations","Unavailable"))
    st.markdown("#### Purchase timeline")
    if not timeline.empty and "user_id" in timeline:
        history=timeline[timeline.user_id.astype(str)==selected].copy()
        if not history.empty:
            history["purchase_time"]=pd.to_datetime(history["purchase_time"],errors="coerce",utc=True)
            history=history.sort_values("purchase_time")
            if "purchase_value" in history: plot(px.line(history,x="purchase_time",y="purchase_value",markers=True,title="Purchase value over time"),340)
            st.dataframe(history,use_container_width=True,hide_index=True)
            download(history,"customer_"+selected+"_purchase_history.csv","Download this customer's purchase history")
        else: st.info("No retained purchase timeline was exported for this customer.")
    else: st.info("Purchase timeline is not present in this research bundle. Run the latest research workflow.")
    st.markdown("#### Shared graph entities")
    if not graph_edges.empty:
        node="user::"+selected
        links=graph_edges[(graph_edges.source==node)|(graph_edges.target==node)]
        if not links.empty:
            links=links.copy()
            links["shared_entity"]=links.apply(lambda x:x.target if x.source==node else x.source,axis=1)
            st.dataframe(links[["shared_entity"]],use_container_width=True,hide_index=True)
        else: st.info("No exported graph links for this customer in the bounded graph snapshot.")
    else: st.info("Graph edge investigation data is not in this bundle.")

elif page=="Customer Explorer":
    st.subheader("Customer Explorer")
    st.caption("Filter the cohort, rank profiles for review and inspect the behavioral evidence behind an individual record.")
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
    st.subheader("Customer Segments")
    st.caption("Compare discovered behavioral groups by size, purchase activity, spend and recency. Segment IDs are labels, not rankings.")
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
    st.markdown("#### Segment intelligence")
    overall=pd.Series({"median_transactions":customers.transaction_count.median() if "transaction_count" in customers else float("nan"),
                       "median_total_spend":customers.total_spend.median() if "total_spend" in customers else float("nan"),
                       "median_avg_ticket":customers.mean_transaction_value.median() if "mean_transaction_value" in customers else float("nan"),
                       "median_recency_days":customers.days_since_last_purchase.median() if "days_since_last_purchase" in customers else float("nan"),
                       "median_devices":customers.distinct_device_id.median() if "distinct_device_id" in customers else float("nan")})
    if not segments.empty:
        for _,seg in segments.iterrows():
            sid=seg.get("segment_id")
            with st.expander(f"Segment {sid} · {int(seg.get('customers',0)):,} customers"):
                for sentence in segment_narrative(seg,overall): st.markdown("- "+sentence)
                st.caption("Descriptions compare segment medians with the full-cohort medians; they are descriptive, not causal claims.")
    st.caption("Segment IDs are descriptive K-means groupings, not ordinal customer-value tiers.")

elif page=="Graph Intelligence":
    st.subheader("Graph intelligence & network analytics")
    st.caption("User-device-IP graph, built from at most the first 50,000 input transactions. Network view is a bounded sample for readability.")
    edge_data=data.get("graph_edges_sample.csv",pd.DataFrame())
    network_fig=graph_network_figure(edge_data,entities)
    if network_fig is not None: st.plotly_chart(network_fig,use_container_width=True,config={"displaylogo":False})
    else: st.info("Sampled graph edges are not in this bundle yet. Rerun the research workflow to generate the network view.")
    if entities.empty: st.warning("Graph metrics are not in this bundle."); st.stop()
    st.markdown("#### Entity relationship investigation")
    if not graph_edges.empty:
        entity_options=entities.entity_id.astype(str).head(5000).tolist()
        target=st.selectbox("Select a user, device or IP entity",entity_options,key="graph_entity_lookup")
        adjacent=graph_edges[(graph_edges.source==target)|(graph_edges.target==target)].copy()
        adjacent["connected_entity"]=adjacent.apply(lambda x:x.target if x.source==target else x.source,axis=1)
        st.metric("Direct connections in exported graph",f"{len(adjacent):,}")
        if not adjacent.empty:
            st.dataframe(adjacent[["connected_entity"]].drop_duplicates(),use_container_width=True,hide_index=True)
            st.caption("Connections are direct edges in the bounded graph export; absence from this export does not prove no relationship exists.")
    else: st.info("Fuller graph investigation requires the latest graph_edges.csv report.")
    degree_col="degree" if "degree" in entities else None
    a,b,c,d=st.columns(4)
    a.metric("Graph entities",f"{len(entities):,}")
    b.metric("Communities",f"{len(communities):,}" if not communities.empty else "—")
    c.metric("Maximum degree",f"{entities[degree_col].max():,.0f}" if degree_col else "—")
    d.metric("Multi-node components",f"{(entities.component_size>1).sum():,}" if "component_size" in entities else "—")
    st.markdown("#### Network composition and connectivity")
    left,right=st.columns(2)
    with left:
        st.markdown("**Entity type composition**")
        if "node_type" in entities:
            types=entities.node_type.value_counts().rename_axis("entity_type").reset_index(name="entities")
            plot(px.pie(types,names="entity_type",values="entities",hole=.55,
                        color_discrete_sequence=["#38bdf8","#a78bfa","#34d399","#f59e0b"]))
    with right:
        st.markdown("**Entity degree distribution**")
        if degree_col: plot(px.histogram(entities,x=degree_col,nbins=35,color_discrete_sequence=["#f59e0b"]))
    left,right=st.columns(2)
    with left:
        st.markdown("**Top connected entities**")
        if degree_col:
            top=entities.nlargest(20,degree_col).copy()
            top["entity_label"]=top.entity_id.astype(str).str.slice(0,32)
            plot(px.bar(top.sort_values(degree_col),x=degree_col,y="entity_label",color="node_type" if "node_type" in top else None,
                        orientation="h",hover_data=[c for c in ["component_size","betweenness"] if c in top]),500)
    with right:
        st.markdown("**Connected component size distribution**")
        if "component_size" in entities:
            components=entities.drop_duplicates("entity_id").component_size.value_counts().head(20).sort_index().rename_axis("component_size").reset_index(name="entities")
            plot(px.bar(components,x="component_size",y="entities",color_discrete_sequence=["#34d399"]))
    st.markdown("#### Community structure")
    if not communities.empty:
        l,r=st.columns([1.2,1])
        with l:
            plot(px.bar(communities.head(20),x="community_id",y="nodes",color="nodes",
                        title="Largest communities by node count",color_continuous_scale="Teal"),360)
        with r:
            plot(px.scatter(communities,x="nodes",y="density",size="edges",hover_data=["community_id"],
                            title="Community size vs density",color="density",color_continuous_scale="Viridis"),360)
        st.dataframe(communities.head(50),use_container_width=True,hide_index=True)
    st.markdown("#### Entity-level investigation table")
    filters=st.columns(2)
    node_types=sorted(entities.node_type.dropna().astype(str).unique()) if "node_type" in entities else []
    chosen=filters[0].multiselect("Entity types",node_types,default=node_types)
    min_degree=filters[1].slider("Minimum degree",0,int(entities[degree_col].max()) if degree_col else 0,0) if degree_col else 0
    entity_view=entities.copy()
    if chosen and "node_type" in entity_view: entity_view=entity_view[entity_view.node_type.astype(str).isin(chosen)]
    if degree_col: entity_view=entity_view[entity_view[degree_col]>=min_degree]
    st.dataframe(entity_view.head(500),use_container_width=True,height=420,hide_index=True)
    download(entity_view,"filtered_graph_entity_metrics.csv","Download filtered graph metrics")
    st.caption("Graph indicators describe connectivity in the bounded source snapshot. They are not evidence of malicious behavior.")

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

elif page=="Executive Report":
    st.subheader("Executive reporting")
    st.caption("Download a point-in-time snapshot of the currently loaded research bundle. Reports describe the available data; they do not imply causal impact or confirmed fraud.")
    high=int((customers.behavioral_risk_band=="High").sum()) if "behavioral_risk_band" in customers else 0
    extreme=int((customers.anomaly_band=="Extreme").sum()) if "anomaly_band" in customers else 0
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Customers",f"{len(customers):,}")
    k2.metric("Transactions",f"{int(meta.get('transactions',0)):,}")
    k3.metric("High relative-risk",f"{high:,}")
    k4.metric("Extreme anomalies",f"{extreme:,}")
    report_cols=[c for c in ["user_id","segment_id","transaction_count","total_spend","mean_transaction_value","days_since_last_purchase","behavioral_risk_index","behavioral_risk_band","anomaly_score","anomaly_rank","anomaly_band","risk_explanations"] if c in customers]
    download(customers[report_cols],"customer_intelligence_full.csv","Download full customer intelligence CSV")
    if not segments.empty: download(segments,"customer_segment_summary.csv","Download segment summary CSV")
    if not entities.empty: download(entities,"graph_entity_metrics.csv","Download graph metrics CSV")
    if transactions is not None and not transactions.empty: download(transactions,"transaction_anomaly_candidates.csv","Download transaction anomaly CSV")
    try:
        pdf=executive_pdf(customers,segments,meta)
        st.download_button("Download executive PDF report",pdf,"customer_intelligence_executive_report.pdf","application/pdf")
    except Exception as exc:
        st.error(f"PDF report generation unavailable: {exc}. Check that ReportLab is installed in the app environment.")

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
