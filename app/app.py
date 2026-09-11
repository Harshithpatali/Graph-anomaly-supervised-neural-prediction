"""
Fast interactive Streamlit Dashboard for E-Commerce Graph Anomaly Detection.

Optimised for low latency:
- Loads a single pre-scored lean parquet/pickle (no feature rebuild, no model fit)
- Aggressive sampling for plots
- Cached data loading
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="Fraud Graph Anomaly Intelligence",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Fast data loader — only precomputed scored table
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading scored data…")
def load_dashboard() -> pd.DataFrame:
    pkl = ROOT / "data" / "processed" / "dashboard.pkl"
    pq = ROOT / "data" / "processed" / "dashboard.parquet"
    if pkl.exists():
        df = pd.read_pickle(pkl)
    elif pq.exists():
        df = pd.read_parquet(pq)
    else:
        st.error("Precomputed dashboard data not found. Run the pipeline first.")
        st.stop()
    # Ensure dtypes that plotly likes
    if "purchase_time" in df.columns:
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], utc=True, errors="coerce")
    if "signup_time" in df.columns:
        df["signup_time"] = pd.to_datetime(df["signup_time"], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def sample_df(df: pd.DataFrame, n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Fixed sample for heavy plots — keeps UI responsive."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🕵️ Fraud Graph Anomaly")
st.sidebar.caption("Pre-scored · low-latency dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Data Explorer",
        "🕸️ Graph Insights",
        "🚨 Anomaly Ranking",
        "🔍 Investigation Workbench",
        "📈 Model Performance",
        "🧠 Neural Fraud Classifier",
        "📋 Stakeholder Findings",
        "ℹ️ About",
    ],
    label_visibility="collapsed",
)

# Load once (cached)
df = load_dashboard()
df_sample = sample_df(df, 20_000)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Dynamic E-Commerce Graph Anomaly Detection")
    st.markdown(
        "Unsupervised anomaly ranking on a multipartite transaction graph. "
        "Fraud labels are used **only for validation**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud rate (label)", f"{df['class'].mean()*100:.2f}%")
    c3.metric("Unique users", f"{df['user_id'].nunique():,}")
    c4.metric("Unique devices", f"{df['device_id'].nunique():,}")

    st.subheader("Anomaly Score Distribution")
    plot_df = df_sample.copy()
    plot_df["label"] = plot_df["class"].map({0: "Normal", 1: "Fraud (label)"})
    fig = px.histogram(
        plot_df,
        x="anomaly_score",
        color="label",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={"Normal": "#4C78A8", "Fraud (label)": "#E45756"},
        title="Anomaly scores (higher = more anomalous) — sampled",
    )
    fig.update_layout(height=360, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top-20 Most Anomalous Transactions")
    top = df.nsmallest(20, "anomaly_rank")
    show_cols = [c for c in [
        "anomaly_rank", "anomaly_score", "user_id", "device_id",
        "purchase_value", "account_age_hours", "country", "class"
    ] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, height=420)

elif page == "📊 Data Explorer":
    st.title("Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        countries = st.multiselect(
            "Country",
            options=sorted(df["country"].dropna().unique())[:80],
            default=[],
        )
    with col2:
        class_filter = st.selectbox("Fraud label", ["All", "Fraud only", "Normal only"])
    with col3:
        max_rows = st.slider("Max rows to show", 100, 2000, 500, 100)

    view = df
    if countries:
        view = view[view["country"].isin(countries)]
    if class_filter == "Fraud only":
        view = view[view["class"] == 1]
    elif class_filter == "Normal only":
        view = view[view["class"] == 0]

    st.caption(f"{len(view):,} rows match filters")

    tab1, tab2, tab3 = st.tabs(["Table", "Purchase Value", "Temporal"])
    with tab1:
        st.dataframe(view.head(max_rows), use_container_width=True, height=400)
    with tab2:
        sample = view if len(view) < 15_000 else view.sample(15_000, random_state=1)
        fig = px.box(
            sample, x="class", y="purchase_value", color="class",
            labels={"class": "Fraud label"}, title="Purchase value by label (sampled)",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if "purchase_time" in view.columns:
            daily = (
                view.dropna(subset=["purchase_time"])
                .set_index("purchase_time")
                .resample("D")
                .size()
                .reset_index(name="count")
            )
            fig = px.line(daily, x="purchase_time", y="count", title="Daily volume")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🕸️ Graph Insights":
    st.title("Graph & Entity Insights")
    st.markdown(
        "Transactions link to **user / device / IP** nodes. "
        "Shared devices and IPs are strong collusion signals."
    )

    # Pre-aggregate once (cached via st.cache_data wrapper below)
    @st.cache_data(show_spinner=False)
    def entity_stats(raw: pd.DataFrame):
        users_per_device = raw.groupby("device_id")["user_id"].nunique()
        users_per_ip = raw.groupby("ip_address")["user_id"].nunique()
        high = (
            raw.groupby("device_id")
            .agg(
                n_users=("user_id", "nunique"),
                n_tx=("user_id", "size"),
                fraud_rate=("class", "mean"),
            )
            .query("n_users > 2")
            .sort_values("fraud_rate", ascending=False)
            .head(20)
        )
        return users_per_device, users_per_ip, high

    users_per_device, users_per_ip, high = entity_stats(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg users / device", f"{users_per_device.mean():.2f}")
    c2.metric("Avg users / IP", f"{users_per_ip.mean():.2f}")
    c3.metric("Devices with >1 user", f"{(users_per_device > 1).sum():,}")

    fig = px.histogram(
        users_per_device.clip(upper=20),
        nbins=20,
        title="Users sharing a device (clipped at 20)",
        labels={"value": "Users per device"},
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-risk shared devices")
    st.dataframe(high, use_container_width=True)

elif page == "🚨 Anomaly Ranking":
    st.title("Anomaly Ranking Workbench")

    top_n = st.slider("Show top-N anomalies", 10, 300, 50, 10)
    q = st.slider(
        "Minimum score percentile",
        50, 99, 90, 1,
        help="Only show transactions above this score percentile",
    )
    min_score = float(np.percentile(df["anomaly_score"], q))

    ranked = df[df["anomaly_score"] >= min_score].nsmallest(top_n, "anomaly_rank")
    st.caption(f"{len(ranked)} transactions · min score {min_score:.4f}")

    display_cols = [c for c in [
        "anomaly_rank", "anomaly_score", "user_id", "device_id", "ip_address",
        "purchase_value", "country", "source", "browser", "class"
    ] if c in ranked.columns]
    st.dataframe(ranked[display_cols], use_container_width=True, height=480)

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranked CSV", csv, "anomaly_ranking.csv", "text/csv")

elif page == "🔍 Investigation Workbench":
    st.title("Investigation Workbench")
    st.markdown("Drill into a high-ranking transaction or entity.")

    mode = st.radio("Investigate by", ["Transaction rank", "User ID", "Device ID"], horizontal=True)

    if mode == "Transaction rank":
        rank = st.number_input("Anomaly rank (1 = most anomalous)", 1, int(df["anomaly_rank"].max()), 1)
        rows = df[df["anomaly_rank"] == rank]
        if rows.empty:
            st.warning("Rank not found")
            st.stop()
        row = rows.iloc[0]
    elif mode == "User ID":
        # Limit selectbox options for speed
        top_users = (
            df.nsmallest(3000, "anomaly_rank")["user_id"].unique().tolist()
        )
        uid = st.selectbox("User ID (from top anomalous)", top_users)
        row = df[df["user_id"] == uid].sort_values("anomaly_score", ascending=False).iloc[0]
    else:
        top_devices = (
            df.nsmallest(3000, "anomaly_rank")["device_id"].unique().tolist()
        )
        did = st.selectbox("Device ID (from top anomalous)", top_devices)
        row = df[df["device_id"] == did].sort_values("anomaly_score", ascending=False).iloc[0]

    st.subheader("Selected transaction")
    info = {
        "anomaly_rank": int(row["anomaly_rank"]),
        "anomaly_score": float(row["anomaly_score"]),
        "user_id": int(row["user_id"]),
        "device_id": str(row["device_id"]),
        "purchase_value": float(row["purchase_value"]),
        "country": str(row.get("country", "")),
        "class (label)": int(row["class"]),
    }
    if "account_age_hours" in row:
        info["account_age_hours"] = float(row["account_age_hours"]) if pd.notna(row["account_age_hours"]) else None
    st.json(info)

    same_user = df[df["user_id"] == row["user_id"]]
    same_device = df[df["device_id"] == row["device_id"]]
    same_ip = df[df["ip_address"] == row["ip_address"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Tx same user", len(same_user), f"fraud {same_user['class'].mean()*100:.1f}%")
    c2.metric("Tx same device", len(same_device), f"fraud {same_device['class'].mean()*100:.1f}%")
    c3.metric("Tx same IP", len(same_ip), f"fraud {same_ip['class'].mean()*100:.1f}%")

    st.markdown("**Other transactions on the same device**")
    cols = [c for c in ["purchase_time", "user_id", "purchase_value", "anomaly_score", "class"] if c in same_device.columns]
    st.dataframe(
        same_device[cols].sort_values("anomaly_score", ascending=False).head(25),
        use_container_width=True,
        height=320,
    )

elif page == "📈 Model Performance":
    st.title("Validation Against Fraud Labels")
    st.warning(
        "Post-hoc validation only — the model never trained on the `class` label."
    )

    from evaluation import evaluate_anomaly_scores, enrichment_factor
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = df["class"].values
    s = df["anomaly_score"].values
    metrics = evaluate_anomaly_scores(y, s)
    k = max(int(y.sum()), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    c3.metric(f"Precision@{k}", f"{metrics['precision_at_k']:.4f}")
    c4.metric("Enrichment@k", f"{enrichment_factor(y, s, k):.2f}×")

    # Curves on a sample for speed if needed (full data is fine for curves)
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        fig1.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision", height=360)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig2.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Score by label (sampled)")
    fig = px.box(
        df_sample, x="class", y="anomaly_score", color="class",
        labels={"class": "Fraud label"}, points=False,
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)




elif page == "🧠 Neural Fraud Classifier":
    st.title("Neural Fraud Classifier (MLP)")
    st.markdown(
        """
        PyTorch **MLP** with leakage-safe features.  
        Split: stratified 70/15/15 · Primary metric: **PR-AUC**  
        Artifact scores **raw CSV columns** (same as `Fraud_Data.csv` without `class`).
        """
    )

    @st.cache_resource(show_spinner="Loading neural model…")
    def load_neural():
        path = ROOT / "artifacts" / "final_supervised_neural_model.joblib"
        summary_path = ROOT / "artifacts" / "neural_fraud_summary.json"
        if not path.exists():
            return None, None
        from supervised_fraud import load_artifact
        import json as _json
        art = load_artifact(path)
        summary = _json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return art, summary

    @st.cache_data(show_spinner="Loading test predictions…")
    def load_test_predictions():
        path = ROOT / "artifacts" / "final_test_predictions.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    art, summary = load_neural()
    if art is None:
        st.error("Missing `artifacts/final_supervised_neural_model.joblib`.")
        st.stop()

    tm = summary.get("test_metrics", {})
    thr = float(summary.get("threshold") or art.metadata.get("threshold", 0.5))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion", str(summary.get("champion_type", art.model_type)).upper())
    c2.metric("Test PR-AUC", f"{tm.get('pr_auc', float('nan')):.4f}")
    c3.metric("Test ROC-AUC", f"{tm.get('roc_auc', float('nan')):.4f}")
    c4.metric("Threshold", f"{thr:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Precision", f"{tm.get('precision', float('nan')):.3f}")
    c6.metric("Recall", f"{tm.get('recall', float('nan')):.3f}")
    c7.metric("F1", f"{tm.get('f1', float('nan')):.3f}")
    c8.metric("Accuracy", f"{tm.get('accuracy', float('nan')):.3f}")

    tab_pred, tab_csv, tab_batch = st.tabs([
        "🆕 New prediction (raw fields)",
        "📄 Locked test predictions CSV",
        "📥 Batch CSV upload",
    ])

    with tab_pred:
        st.subheader("Enter one transaction (original raw columns)")
        st.caption("Same fields as `Fraud_Data.csv` except `class` (not required for scoring).")
        with st.form("neural_raw_form"):
            r1, r2 = st.columns(2)
            with r1:
                user_id = st.number_input("user_id", value=22058, step=1)
                signup_time = st.text_input("signup_time", value="2015-02-24 22:55:49")
                purchase_time = st.text_input("purchase_time", value="2015-04-18 02:47:11")
                purchase_value = st.number_input("purchase_value", value=34.0, min_value=0.0)
                device_id = st.text_input("device_id", value="QVPSPJUOCKZAR")
            with r2:
                source = st.selectbox("source", ["SEO", "Ads", "Direct"], index=0)
                browser = st.selectbox("browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"], index=0)
                sex = st.selectbox("sex", ["M", "F"], index=0)
                age = st.number_input("age", value=39, min_value=1, max_value=120)
                ip_address = st.number_input("ip_address (numeric)", value=732758368.0, format="%.0f")
            go = st.form_submit_button("Predict fraud probability", type="primary")

        if go:
            row = pd.DataFrame([{
                "user_id": int(user_id),
                "signup_time": signup_time,
                "purchase_time": purchase_time,
                "purchase_value": float(purchase_value),
                "device_id": str(device_id),
                "source": source,
                "browser": browser,
                "sex": sex,
                "age": int(age),
                "ip_address": float(ip_address),
            }])
            try:
                proba = float(art.predict_proba(row)[0, 1])
                pred = int(proba >= thr)
                if pred:
                    st.error(f"**FRAUD**  ·  probability = **{proba:.4f}**  (threshold {thr:.3f})")
                else:
                    st.success(f"**Normal**  ·  probability = **{proba:.4f}**  (threshold {thr:.3f})")
                st.json({"fraud_probability": proba, "predicted_class": pred, "threshold": thr})
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_csv:
        st.subheader("Locked test set — scored predictions")
        pred_df = load_test_predictions()
        if pred_df is None:
            st.warning("`artifacts/final_test_predictions.csv` not found.")
        else:
            st.caption(f"{len(pred_df):,} rows · columns include raw fields + fraud_probability + predicted_class + actual_class")
            filt = st.selectbox(
                "Filter",
                ["All", "Predicted fraud", "Predicted normal", "Correct", "False positive", "False negative"],
            )
            view = pred_df
            if filt == "Predicted fraud":
                view = pred_df[pred_df["predicted_class"] == 1]
            elif filt == "Predicted normal":
                view = pred_df[pred_df["predicted_class"] == 0]
            elif filt == "Correct":
                view = pred_df[pred_df["predicted_class"] == pred_df["actual_class"]]
            elif filt == "False positive":
                view = pred_df[(pred_df["predicted_class"] == 1) & (pred_df["actual_class"] == 0)]
            elif filt == "False negative":
                view = pred_df[(pred_df["predicted_class"] == 0) & (pred_df["actual_class"] == 1)]

            n_show = st.slider("Rows to display", 20, 500, 100, 20)
            st.dataframe(
                view.sort_values("fraud_probability", ascending=False).head(n_show),
                use_container_width=True,
                height=420,
            )
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full final_test_predictions.csv",
                csv_bytes,
                file_name="final_test_predictions.csv",
                mime="text/csv",
            )

    with tab_batch:
        st.subheader("Batch score a CSV (raw columns)")
        st.markdown(
            "Upload a CSV with columns: "
            "`user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address`"
        )
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                batch = pd.read_csv(up)
                missing = [c for c in [
                    "user_id", "signup_time", "purchase_time", "purchase_value",
                    "device_id", "source", "browser", "sex", "age", "ip_address"
                ] if c not in batch.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    with st.spinner("Scoring…"):
                        proba = art.predict_proba(batch)[:, 1]
                    scored = batch.copy()
                    scored["fraud_probability"] = proba
                    scored["predicted_class"] = (proba >= thr).astype(int)
                    st.dataframe(
                        scored.sort_values("fraud_probability", ascending=False).head(100),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download scored CSV",
                        scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_transactions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Batch scoring failed: {e}")


elif page == "📋 Stakeholder Findings":
    st.title("Stakeholder Findings — What We Learned")
    import json as _json
    findings_path = ROOT / "artifacts" / "stakeholder_findings.json"
    if findings_path.exists():
        findings = _json.loads(findings_path.read_text())
    else:
        findings = {}

    st.header("1. Executive summary")
    st.markdown(f"**{findings.get('headline', '')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Majority baseline accuracy", f"{findings.get('majority_baseline_accuracy', 0.95):.1%}")
    c2.metric("Fraud rate in locked test", f"{findings.get('fraud_base_rate_test', 0.046):.1%}")
    model_block = findings.get("model_on_locked_test", {})
    c3.metric("Model enrichment @ top-k", f"{model_block.get('enrichment_at_k', 0):.1f}×")

    st.header("2. Why accuracy looks “worse than random”")
    st.warning(findings.get("why_accuracy_misleads", ""))
    st.markdown(
        """
        | Approach | Approx. accuracy on test | Catches fraud? |
        |----------|--------------------------|----------------|
        | Always predict normal | **~95%** | No |
        | Supervised model (F1 threshold) | ~92% | Partially — ranks risk |
        | Random guessing of fraud labels | ~50% balanced | No |
        """
    )
    st.caption("“Worse than majority accuracy” ≠ “worse than random ranking”. Ranking metrics are the right lens.")

    st.header("3. Critical data insight (concept drift)")
    for bullet in findings.get("key_data_insights", []):
        st.markdown(f"- {bullet}")

    st.header("4. Simple rule vs model")
    rule = findings.get("simple_rule_train", {})
    st.markdown(
        f"""
        **Rule:** `{rule.get('rule', '')}`

        - Train precision: **{rule.get('train_precision', '—')}** (historically perfect on that pattern)
        - Train recall of fraud: **{rule.get('train_recall_of_fraud', '—')}**
        - Val/Test recall: **{rule.get('val_test_recall', '—')}**
        - Note: {rule.get('note', '')}
        """
    )

    st.header("5. Model quality on locked future test")
    st.markdown(
        f"""
        - **Model:** {model_block.get('selected_model', '—')}
        - **PR-AUC:** {model_block.get('pr_auc', '—')} (higher than base rate ≈ 0.046)
        - **ROC-AUC:** {model_block.get('roc_auc', '—')}
        - **Precision@k:** {model_block.get('precision_at_k', '—')} with **{model_block.get('enrichment_at_k', '—')}×** enrichment
        - **Accuracy:** {model_block.get('accuracy', '—')} (below majority baseline — expected)

        {model_block.get('interpretation', '')}
        """
    )

    st.header("6. Recommended use for the business")
    for rec in findings.get("recommended_use", []):
        st.markdown(f"- {rec}")
    og = findings.get("operating_guidance", {})
    st.markdown(f"**Default workflow:** {og.get('default', '')}")
    st.markdown(f"**Threshold policy:** {og.get('threshold_policy', '')}")

    st.header("7. Portfolio takeaway")
    st.success(
        "This project is honest about limits: chronological evaluation, label used only for validation "
        "in the unsupervised track, explicit imbalance handling, and clear separation between "
        "ranking quality and accuracy. The main scientific finding is severe **temporal drift** "
        "in attack patterns — instant signup fraud dominates early data and vanishes later."
    )

elif page == "ℹ️ About":
    st.title("About this project")
    st.markdown(
        """
        ### Scientific stance
        - Fraud label is **validation ground truth only**
        - Chronological train / val / test split
        - Graph features (degree, shared device/IP) + temporal + behavioural
        - This UI loads **pre-scored** data for low latency

        ### Pipeline
        Dataset → Audit → Graph → Features → IsolationForest → Ranking → Validation → Dashboard

        ### Run
        ```bash
        pip install -r requirements.txt
        streamlit run app/app.py
        ```
        """
    )
