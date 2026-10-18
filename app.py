"""Streamlit dashboard for exploring and training customer segments."""
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.segmentation import (FEATURES, normalize_columns, train_run, load_artifact,
                              predict_customer, dataset_fingerprint)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "wholesale+customers" / "Wholesale customers data.csv"
ARTIFACT_PATH = ROOT / "models" / "customer_segmentation_artifact.pkl"

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("Customer Segmentation Using Machine Learning")
st.caption("AIML mini-project · K-Means clustering · Cluster labels describe groups, not rankings")

uploaded = st.sidebar.file_uploader("Active dataset CSV", type="csv", help="A different uploaded file invalidates results trained on another dataset.")
try:
    active_raw = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DATA_PATH)
    active_raw = normalize_columns(active_raw)
    active_fingerprint = dataset_fingerprint(active_raw)
except FileNotFoundError:
    active_raw, active_fingerprint = None, None
except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    active_raw, active_fingerprint = None, None
    st.sidebar.error(f"Dataset could not be read: {exc}")

def artifact_matches_active_dataset(run):
    """Check that a run belongs to the active data and source context."""
    if not run or not active_fingerprint:
        return False
    metadata = run.get("metadata", {})
    if metadata.get("dataset_fingerprint") != active_fingerprint:
        return False
    source = str(metadata.get("dataset_source", ""))
    # A byte-identical upload is still a distinct user-selected source.
    return source.startswith("Uploaded CSV") if uploaded is not None else not source.startswith("Uploaded CSV")


if active_raw is None:
    st.session_state.active_run = None
    st.session_state.active_source = None
else:
    current_run = st.session_state.get("active_run")
    if not artifact_matches_active_dataset(current_run):
        st.session_state.active_run = None
        st.session_state.active_source = None
        if ARTIFACT_PATH.exists():
            try:
                saved_run = load_artifact(ARTIFACT_PATH)
            except (ValueError, OSError, KeyError, AttributeError):
                saved_run = None
            if artifact_matches_active_dataset(saved_run):
                st.session_state.active_run = saved_run
                st.session_state.active_source = f"Saved model artifact · {saved_run['metadata']['dataset_source']}"

    if st.session_state.active_run is None:
        st.sidebar.warning("No compatible trained model is available for the active dataset. Train a model or upload a dataset and train it.")

page = st.sidebar.radio("Navigate", ["Overview", "Dataset Explorer", "Train / Upload", "Evaluation", "Customer Segments", "Predict"])
artifact = st.session_state.active_run

def profile_table(run):
    data, features = run["clustered"], run["features"]
    profile_fields = list(dict.fromkeys(features + ["Age", *FEATURES]))
    numeric = [c for c in profile_fields if c in data and pd.api.types.is_numeric_dtype(data[c])]
    return data.groupby("Cluster").agg(customer_count=("Cluster", "size"), **{
        c: (c, "mean") for c in numeric}).reset_index()

if page == "Overview":
    st.subheader("Project overview")
    st.write("This application groups customers with similar selected attributes. It validates the dataset, preprocesses features, evaluates candidate K values, fits K-Means, profiles the resulting groups, and can assign a new customer using the same fitted preprocessing pipeline.")
    if artifact:
        meta = artifact["metadata"]
        a, b, c, d = st.columns(4)
        a.metric("Customers used", meta["rows_used"]); b.metric("Features", len(artifact["features"]))
        c.metric("Selected K", artifact["selected_k"])
        d.metric("Silhouette", "N/A" if meta["selected_silhouette_score"] is None else f"{meta['selected_silhouette_score']:.3f}")
        st.info(f"Active results: {st.session_state.active_source}. Source: {meta['dataset_source']}; trained {meta['trained_at_utc']}.")
        st.dataframe(artifact["metadata"], use_container_width=True)
    else:
        st.warning("No compatible model is loaded. Train a model from the local dataset or upload a CSV.")
    st.markdown("**K-Means** assigns each record to the nearest of K learned centers. The elbow curve shows diminishing returns in within-cluster variation; silhouette scores summarize how separated and cohesive groups are. Neither metric alone defines the universally best K.")
elif page == "Dataset Explorer":
    try:
        if active_raw is None:
            raise ValueError("No readable dataset is active. Upload a valid CSV in the sidebar.")
        df = active_raw
        st.caption(f"Active dataset: {st.session_state.active_source or ('Uploaded CSV' if uploaded is not None else DATA_PATH.name)}")
        st.write(f"Dimensions: **{df.shape[0]} rows × {df.shape[1]} columns**")
        st.subheader("Preview"); st.dataframe(df.head(30), use_container_width=True)
        st.subheader("Columns and types"); st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()}), use_container_width=True)
        st.subheader("Descriptive statistics"); st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        st.write(f"Exact duplicate rows: **{int(df.duplicated().sum())}**")
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Could not load the local dataset: {exc}")
elif page == "Train / Upload":
    try:
        if active_raw is None:
            raise ValueError("No readable dataset is active. Upload a valid CSV in the sidebar.")
        raw = active_raw
        candidates = [c for c in raw.columns if c.lower() not in {"customerid", "customer id", "id"}]
        saved_features = (st.session_state.active_run.get("features", [])
                          if artifact_matches_active_dataset(st.session_state.active_run) else [])
        numeric_defaults = [c for c in FEATURES if c in candidates]
        default_features = [c for c in saved_features if c in candidates] or numeric_defaults or candidates[:2]
        features = st.multiselect("Clustering features", candidates, default=default_features, help="CustomerID and other identifier columns are excluded.")
        if (st.session_state.active_run and
                (st.session_state.active_run.get("metadata", {}).get("dataset_fingerprint") != active_fingerprint or
                 st.session_state.active_run.get("features") != list(features))):
            st.session_state.active_run = None
            st.session_state.active_source = None
            st.warning("Dataset or selected features changed. Previous model and results were invalidated; train again.")
            artifact = None
        left, right = st.columns(2)
        min_k = left.number_input("Minimum K", 2, max(2, len(raw)-1), 2)
        max_k = right.number_input("Maximum K", min_k, max(int(min_k), min(10, len(raw)-1)), min(8, max(2, len(raw)-1)))
        seed = st.number_input("Random seed", 0, 100000, 42)
        if st.button("Train clustering model", type="primary"):
            if not features:
                st.error("Select at least one feature.")
            else:
                with st.spinner("Validating data and evaluating candidate K values..."):
                    source = ("Uploaded CSV (source as supplied by user)" if uploaded is not None else
                              "UCI Wholesale Customers dataset" if DATA_PATH.name == "Wholesale customers data.csv" else
                              "Synthetic practice data (bundled file; see data/README.txt)")
                    run = train_run(raw, features, int(min_k), int(max_k), int(seed), source)
                st.session_state.active_run = run
                st.session_state.active_source = source
                st.success(f"Training complete. {run['metadata']['rows_used']} records used; selected K={run['selected_k']}.")
                st.dataframe(run["metadata"], use_container_width=True)
                st.dataframe(run["results"], use_container_width=True)
    except FileNotFoundError:
        st.error("Local dataset is missing. Upload a compatible CSV or place Mall_Customers.csv in data/.")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read this CSV: {exc}")
    except ValueError as exc:
        st.error(str(exc))
elif page == "Evaluation":
    if not artifact:
        st.warning("Train a model to see evaluation results.")
    else:
        results = artifact["results"]
        st.subheader("Candidate K evaluation")
        st.dataframe(results.round(4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=results, x="k", y="inertia", marker="o", ax=ax); ax.set(title="Elbow curve", xlabel="K", ylabel="Inertia"); st.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=results, x="k", y="silhouette_score", marker="o", ax=ax); ax.set(title="Silhouette comparison", xlabel="K", ylabel="Silhouette score (−1 to 1)"); st.pyplot(fig); plt.close(fig)
        st.write("Inertia always tends to decrease as K grows, so look for a bend with diminishing returns. A higher silhouette score suggests better separation and cohesion for this dataset; use both charts and the cluster profiles to make a defensible choice.")
        counts = artifact["clustered"]["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Customers")
        fig, ax = plt.subplots(figsize=(7, 4)); sns.barplot(data=counts, x="Cluster", y="Customers", ax=ax); ax.set_title("Cluster sizes"); st.pyplot(fig); plt.close(fig)
        st.subheader("Cluster profiles"); st.dataframe(profile_table(artifact).round(2), use_container_width=True)
        st.caption(f"Measured training duration: {artifact['metadata']['training_seconds']:.3f} seconds. This is run-specific and machine-dependent.")
elif page == "Customer Segments":
    if not artifact:
        st.warning("Train a model to see customer segments.")
    else:
        data, features = artifact["clustered"], artifact["features"]
        clusters = sorted(data.Cluster.unique().tolist())
        selected = st.multiselect("Filter clusters", clusters, default=clusters)
        view = data[data.Cluster.isin(selected)]
        st.dataframe(profile_table(artifact).query("Cluster in @selected").round(2), use_container_width=True)
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
        if len(numeric) >= 2 and not view.empty:
            x = st.selectbox("X axis", numeric, index=0); y = st.selectbox("Y axis", numeric, index=1)
            fig, ax = plt.subplots(figsize=(9, 5)); sns.scatterplot(data=view, x=x, y=y, hue="Cluster", palette="tab10", ax=ax); ax.set_title("Customers by cluster"); st.pyplot(fig); plt.close(fig)
        st.dataframe(view, use_container_width=True)
        st.download_button("Download assigned customers", data.to_csv(index=False).encode("utf-8"), "customer_segments.csv", "text/csv")
        st.download_button("Download cluster profiles", profile_table(artifact).to_csv(index=False).encode("utf-8"), "cluster_profiles.csv", "text/csv")
        st.download_button("Download evaluation and metadata", pd.concat([artifact["results"], pd.DataFrame([artifact["metadata"]])], ignore_index=True).to_csv(index=False).encode("utf-8"), "training_results.csv", "text/csv")
else:
    if not artifact:
        st.warning("Train or load a compatible model before prediction.")
    else:
        st.write("Enter one record. Numeric inputs use the training imputer and scaler; categories use the fitted encoder.")
        row = {}
        for feature in artifact["features"]:
            series = artifact["clustered"][feature]
            if pd.api.types.is_numeric_dtype(series):
                row[feature] = st.number_input(feature, value=float(series.median()))
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                row[feature] = st.selectbox(feature, options) if options else st.text_input(feature)
        if st.button("Predict segment", type="primary"):
            try:
                cluster = predict_customer(artifact, row)
                st.success(f"Assigned to Cluster {cluster}")
                st.caption("This is a similarity-based cluster assignment, not a guarantee of future customer behavior.")
                st.dataframe(profile_table(artifact).query("Cluster == @cluster").round(2), use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))
