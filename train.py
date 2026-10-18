"""Train on a user-supplied local CSV and write reproducible outputs."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.segmentation import FEATURES, train_run, save_artifact

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
ARTIFACT_PATH = ROOT / "models" / "customer_segmentation_artifact.pkl"
REPORTS = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA_PATH, help="Path to a local CSV; no download or generated fallback is used.")
    parser.add_argument("--features", nargs="+", default=FEATURES, help="Exact clustering feature names in desired model order.")
    parser.add_argument("--source", default=None, help="Dataset provenance label recorded in the model artifact.")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}. Supply a local compatible CSV; training does not download or synthesize data.")
    raw = pd.read_csv(args.dataset)
    source = args.source or ("Synthetic practice data (bundled file; see data/README.txt)"
                             if args.dataset.resolve() == DATA_PATH.resolve()
                             else f"User-supplied local CSV: {args.dataset.name}; provenance must be documented by user")
    artifact = train_run(raw, args.features, args.min_k, args.max_k, args.seed, source)
    REPORTS.mkdir(exist_ok=True)
    artifact["results"].to_csv(REPORTS / "cluster_evaluation.csv", index=False)
    profiles = artifact["clustered"].groupby("Cluster")[artifact["features"]].agg(["count", "mean", "median"]).round(2)
    profiles.to_csv(REPORTS / "cluster_profiles.csv")
    artifact["clustered"].to_csv(ROOT / "data" / "clustered_customers.csv", index=False)
    save_artifact(artifact, ARTIFACT_PATH)
    sns.set_theme(style="whitegrid")
    for metric, title, y_label, filename in [
        ("inertia", "K-Means Elbow Curve", "Inertia (within-cluster sum of squares)", "elbow_method.png"),
        ("silhouette_score", "Silhouette Score by Candidate K", "Silhouette score", "silhouette_scores.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=artifact["results"], x="k", y=metric, marker="o", ax=ax)
        ax.set(title=title, xlabel="Number of clusters (K)", ylabel=y_label)
        fig.tight_layout(); fig.savefig(REPORTS / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_features = [f for f in artifact["features"] if pd.api.types.is_numeric_dtype(artifact["clustered"][f])]
    if len(numeric_features) >= 2:
        sns.scatterplot(data=artifact["clustered"], x=numeric_features[0], y=numeric_features[1], hue="Cluster", palette="tab10", ax=ax)
    else:
        ax.text(0.5, 0.5, "Scatter plot requires at least two numeric selected features", ha="center", va="center")
    ax.set_title(f"Customer segments (K={artifact['selected_k']})")
    fig.tight_layout(); fig.savefig(REPORTS / "customer_segments.png", dpi=160); plt.close(fig)
    print(f"Training complete: {artifact['metadata']['rows_used']} rows; K={artifact['selected_k']}; "
          f"silhouette={artifact['metadata']['selected_silhouette_score']}")
    print(f"Dataset source: {source}")
    print(f"Artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
