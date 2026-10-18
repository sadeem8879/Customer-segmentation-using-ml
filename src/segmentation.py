"""Reusable data validation, preprocessing, and K-Means utilities."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
DEFAULT_K_RANGE = (2, 8)
ID_COLUMNS = {"customerid", "customer id", "id"}


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 fingerprint for dataset content and schema."""
    normalized = normalize_columns(df)
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, normalized.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(normalized, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Mall Customers headers without changing other names."""
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    aliases = {
        "annual income": FEATURES[0], "annual income (k$)": FEATURES[0], "income": FEATURES[0],
        "spending score": FEATURES[1], "spending score (1-100)": FEATURES[1], "spending": FEATURES[1],
        "customerid": "CustomerID", "customer id": "CustomerID", "gender": "Gender", "sex": "Gender",
    }
    rename = {c: aliases[c.lower().replace("_", " ").strip()] for c in result.columns
              if c.lower().replace("_", " ").strip() in aliases}
    return result.rename(columns=rename)


def validate_dataset(df: pd.DataFrame, features: list[str] | None = None,
                     min_rows: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean selected fields; report every excluded record."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The CSV has no data rows. Choose a non-empty dataset.")
    clean = normalize_columns(df)
    if clean.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")
    features = features or [c for c in clean if c not in ID_COLUMNS and c.lower() not in {"customerid", "gender"}
                            and pd.api.types.is_numeric_dtype(clean[c])]
    missing = [c for c in features if c not in clean.columns]
    if not features or missing:
        raise ValueError("Select existing feature columns. Missing: " + (", ".join(missing) or "no usable features"))
    if any(c.lower() in ID_COLUMNS or c == "CustomerID" for c in features):
        raise ValueError("Identifier columns such as CustomerID cannot be clustering features.")
    work = clean.copy()
    invalid = 0
    for col in features:
        converted = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Mixed numeric/text columns are treated as numeric when at least one
        # non-null value parses numerically; pure labels remain categorical.
        if (pd.api.types.is_numeric_dtype(work[col]) or
                (work[col].notna().any() and converted.notna().any())):
            invalid += int((work[col].notna() & converted.isna()).sum())
            work[col] = converted
    # Exact duplicate rows are reported and removed to prevent weighting repeats accidentally.
    duplicate_count = int(work.duplicated().sum())
    work = work.drop_duplicates()
    rows_before_missing = len(work)
    work = work.dropna(subset=features)
    dropped_missing = rows_before_missing - len(work)
    if len(work) < min_rows:
        raise ValueError(f"At least {min_rows} usable rows are required; found {len(work)}.")
    summary = {"input_rows": len(df), "rows_used": len(work), "duplicates_removed": duplicate_count,
               "rows_with_missing_features_removed": dropped_missing, "invalid_numeric_values": invalid,
               "features": list(features)}
    return work.reset_index(drop=True), summary


def _preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                    ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                        ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def fit_candidate_models(df: pd.DataFrame, min_k: int = 2, max_k: int = 8,
                         random_state: int = 42, features: list[str] | None = None,
                         n_init: int = 10):
    """Fit preprocessing once and candidate models; return actual metrics and fitted objects."""
    features = list(features or FEATURES)
    clean, summary = validate_dataset(df, features)
    if min_k < 2 or max_k < min_k:
        raise ValueError("K range must start at 2 or higher and max K must be >= min K.")
    upper = min(max_k, len(clean) - 1)
    if min_k > upper:
        raise ValueError(f"Dataset has {len(clean)} rows, insufficient for candidate K={min_k}.")
    prep = _preprocessor(clean, features)
    X = prep.fit_transform(clean[features])
    if hasattr(X, "toarray"):
        X = X.toarray()
    rows, models = [], {}
    for k in range(min_k, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)
        score = float(silhouette_score(X, labels)) if len(set(labels)) > 1 and len(set(labels)) < len(labels) else np.nan
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette_score": score})
        models[k] = model
    results = pd.DataFrame(rows)
    # Silhouette is a recommendation aid, not a universal optimum.
    valid = results.dropna(subset=["silhouette_score"])
    best_k = int(valid.loc[valid.silhouette_score.idxmax(), "k"]) if not valid.empty else int(min_k)
    return clean, prep, models, results, best_k, summary


def train_run(df: pd.DataFrame, features: list[str] | None = None, min_k: int = 2,
              max_k: int = 8, random_state: int = 42, source: str = "local CSV") -> dict[str, Any]:
    """Build a serializable, synchronized run artifact."""
    import time
    started = time.perf_counter()
    clean, prep, models, results, best_k, summary = fit_candidate_models(
        df, min_k, max_k, random_state, features)
    model = models[best_k]
    transformed = prep.transform(clean[list(features or FEATURES)])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    labels = model.predict(transformed) + 1
    clustered = clean.copy()
    clustered["Cluster"] = labels
    summary.update({"dataset_source": source, "dataset_fingerprint": dataset_fingerprint(df),
                    "selected_k": best_k, "random_state": random_state,
                    "n_init": 10, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "training_seconds": time.perf_counter() - started,
                    "preprocessing": "numeric: median imputation + StandardScaler; categorical: most-frequent imputation + one-hot encoding"})
    chosen = results.loc[results.k == best_k, "silhouette_score"].iloc[0]
    summary["selected_silhouette_score"] = None if pd.isna(chosen) else float(chosen)
    return {"model": model, "preprocessor": prep, "features": list(features or FEATURES), "selected_k": best_k,
            "results": results, "clustered": clustered, "metadata": summary}


def predict_customer(artifact: dict[str, Any], customer: dict[str, Any] | pd.DataFrame) -> int:
    """Predict a one-based cluster using the fitted preprocessing and active model."""
    if not artifact or not all(k in artifact for k in ("model", "preprocessor", "features")):
        raise ValueError("A compatible trained model artifact is required. Train a model first.")
    row = customer if isinstance(customer, pd.DataFrame) else pd.DataFrame([customer])
    missing = [c for c in artifact["features"] if c not in row]
    if missing:
        raise ValueError("Prediction is missing feature(s): " + ", ".join(missing))
    transformed = artifact["preprocessor"].transform(row[artifact["features"]])
    return int(artifact["model"].predict(transformed)[0]) + 1


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not all(k in artifact for k in ("model", "preprocessor", "features", "selected_k")):
        raise ValueError("Model artifact is incomplete or incompatible. Retrain with the current project version.")
    if list(artifact["preprocessor"].feature_names_in_) != list(artifact["features"]):
        raise ValueError("Artifact feature order does not match its fitted preprocessing pipeline. Retrain the model.")
    return artifact
