# Customer Segmentation Using Machine Learning

## Abstract

This educational application uses K-Means, an unsupervised learning algorithm, to group customers with similar selected attributes. It provides a data explorer, preprocessing and candidate-K evaluation, cluster profiles, new-customer assignment, CSV exports, and reproducible model artifacts. The bundled `data/Mall_Customers.csv` is treated as a local input; the application does not download data or silently generate synthetic records.

## Problem statement, aim, and objectives

Retailers often need a compact way to describe customer groups for exploratory analysis. The aim is to demonstrate a complete, explainable clustering workflow. Objectives are to validate and preprocess data, scale numeric features, fit K-Means for candidate values of K, compare inertia and silhouette scores, interpret resulting profiles, assign a new record, and export computed results.

## Introduction

Customer segmentation partitions observations into groups with similar measured characteristics. The cluster IDs are arbitrary labels: Cluster 1 is not inherently better or more valuable than Cluster 2. The output can support exploratory marketing analysis, but it does not establish customer intent or predict future behavior.

## Dataset

The bundled `data/Mall_Customers.csv` has 200 rows of **synthetic practice data** matching the prior seeded fallback generator. It is not genuine Mall Customers data and must not be cited as such. Keep it labelled as practice data. The recommended genuine public source is UCI's [Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) dataset (440 distributor clients; CC BY 4.0). Attribution: Cardoso, M. (2013), *Wholesale customers* [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5030X. Instructions for manual download, column meanings, terms, and limitations are in [data/README.txt](data/README.txt). No genuine dataset is currently included or trained in this repository.

The app shows dimensions, column types, missing values, duplicate counts, preview rows, and descriptive statistics. Selected-feature records with missing values are excluded and the count is reported; exact duplicate rows are removed and counted. Numeric non-finite and non-numeric values become missing and are included in the cleanup summary. Identifier features are rejected. Numeric columns use median imputation and StandardScaler; categorical columns use most-frequent imputation and one-hot encoding. The fitted transformations are reused for prediction.

## K-Means and evaluation

K-Means iteratively assigns each observation to its nearest centroid and updates centroids to reduce within-group squared distance. Numeric features are standardized so larger measurement scales do not dominate distance. Categorical selections are one-hot encoded.

The training workflow fits candidate K values (default 2 through 8, limited by usable rows), with a fixed random seed and `n_init=10`. Inertia is the within-cluster sum of squared distances; it cannot increase as K grows, so an elbow or diminishing-return point can help select K. The silhouette score ranges from -1 to 1 and compares cohesion with separation; larger values are often preferable, but do not guarantee meaningful business segments. The app suggests the valid candidate with the highest silhouette score and shows both curves so a human can make a contextual choice. K-Means is unsupervised: classification accuracy, precision, recall, and F1 are not applicable without independent ground-truth labels.

## System workflow

1. Load local or uploaded CSV and inspect its schema.
2. Choose features and candidate K range.
3. Validate records, remove exact duplicates, and report exclusions.
4. Fit imputation, scaling/encoding, then candidate K-Means models.
5. Compare inertia, silhouette, cluster sizes, and profile statistics.
6. Assign a new customer using the saved fitted preprocessing and model.
7. Download assigned records, profiles, or evaluation and metadata.

## Results and interpretation

**Genuine-data results: pending.** No UCI file has been supplied, so this project has no genuine-dataset K, silhouette score, cluster profile, or conclusion to report. The default `python train.py` run is only a synthetic practice run. For the UCI analysis, download and prepare the dataset as described in `data/README.txt`, then run the command below. Generated outputs are dataset-specific; the artifact stores a SHA-256 content/schema fingerprint, source label, features, cleaning counts, parameters, and timestamp. The dashboard invalidates saved results when the selected dataset fingerprint changes, and invalidates the current run when its selected features change.

## Requirements and installation (Windows PowerShell)

Python 3.10 or newer is recommended. The project uses pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, and joblib. No network access is required after dependencies and local CSV are present.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py  # synthetic practice run only
streamlit run app.py
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe train.py` and `.venv\Scripts\streamlit.exe run app.py`. After manually obtaining the genuine UCI file, run:

```powershell
python train.py --dataset data/Wholesale_customers_data.csv --features Fresh Milk Grocery Frozen Detergents_Paper Delicassen --source "UCI Wholesale customers, DOI 10.24432/C5030X, CC BY 4.0"
```

Alternatively, upload a compatible CSV in the dashboard's **Active dataset CSV** sidebar control and train there. Uploading a different file invalidates model results for the previous dataset. Keep an untouched copy of downloaded/original data; do not replace the bundled practice CSV and lose its label.

## Testing

Run the deterministic unit checks with:

```powershell
python -m unittest discover -s tests -v
```

The tests cover validation, feature restrictions, invalid values, K range validation, training, prediction, and artifact round-trip. A basic end-to-end check is to run `python train.py`, then start the app using the command above. This default end-to-end run tests the application only on synthetic practice data; genuine UCI evaluation remains pending until its file is supplied.

## Software and hardware

The app is intended for a typical student laptop with a supported Python installation, several hundred MB of available memory, and a modern browser. Runtime and model quality depend on data size and characteristics; the bundled educational data is small enough for local execution.

## Advantages and applications

Clustering can help explore broad behavioral patterns, create summaries for discussion, and support hypotheses for further analysis. Results are easy to visualize and do not require pre-labeled examples.

## Limitations

K-Means favors roughly compact, distance-based clusters and can be affected by outliers, feature choice, and K. Different seeds or datasets can produce different labels. Silhouette is not a measure of business value. The bundled file is synthetic; UCI Wholesale Customers represents one distributor's Portuguese clients and is not representative of consumers broadly. No demographic fairness or causal claims are made.

## Conclusion and viva notes

The project demonstrates a reproducible unsupervised learning pipeline from CSV validation through cluster interpretation and prediction. For a viva: explain that scaling makes numeric features comparable; K controls the number of groups; inertia measures within-cluster spread; silhouette summarizes cohesion and separation; and cluster labels have no inherent ordering. The predicted label means the new row is nearest to a learned centroid after the same preprocessing, not that it will behave in a guaranteed way.

## Troubleshooting

- **Dataset missing:** add a compatible `data/Mall_Customers.csv` or upload one in the app.
- **Missing feature:** choose existing columns; headers for common income/spending fields are normalized.
- **Too few usable records:** provide at least three valid rows and ensure the requested candidate K range is feasible.
- **Old model artifact:** retrain if the dashboard says no compatible artifact is available.
- **CSV parse error:** re-save as UTF-8 comma-separated data and check for malformed quotes or duplicate headers.
- **No silhouette value:** it is undefined when a fitted result contains only one label or every row has a unique label; interpret inertia and profiles cautiously.
#   C u s t o m e r - s e g m e n t a t i o n - u s i n g - m l  
 