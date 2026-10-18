## Existing bundled file

`Mall_Customers.csv` currently contains 200 synthetic practice rows matching the former seeded fallback generator. It is not an authentic or verified Mall Customers dataset. Keep it as practice data; do not cite its outputs as real customer research. This file is retained unchanged.

## Recommended genuine public source

Use UCI Machine Learning Repository's **Wholesale customers** dataset: https://archive.ics.uci.edu/dataset/292/wholesale+customers. The repository describes 440 clients of a wholesale distributor and six annual product-category spending amounts. UCI lists the dataset license as Creative Commons Attribution 4.0 International (CC BY 4.0): sharing and adaptation are permitted with appropriate credit. Cite it as: Cardoso, M. (2013). *Wholesale customers* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X.

### Obtain and prepare it

1. Open the official UCI page above and use its **Download** control; download is a manual user action. Do not substitute a third-party mirror.
2. Extract the archive and keep an untouched copy of the downloaded source file.
3. Inspect its first line. UCI's documented raw field order is `Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen`. If the downloaded CSV has no header row, add these exact headers to a working copy, preserving this field order. If headers are present, verify them against the UCI page before use.
4. Place the prepared working copy at `data/Wholesale_customers_data.csv`, or upload it in the dashboard sidebar. The application does not fetch it automatically.
5. For the primary analysis, use the six numeric annual spend columns: `Fresh Milk Grocery Frozen Detergents_Paper Delicassen`. Do not select `Region` (UCI documents it as target) or `Channel` in the primary spend-only experiment. A separate exploratory model may include `Channel` as a categorical feature if that choice is justified and recorded.

### Field meanings and limitations

- `Channel`: distributor channel (Horeca or Retail); categorical.
- `Region`: Lisbon, Oporto, or Other; UCI identifies this as a target field. Exclude it from unsupervised clustering to avoid using a supplied grouping label.
- `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`: annual spending in monetary units (m.u.) for each named category; integer-valued in the source and continuous in the data description.

The data concerns one wholesale distributor's clients in Portugal and is not a representative sample of all consumers or retailers. Currency/unit scale and collection period should not be inferred beyond the source documentation. Check the UCI page for current terms and citation guidance when submitting.

### Reproducible training

From the project root after obtaining the file:

```powershell
python train.py --dataset data/Wholesale_customers_data.csv --features Fresh Milk Grocery Frozen Detergents_Paper Delicassen --source "UCI Wholesale customers, DOI 10.24432/C5030X, CC BY 4.0"
```

This command writes the fitted artifact and generated reports. Verify the recorded source and fingerprint before citing results. No genuine-dataset metrics are reported in this repository until this command has been run successfully on the supplied UCI file.
