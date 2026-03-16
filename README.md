# MSCS-634: Advanced Big Data and Data Mining
### Lab 5: Clustering Techniques Using DBSCAN and Hierarchical Clustering

**Name:** Oishani Ganguly
---

## Purpose

The goal of this lab is to apply and compare two unsupervised clustering algorithms — **Agglomerative Hierarchical Clustering** and **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** — on the Wine Dataset from sklearn. Unlike the classification task in Lab 2, here the algorithms receive no label information; true cultivar labels are reserved exclusively for post-hoc evaluation. Specifically, the lab covers:

- **Data Preparation and Exploration:** Loading the Wine dataset, examining its structure with `.head()`, `.info()`, and `.describe()`, confirming there are no missing values, and applying StandardScaler to normalize all 13 features before distance-based clustering
- **Hierarchical Clustering:** Applying Agglomerative Clustering with Ward linkage across n_clusters ∈ {2, 3, 4, 5}; generating and interpreting a dendrogram to identify the natural cluster cut point; evaluating each partition with Silhouette, Homogeneity, and Completeness scores
- **DBSCAN Clustering:** Running a parameter grid over eps ∈ {1.5, 2.0, 2.5, 3.0} and min_samples ∈ {3, 5, 7}; using a k-distance plot to guide eps selection; visualizing cluster assignments and noise points; reporting evaluation metrics for each configuration
- **Analysis and Insights:** Producing side-by-side comparison visualizations and a summary metrics table; discussing how parameter choices influenced outcomes; identifying the strengths and weaknesses of each algorithm based on observed results

The dataset used is the [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) from `sklearn.datasets`, containing 178 wine samples from three Italian cultivars described by 13 continuous chemical features.

---

## Key Insights

### Hierarchical Clustering

- The dendrogram generated with Ward linkage clearly supports **n=3 as the natural number of clusters**: the merge distance between 3 and 2 clusters is substantially larger than any within-cluster merge, indicating a genuine gap in the data's density structure.
- At n=3, the algorithm achieves the best balance across all three metrics — high Silhouette Score, Homogeneity, and Completeness — confirming that the discovered clusters correspond closely to the true wine cultivar boundaries.
- n=2 produces the single highest Silhouette Score (well-separated two-group partition) but at the cost of merging two cultivars, reflected in lower Homogeneity and Completeness.
- n=4 and n=5 begin splitting natural clusters artificially, degrading all metrics as the algorithm fragments coherent groups.
- Ward linkage's objective of minimizing within-cluster variance is well-matched to the Wine dataset's compact, roughly spherical clusters in the standardized feature space.

### DBSCAN Clustering

- The k-distance plot (5th nearest neighbor distances) revealed an elbow around eps=2.0–2.5 in the standardized feature space, directing the grid search toward the most meaningful parameter range.
- Small eps values (1.5) produce excessive noise — most points cannot find enough neighbors to qualify as core points, and the algorithm collapses to a near-degenerate solution.
- Moderate eps values (2.0–2.5) yield 2–3 clusters with a moderate noise fraction, producing the best Silhouette Scores among valid multi-cluster configurations.
- Large eps values (3.0) absorb most points into a single mega-cluster, losing the multi-group structure entirely.
- Noise points (flagged as label=-1) correspond to chemically atypical wine samples that sit in low-density regions — wines that don't belong firmly to any cultivar's tight cluster. This explicit outlier identification is DBSCAN's most distinctive and practically valuable feature.

### Algorithm Comparison

- **Hierarchical Clustering (Ward, n=3)** outperforms DBSCAN on all three evaluation metrics for the Wine dataset. The compact, uniformly-dense cultivar clusters are ideally suited to Ward linkage.
- **DBSCAN** is better suited to datasets with irregular cluster shapes, varying densities, or meaningful outlier populations. On Wine — a small, clean, uniform-density dataset — DBSCAN's density-contrast mechanism does not gain traction.
- For deployment on this dataset, Hierarchical Clustering with n=3 is the recommended algorithm.

---

## Challenges and Decisions

**No Train/Test Split:**
Clustering is an unsupervised task — there is no target variable to predict, so the full dataset is used for fitting. The true class labels are stored separately and only used after clustering to compute Homogeneity and Completeness; they are never passed to the clustering algorithms themselves.

**PCA for Visualization:**
Clustering runs on all 13 standardized features. PCA is applied separately and used purely for 2D scatter plot visualization. The two principal components capture approximately 55% of total variance — enough to reveal the broad cluster structure while acknowledging that some information is lost in the projection.

**Dendrogram Truncation:**
The full dendrogram for 178 samples is visually cluttered. `truncate_mode='lastp'` is used to display only the last 30 merges, which captures the high-level cluster structure (the three main branches) while keeping the chart readable.

**DBSCAN Eps Calibration:**
The lab specifies eps values of 1.5–3.0. On 13-dimensional standardized data, these values are geometrically appropriate — each standardized feature has a unit-variance spread, so a radius of 2.0 corresponds roughly to "within 2 standard deviations" across all dimensions jointly. The k-distance plot was used to independently confirm that this range straddles the elbow, validating the parameter choices empirically.

**Silhouette Score with Noise:**
DBSCAN noise points (label=-1) are excluded from Silhouette Score computation, since they are not assigned to any cluster and would distort the metric. Configurations that produce fewer than 2 valid clusters are reported as N/A for Silhouette.

**Visualization Design:**
All charts were built with `matplotlib` only to ensure the notebook runs in any standard Python environment without additional dependencies.
