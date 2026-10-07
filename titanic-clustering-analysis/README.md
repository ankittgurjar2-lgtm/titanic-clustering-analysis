# Titanic Dataset — Unsupervised Learning & Clustering Analysis

Week 3 task for a data science internship: segment a public dataset with clustering,
then interpret the resulting groups. Builds on the Week 1 (data cleaning) and
Week 2 (EDA) work on the same dataset.

## Dataset

Same source as Weeks 1-2 — the raw Titanic passenger manifest (891 passengers) from
Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
competition, mirrored at
[`mwaskom/seaborn-data`](https://github.com/mwaskom/seaborn-data) (`raw/titanic.csv`).

**Important methodological note:** `survived` is deliberately excluded from the
clustering features. It is used exactly once, after clustering, purely to check
whether the unsupervised groups line up with a real outcome the algorithm never saw.

## What the pipeline does

`titanic_clustering_pipeline.py` is self-contained (re-applies Week 1's cleaning via
`prepare_data()`), then:

1. **Feature selection & scaling** — 6 interpretable features (age, log-fare, family
   size, class, sex, has-cabin), standardized with `StandardScaler`
2. **Choosing k** — elbow method + silhouette analysis across k=2-8
3. **K-Means** (primary method) — fits k=4, profiles each cluster, visualizes via PCA
   and a per-sample silhouette diagram
4. **Hierarchical clustering** (secondary method) — Ward linkage, dendrogram, and an
   Adjusted Rand Index comparison against the K-Means result
5. **Post-hoc validation** — checks cluster membership against `survived`

Full narrative, code, and all 6 figures are written up in
[`Week3_Clustering_Report.docx`](./Week3_Clustering_Report.docx).

## The four clusters found

| # | Persona | n | Survival rate |
|---|---|---|---|
| 0 | Large steerage families | 55 | 10.9% |
| 1 | Lower/middle-class solo men | 437 | 14.4% |
| 2 | Women, 2nd/3rd class | 190 | 71.6% |
| 3 | Affluent 1st class, both sexes | 209 | 65.6% |

K-Means and hierarchical clustering agree strongly on this structure (Adjusted Rand
Index = 0.862), and — despite never seeing the survival label — the four groups
reproduce the two dominant explanatory factors (sex and wealth/class) found via
supervised EDA in Week 2.

## Running it

```bash
pip install -r requirements.txt
python titanic_clustering_pipeline.py
```

Outputs a clustered CSV to `data/titanic_clustered.csv` and six charts to `figures/`.

## Structure

```
titanic_clustering_pipeline.py   # full clustering pipeline, run top to bottom
data/titanic_raw.csv             # input (raw, as downloaded)
data/titanic_clustered.csv       # output (with cluster_kmeans / cluster_hier columns)
figures/                         # six generated charts
Week3_Clustering_Report.docx     # full written report
requirements.txt
```
