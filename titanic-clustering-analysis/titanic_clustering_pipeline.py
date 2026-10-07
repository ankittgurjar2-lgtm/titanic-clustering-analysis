"""
Week 3 Task — Unsupervised Learning and Clustering Analysis
Dataset : Titanic passenger manifest (same source used in Weeks 1-2)
Source  : Kaggle "Titanic: Machine Learning from Disaster" competition data,
          mirrored at https://github.com/mwaskom/seaborn-data (raw/titanic.csv)

This script:
  1. Rebuilds the analysis-ready dataframe (same cleaning as Weeks 1-2).
  2. Selects and scales a compact, interpretable feature set for clustering
     (the survival label is deliberately excluded from the clustering
     features themselves — it is only used afterwards, to check whether the
     unsupervised clusters happen to align with a real-world outcome).
  3. Determines a reasonable number of clusters via the elbow method and
     silhouette analysis.
  4. Fits K-Means (primary method) and Agglomerative/Hierarchical clustering
     (secondary method, for comparison and a dendrogram view).
  5. Profiles each cluster, visualizes them (PCA projection, silhouette
     diagram, dendrogram, cluster-profile heatmap), and checks cluster
     membership against survival as a post-hoc validation.

Run: python3 titanic_clustering_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

sns.set_theme(style="whitegrid", font_scale=0.95)
FIG_DIR = "figures"
DATA_IN = "data/titanic_raw.csv"
RANDOM_STATE = 42
pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# 0. REBUILD THE CLEANED DATASET (decisions justified in the Week 1 report)
# ---------------------------------------------------------------------------
def prepare_data(path):
    df = pd.read_csv(path)
    df["title"] = df["name"].str.extract(r",\s*([^\.]*)\.")
    title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
                 "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    df["title_group"] = df["title"].map(title_map).fillna("Rare")
    df["has_cabin"] = df["cabin"].notnull().astype(int)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["embarked"] = df["embarked"].fillna("S")
    df["age"] = df.groupby("title_group")["age"].transform(lambda s: s.fillna(s.median()))
    df["fare_log"] = np.log1p(df["fare"])
    df["sex_num"] = df["sex"].map({"male": 0, "female": 1})
    df["survived_label"] = df["survived"].map({0: "Died", 1: "Survived"})
    return df


section("0. DATA PREPARATION (recap of Week 1 cleaning)")
df = prepare_data(DATA_IN)
print(f"Analysis-ready shape: {df.shape}")

# ---------------------------------------------------------------------------
# 1. FEATURE SELECTION AND SCALING FOR CLUSTERING
# ---------------------------------------------------------------------------
section("1. FEATURE SELECTION AND SCALING")

FEATURES = ["age", "fare_log", "family_size", "pclass", "sex_num", "has_cabin"]
X = df[FEATURES].copy()
print("Clustering feature set:", FEATURES)
print("\nRaw feature ranges (why scaling matters):")
print(X.describe().loc[["min", "max", "mean", "std"]].round(2))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
print("\nScaled feature summary (should be ~mean 0, std 1):")
print(X_scaled_df.describe().loc[["mean", "std"]].round(2))

# ---------------------------------------------------------------------------
# 2. CHOOSING K — ELBOW METHOD + SILHOUETTE ANALYSIS
# ---------------------------------------------------------------------------
section("2. CHOOSING K")

inertias, sil_scores, ks = [], [], list(range(2, 9))
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(ks, inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[1].plot(ks, sil_scores, marker="o", color="#55A868")
axes[1].set_title("Silhouette score by k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Mean silhouette score")
best_k_by_sil = ks[int(np.argmax(sil_scores))]
axes[1].axvline(best_k_by_sil, color="red", ls="--", lw=1, label=f"best = {best_k_by_sil}")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_elbow_silhouette.png", dpi=150)
plt.close()

K = 4  # chosen: elbow bends around 4, and k=4 keeps clusters large enough to interpret
print(f"\nChosen K = {K} (see report Section 4 for the rationale)")

# ---------------------------------------------------------------------------
# 3. K-MEANS (PRIMARY METHOD)
# ---------------------------------------------------------------------------
section("3. K-MEANS CLUSTERING")

kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
sil_final = silhouette_score(X_scaled, df["cluster_kmeans"])
print(f"Final K-Means (k={K}) silhouette score: {sil_final:.3f}")
print("\nCluster sizes:\n", df["cluster_kmeans"].value_counts().sort_index())

profile = df.groupby("cluster_kmeans")[FEATURES + ["survived"]].mean().round(2)
profile["n"] = df["cluster_kmeans"].value_counts().sort_index()
print("\nCluster profiles (mean feature values, original units):\n", profile)

# PCA projection for visualization only (clustering itself used the 6 scaled features)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}, total={pca.explained_variance_ratio_[:2].sum():.2%}")

plt.figure(figsize=(6.5, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster_kmeans"], cmap="Set1", alpha=0.7, s=35)
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.title(f"K-Means clusters (k={K}) projected onto first 2 principal components")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(*scatter.legend_elements(), title="Cluster", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_pca_clusters.png", dpi=150)
plt.close()

# Silhouette diagram (per-sample)
sample_sil = silhouette_samples(X_scaled, df["cluster_kmeans"])
fig, ax = plt.subplots(figsize=(6.5, 5))
y_lower = 10
for i in range(K):
    vals = np.sort(sample_sil[df["cluster_kmeans"] == i])
    size = vals.shape[0]
    y_upper = y_lower + size
    color = cm.Set1(i / K)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, edgecolor=color, alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10
ax.axvline(sil_final, color="red", ls="--", label=f"mean = {sil_final:.2f}")
ax.set_title(f"Silhouette diagram (k={K})")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster (sorted samples)")
ax.legend(fontsize=8)
ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_silhouette_diagram.png", dpi=150)
plt.close()

# Cluster profile heatmap (z-scored for comparability across features)
profile_z = X_scaled_df.copy()
profile_z["cluster_kmeans"] = df["cluster_kmeans"].values
profile_z_mean = profile_z.groupby("cluster_kmeans").mean()
plt.figure(figsize=(6.5, 4.2))
sns.heatmap(profile_z_mean, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "Mean (standardized units)"})
plt.title("Cluster profiles (standardized feature means)")
plt.xlabel("Feature")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_cluster_profile_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. HIERARCHICAL CLUSTERING (SECONDARY METHOD, FOR COMPARISON)
# ---------------------------------------------------------------------------
section("4. HIERARCHICAL (AGGLOMERATIVE) CLUSTERING")

Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(8, 4.5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
plt.axhline(y=Z[-(K - 1), 2], color="red", ls="--", lw=1, label=f"cut for k={K}")
plt.title("Hierarchical clustering dendrogram (Ward linkage, truncated)")
plt.xlabel("Cluster (or sample count in parentheses)")
plt.ylabel("Ward distance")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_dendrogram.png", dpi=150)
plt.close()

agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
df["cluster_hier"] = agg.fit_predict(X_scaled)
sil_hier = silhouette_score(X_scaled, df["cluster_hier"])
print(f"Hierarchical clustering (k={K}) silhouette score: {sil_hier:.3f}")

ari = adjusted_rand_score(df["cluster_kmeans"], df["cluster_hier"])
print(f"Adjusted Rand Index (K-Means vs. Hierarchical): {ari:.3f}")
print("\nCrosstab of K-Means vs. Hierarchical cluster labels:")
print(pd.crosstab(df["cluster_kmeans"], df["cluster_hier"]))

# ---------------------------------------------------------------------------
# 5. POST-HOC VALIDATION AGAINST SURVIVAL (NOT USED AS A CLUSTERING FEATURE)
# ---------------------------------------------------------------------------
section("5. POST-HOC: CLUSTERS VS. SURVIVAL")

surv_by_cluster = df.groupby("cluster_kmeans")["survived"].agg(["mean", "count"]).round(3)
print(surv_by_cluster)

plt.figure(figsize=(6, 4.2))
sns.barplot(x=surv_by_cluster.index, y=surv_by_cluster["mean"], hue=surv_by_cluster.index,
            legend=False, palette="Set1")
plt.title("Survival rate by K-Means cluster\n(survival was NOT a clustering feature)", fontsize=11)
plt.xlabel("Cluster")
plt.ylabel("Survival rate")
for i, (m, c) in enumerate(zip(surv_by_cluster["mean"], surv_by_cluster["count"])):
    plt.text(i, m + 0.02, f"n={c}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_cluster_survival.png", dpi=150)
plt.close()

df.to_csv("data/titanic_clustered.csv", index=False)
print("\nSaved clustered dataset -> data/titanic_clustered.csv")
print("All figures written to ./figures/")
print("DONE.")
