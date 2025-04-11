from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Path to dataset
path = "data/WineQT.csv"

# Reading Dataset
df = pd.read_csv(path)


# Convert quality to binary class: good (>=6) vs bad (<=5)
df['quality_label'] = (df['quality'] >= 6).astype(int)


# Split into features (X) and label (y)
X = df.drop(columns=['quality', 'quality_label'])
y = df['quality_label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# PCA projection for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

k_values = [2, 3, 4, 5, 6]
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K = {k}, Silhouette Score = {score:.3f}")

    # Plot PCA clusters
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
    plt.title(f"K-Means Clustering (k={k})")
    plt.tight_layout()
    plt.savefig(f"Figures/kmeans_k{k}.svg")
    plt.show()
