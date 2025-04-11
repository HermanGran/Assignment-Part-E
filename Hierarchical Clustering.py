from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


methods = ['ward', 'complete', 'average']
for method in methods:
    print(f"\nLinkage method: {method}")

    linked = linkage(X, method=method)

    # Dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title(f"Dendrogram ({method} linkage)")
    plt.tight_layout()
    plt.savefig(f"Figures/dendrogram_{method}.svg")
    plt.show()

    # Clustering
    model = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.3f}")

