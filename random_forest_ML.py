import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/WineQT.csv")

# Converting to binary class
df['quality_label'] = (df['quality'] >= 6).astype(int)
df = df.drop(columns=['quality', 'Id'])

# Splitting into to X (Features) and y (Quality)
X = df.iloc[:, 0:11]
y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.3)

# Try different numbers of trees (n_estimators)
tree_counts = [250, 500, 1000]

print("Random Forest Experiments:")

for n in tree_counts:
    print(f"\nNumber of Trees: {n}")

    rf = RandomForestClassifier(n_estimators=n, random_state=42,  criterion='entropy', max_depth=14, min_samples_split=8)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print(classification_report(y_test, y_pred))
