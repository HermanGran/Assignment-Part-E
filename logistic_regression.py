import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Recourses used for creating this algorithm https://www.youtube.com/watch?v=aL21Y-u0SRs

df = pd.read_csv('data/WineQT.csv')


# Converting to binary class
df['quality_label'] = (df['quality'] >= 6).astype(int)
df = df.drop(columns=['quality', 'Id'])

# Splitting into to X (Features) and y (Quality)
X = df.iloc[:, 0:11]
y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.3)

lr = LogisticRegression(max_iter=1000, C=2.5, random_state=100)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(y_pred)

print(lr.score(X_test, y_test))

print(classification_report(y_test, y_pred))
