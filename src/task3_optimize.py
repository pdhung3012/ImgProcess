import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# I have several strategies to improve the original baseline:
# 1) Baseline KNN
#
# 2) Standardized KNN
#
# 3) Hyperparameter-tuned KNN
#
# 4) PCA + KNN
#
# 5) Random Forest

# Load training and fixed test data
train = pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')
test = pd.read_csv('../fashion-mnist/fashion-mnist_fixed.csv')

X_train, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
X_test, y_test = test.iloc[:, 1:].values, test.iloc[:, 0].values

results = {}

# ───────────── Baseline KNN ─────────────
knn_baseline = KNeighborsClassifier(n_neighbors=3)
knn_baseline.fit(X_train, y_train)
preds = knn_baseline.predict(X_test)
acc = accuracy_score(y_test, preds)
results['Baseline KNN (k=3)'] = acc

# ───────────── Standardized KNN ─────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
preds = knn_scaled.predict(X_test_scaled)
results['Standardized KNN (k=3)'] = accuracy_score(y_test, preds)

# ───────────── Hyperparameter Tuning ─────────────
print("\n[Hyperparameter Tuning for KNN]")
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[f'KNN with k={k}'] = acc
    print(f"  k={k}: Accuracy = {acc:.4f}")

# ───────────── PCA + KNN ─────────────
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_train_pca, y_train)
preds = knn_pca.predict(X_test_pca)
results['PCA (50) + KNN (k=3)'] = accuracy_score(y_test, preds)

# ───────────── Random Forest ─────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
results['Random Forest'] = accuracy_score(y_test, rf_preds)

# ───────────── Print and Plot Results ─────────────
print("\n=== Accuracy Summary ===")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

# Optional: Plot accuracies
plt.figure(figsize=(10, 6))
plt.barh(list(results.keys()), list(results.values()), color='skyblue')
plt.xlabel("Accuracy")
plt.title("Accuracy Comparison of Models and Strategies")
plt.grid(axis='x')
plt.xlim(0.7, 1.0)
plt.tight_layout()
plt.show()

