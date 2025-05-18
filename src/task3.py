from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')
test_data_fixed=pd.read_csv('../fashion-mnist/fashion-mnist_fixed.csv')

X_train = data.iloc[:, 1:].values
y_train = data.iloc[:, 0].values
X_test = test_data_fixed.iloc[:, 1:].values
y_test = test_data_fixed.iloc[:, 0].values

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"KNN Test Accuracy: {acc:.4f}")