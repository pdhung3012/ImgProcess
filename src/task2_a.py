import pandas as pd
test_data = pd.read_csv('../fashion-mnist/fashion-mnist_missing.csv')
invalid = (test_data.iloc[:, 1:] > 255) | (test_data.iloc[:, 1:] < 0)
incomplete_rows = invalid.any(axis=1)
print("Indexes with invalid pixels:", test_data[incomplete_rows].index.tolist())