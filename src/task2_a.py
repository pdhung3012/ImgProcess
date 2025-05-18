import pandas as pd
import matplotlib.pyplot as plt

# Load data
test_data = pd.read_csv('../fashion-mnist/fashion-mnist_missing.csv')

# Identify invalid pixels (values outside [0, 255])
invalid = (test_data.iloc[:, 1:] > 255) | (test_data.iloc[:, 1:] < 0)
incomplete_rows = invalid.any(axis=1)

# Get indexes of invalid rows
invalid_indexes = test_data[incomplete_rows].index.tolist()
print("Indexes with invalid pixels:", invalid_indexes[:10])

# Display top 10 images with invalid pixels
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Top 10 Images with Invalid Pixels', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i >= len(invalid_indexes):
        break
    idx = invalid_indexes[i]
    image = test_data.iloc[idx, 1:].values.reshape(28, 28)
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_title(f'Index: {idx}')
    ax.axis('off')

plt.tight_layout()
plt.show()
