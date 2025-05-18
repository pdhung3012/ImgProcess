import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')
index = 2351
pixels = data.iloc[index, 1:].values  # First column is label
plt.hist(pixels, bins=256, range=(0, 255))
plt.title(f'Gray Level Histogram for Image {index}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()