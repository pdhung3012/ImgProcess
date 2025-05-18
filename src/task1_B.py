import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')
index = 2351
pixels = data.iloc[index, 1:].values  # First column is label

pixels_2d = pixels.reshape(28, 28)
print("Pixel Intensities (28x28):")
print(pixels_2d)
plt.imshow(pixels_2d, cmap='gray')
plt.title(f'Image {index}')
plt.colorbar()
plt.show()