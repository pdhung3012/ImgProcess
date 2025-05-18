import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')
index = 2351
pixels = data.iloc[index, 1:].values  # First column is label

pixels_2d = pixels.reshape(28, 28)

cross_img = pixels_2d.copy()
cross_img[13:15, :] = 255  # horizontal bar
cross_img[:, 13:15] = 255  # vertical bar
plt.imshow(cross_img, cmap='gray')
plt.title('Image with Cross Added')
plt.show()