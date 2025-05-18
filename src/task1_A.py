import traceback

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../fashion-mnist/fashion-mnist_train.csv')


# get index from input. Users are required to input until he/ she inputs correctly.
# index = 2351

while(True):
    try:
        index = input('get input index (input index from {} to {}): '.format(0, len(data) - 1))
        index=int(index)
        if not (index>=0 and index<len(data)):
            print('index is not in a valid range (input index from {} to {})'.format(0,len(data)-1))
        else:
            break
    except Exception as e:
        traceback.print_exc()
        print('index input is not in correct format')

# get data values only, exclude labels
pixels = data.iloc[index, 1:].values  # First column is label

# 1a get histogram of grey level
plt.hist(pixels, bins=256, range=(0, 256))
plt.title(f'Gray Level Histogram for Image {index}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()