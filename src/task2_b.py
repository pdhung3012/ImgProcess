import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load original test data with possible invalid pixels
data_path = '../fashion-mnist/fashion-mnist_missing.csv'
df = pd.read_csv(data_path)

# Copy of original data for fixing
df_fixed = df.copy()

# Identify invalid pixels (outside range [0, 255])
invalid_mask = (df.iloc[:, 1:] < 0) | (df.iloc[:, 1:] > 255)
incomplete_rows = invalid_mask.any(axis=1)
invalid_indexes = df[incomplete_rows].index.tolist()

# Function to fix a single 28x28 image
def fix_image(image_flat):
    image = image_flat.reshape(28, 28)
    fixed_image = image.copy()

    for i in range(28):
        for j in range(28):
            if image[i, j] < 0 or image[i, j] > 255 or np.isnan(image[i, j]):
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < 28 and 0 <= nj < 28 and not (dx == 0 and dy == 0):
                            val = image[ni, nj]
                            if 0 <= val <= 255 and not np.isnan(val):
                                neighbors.append(val)
                if neighbors:
                    fixed_image[i, j] = np.mean(neighbors)
                else:
                    fixed_image[i, j] = 0  # fallback
    # Ensure no NaNs
    fixed_image = np.nan_to_num(fixed_image, nan=0.0)
    return fixed_image.flatten()

# Fix each invalid image
for idx in invalid_indexes:
    original_image = df.iloc[idx, 1:].values.astype(float)
    fixed_image = fix_image(original_image)
    df_fixed.iloc[idx, 1:] = fixed_image

# Final check: ensure all pixel columns contain no NaN
df_fixed.iloc[:, 1:] = df_fixed.iloc[:, 1:].fillna(0)

# Save the fixed dataframe
fixed_path = '../fashion-mnist/fashion-mnist_fixed.csv'
df_fixed.to_csv(fixed_path, index=False)
print(f"✅ Fixed dataset saved to: {fixed_path}")
print("✅ Confirmed: No NaN values in fixed dataset.")

# Visualize top 10 before/after fixes
fig, axes = plt.subplots(10, 2, figsize=(6, 20))
fig.suptitle('Top 10 Before and After Fixing Invalid Pixels', fontsize=16)

for i in range(min(10, len(invalid_indexes))):
    idx = invalid_indexes[i]

    before = df.iloc[idx, 1:].values.reshape(28, 28)
    after = df_fixed.iloc[idx, 1:].values.reshape(28, 28)

    axes[i, 0].imshow(before, cmap='gray', vmin=0, vmax=255)
    axes[i, 0].set_title(f'Before Fix (Index: {idx})')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(after, cmap='gray', vmin=0, vmax=255)
    axes[i, 1].set_title(f'After Fix (Index: {idx})')
    axes[i, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()
