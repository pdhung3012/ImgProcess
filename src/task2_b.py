import pandas as pd
test_data = pd.read_csv('../fashion-mnist/fashion-mnist_missing.csv')

test_data_fixed = test_data.copy()
for col in test_data.columns[1:]:
    col_values = test_data_fixed[col]
    mean_val = col_values[(col_values >= 0) & (col_values <= 255)].mean()
    test_data_fixed[col] = col_values.clip(lower=0, upper=255).fillna(mean_val)

test_data_fixed.to_csv('../fashion-mnist/fashion-mnist_fixed.csv',index=None)
