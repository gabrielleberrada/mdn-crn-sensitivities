import torch
import pandas as pd

# parameters
max_y = 142
n_comps = 5
num_params = 4

# train data
X_train_data_pd = pd.read_csv('afl_data/X_train.csv', names=[f'column_{k}' for k in range(num_params + 1)])
y_train_data_pd = pd.read_csv('afl_data/y_train.csv', names=[f'column_{k}' for k in range(max_y)])
X_train = torch.tensor(X_train_data_pd.values)
y_train = torch.tensor(y_train_data_pd.values)
train_data = [X_train, y_train]

# valid data
X_valid_data_pd = pd.read_csv('afl_data/X_valid.csv', names=[f'column_{k}' for k in range(num_params + 1)])
y_valid_data_pd = pd.read_csv('afl_data/y_valid.csv', names=[f'column_{k}' for k in range(max_y)])
X_valid = torch.tensor(X_valid_data_pd.values)
y_valid = torch.tensor(y_valid_data_pd.values)
valid_data = [X_valid, y_valid]

# test data
X_test_data_pd = pd.read_csv('afl_data/X_test.csv', names=[f'column_{k}' for k in range(num_params + 1)])
y_test_data_pd = pd.read_csv('afl_data/y_test.csv', names=[f'column_{k}' for k in range(max_y)])
X_test = torch.tensor(X_test_data_pd.values)
y_test = torch.tensor(y_test_data_pd.values)
test_data = [X_test, y_test]

print('Downloaded csv files.\n')
# print(y_train.abs())