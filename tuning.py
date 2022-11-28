import convert_csv
import hyperparameters_test
import hyperparameters_tuning

FILE_NAME = 'CRN5_isomeric_pd/data'
CRN_NAME = 'CRN5'
NUM_PARAMS = 6

# loading data
X_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_train1.csv')
y_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_train1.csv')
X_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_valid1.csv')
y_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_valid1.csv')

train_data = [X_train1, y_train1]
valid_data = [X_valid1, y_valid1]

N_COMPS = 4
MIXTURE = 'NB'

def testing_function(lr: list, max_rounds: list, batchsize: list, n_hidden: list):
    return hyperparameters_test.test_comb(lr, 
                    max_rounds, 
                    batchsize,
                    n_hidden,
                    train_data,
                    valid_data,
                    NUM_PARAMS,
                    N_COMPS,
                    (False, None, None),
                    MIXTURE)

def testing_function2(lr: list, max_rounds: list, batchsize: list, n_hidden: list, patience: list, delta: list):
    return hyperparameters_test.test_comb(lr, 
                    max_rounds,
                    batchsize,
                    n_hidden,
                    train_data,
                    valid_data,
                    NUM_PARAMS,
                    N_COMPS,
                    (True, patience, delta),
                    MIXTURE)


EARLY_STOPPING = True
n_rounds = [300, 500]
lrs = [0.005, 0.001]
batchsizes = [32]
n_hidden = [256]
patience = [50, 60]
delta = [0, 1e-6, 1e-5]


if __name__ == '__main__':
    if EARLY_STOPPING:
        hyperparameters_tuning.test_multiple_combs(testing_function2, lrs, n_rounds, batchsizes, n_hidden, (True, patience, delta), 'CRN5_optimisation_early_stopping2')
    else:
        hyperparameters_tuning.test_multiple_combs(testing_function, lrs, n_rounds, batchsizes, n_hidden, (False, None, None), 'CRN5_optimisation_early_stopping2')