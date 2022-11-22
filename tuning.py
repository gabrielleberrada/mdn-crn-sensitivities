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

N_COMPS = 3
MIXTURE = 'NB'

def testing_function(lr, max_rounds, batchsize, n_hidden):
    return hyperparameters_test.test_comb(lr, 
                    max_rounds, 
                    batchsize,
                    n_hidden,
                    train_data,
                    valid_data,
                    NUM_PARAMS,
                    N_COMPS,
                    MIXTURE)

n_rounds = [300, 500, 700]
lrs = [0.1, 0.005, 0.001]
batchsizes = [32, 64, 128]
n_hidden = [128, 256, 512]

if __name__ == '__main__':
    hyperparameters_tuning.test_multiple_combs(testing_function, lrs, n_rounds, batchsizes, n_hidden, 'CRN5_optimisation')