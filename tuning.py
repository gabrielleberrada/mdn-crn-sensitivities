import convert_csv
import hyperparameters_test
import hyperparameters_tuning

FILE_NAME = 'CRN4_bursting_gene'
CRN_NAME = 'bursting_gene'
NUM_PARAMS = 4

# loading data
X_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_train1.csv')
y_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_train1.csv')
X_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_valid1.csv')
y_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_valid1.csv')
X_test = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_test.csv')
y_test = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_test.csv')

train_data = [X_train1, y_train1]
valid_data = [X_valid1, y_valid1]
test_data = [X_test, y_test]

N_COMPS = 3
MIXTURE = 'NB'

def testing_function(lr, max_rounds, batchsize, n_hidden):
    return hyperparameters_test.test_comb(lr, 
                    max_rounds, 
                    batchsize,
                    n_hidden,
                    train_data,
                    valid_data,
                    test_data,
                    NUM_PARAMS,
                    N_COMPS,
                    MIXTURE)


n_rounds = [300, 500, 700]
lrs = [0.01, 0.005, 0.001]
batchsizes = [32, 64, 128]
n_hidden = [64, 128, 256]

if __name__ == '__main__':
    hyperparameters_tuning.test_multiple_combs(testing_function, lrs, n_rounds, batchsizes, n_hidden, 'CRN4_optimisation_newdata')