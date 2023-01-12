import convert_csv
import hyperparameters_test
import hyperparameters_tuning
from typing import Tuple

FILE_NAME = 'CRN6_toggle_switch/data'
CRN_NAME = 'toggle'
NUM_PARAMS = 13

# loading data
X_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_train1.csv')
y_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_train1.csv')
X_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_valid1.csv')
y_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_valid1.csv')

train_data = [X_train1, y_train1]
valid_data = [X_valid1, y_valid1]

N_COMPS = 4
MIXTURE = 'NB'

def testing_function(lr: list, max_rounds: list, batchsize: list, n_hidden: list) -> Tuple[list]:
    """Computes the hyperparameters testing without the early stopping method.

    Args:
        - **lr** (list): List of learning rates to test.
        - **max_rounds** (list): List of maximum number of rounds to test.
        - **batchsize** (list): List of batchsizes to test.
        - **n_hidden** (list): List of number of neurons in the hidden layer to test.

    Returns:
        - Losses for the training and validation datasets and a list of the hyperparameters used for the training.
    """    
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

def testing_function2(lr: list, max_rounds: list, batchsize: list, n_hidden: list, patience: list, delta: list) -> Tuple[list]:
    """Computes the hyperparameters testing with the early stopping method.

    Args:
        - **lr** (list): List of learning rates to test.
        - **max_rounds** (list): List of maximum number of rounds to test.
        - **batchsize** (list): List of batchsizes to test.
        - **n_hidden** (list): List of number of neurons in the hidden layer to test.
        - **patience** (list): List of patience levels to test.
        - **delta** (list): List of tolerance thresholds to test.

    Returns:
        - Losses for the training and validation datasets and a list of the hyperparameters used for the training.
    """
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



if __name__ == '__main__':

    EARLY_STOPPING = True
    n_rounds = [300, 500, 700]
    lrs = [0.005, 0.001]
    batchsizes = [32, 64]
    n_hidden = [128, 256]
    patience = [20, 40, 60]
    delta = [0, 1e-6, 1e-5]

    if EARLY_STOPPING:
        hyperparameters_tuning.test_multiple_combs(testing_function2, lrs, n_rounds, batchsizes, n_hidden, (True, patience, delta), 'CRN5_optimisation_early_stopping2')
    else:
        hyperparameters_tuning.test_multiple_combs(testing_function, lrs, n_rounds, batchsizes, n_hidden, (False, None, None), 'CRN5_optimisation_early_stopping2')