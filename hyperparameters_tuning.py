import numpy as np
import convert_csv
import concurrent.futures
from tqdm import tqdm
from typing import Tuple, Callable

def test_multiple_combs(testing_function: Callable,
                        lrs: list, 
                        max_rounds: list, 
                        batchsizes: list, 
                        n_hidden: list,
                        early_stopping: Tuple[bool, list, list],
                        file_name: str) -> Tuple[np.ndarray]:
    r"""Tests all combinations of the given hyperparameters.

    Args:
        - **lrs** (list): List of learning rates to test.
        - **max_rounds** (list): List of maximal number of training rounds to test.
        - **batchsizes** (list): List of batchsizes to test.
        - **n_hidden** (list): List of number of hidden layer neurons to test.
        - **early_stopping**: 
        
            - (bool): If True, use the early stopping regularization. Defaults to False.            
            - **patience** (list): List of patience levels :math:`n_p`.
              At epoch :math:`n`, the :math:`(n-n_p)` -th epoch is compared pairwise with that
              of the last :math:`n_p` epochs.
            - **delta** (list): List of tolerance thresholds :math:`\delta`. Training is stopped if the decrease between
              the elements of one of those pairs is lower than :math:`\delta`.
        - **file_name** (str): Name of the CSV file under which to save the results. Each line of the CSV file corresponds to
          one hyperparameter combination. The first elements of each line are the selected hyperparameters. The next and last 
          2 elements of each line are the training and validation losses for these hyperparameters.

    Returns:
        - Array of the training and validation losses for each combination of hyperparameters. Has shape 
          :math:`(N_{\text{comb}}, 2)` where :math:`N_{\text{comb}}` is the number of possible hyperparameters combinations.
        - Array of tested combinations of hyperparameters. Has shape :math:`(N_{\text{comb}}, M_{\text{tot}})`.
    """                        
    if early_stopping[0]:
      n_params = 6
      # all combinations
      comb = np.meshgrid(lrs, max_rounds, batchsizes, n_hidden, early_stopping[1], early_stopping[2])
      length = len(lrs) * len(max_rounds) * len(batchsizes) * len(n_hidden) * len(early_stopping[1]) * len(early_stopping[2])
      # multiprocessing
      with concurrent.futures.ProcessPoolExecutor() as executor:
          res = list(tqdm(executor.map(testing_function,
                                      np.reshape(comb[0], length), 
                                      np.reshape(comb[1], length), 
                                      np.reshape(comb[2], length), 
                                      np.reshape(comb[3], length),
                                      np.reshape(comb[4], length), 
                                      np.reshape(comb[5], length)), total=length, desc='Computing..'))
    else:
      n_params = 4
      # all combinations
      comb = np.meshgrid(lrs, max_rounds, batchsizes, n_hidden)
      length = len(lrs) * len(max_rounds) * len(batchsizes) * len(n_hidden)
      # multiprocessing
      with concurrent.futures.ProcessPoolExecutor() as executor:
          res = list(tqdm(executor.map(testing_function,
                                      np.reshape(comb[0], length), 
                                      np.reshape(comb[1], length), 
                                      np.reshape(comb[2], length), 
                                      np.reshape(comb[3], length)), total=length, desc='Computing..'))
    losses = np.zeros((length, 2))
    parameters = np.zeros((length, n_params))
    csv = np.zeros((length, 2+n_params))
    for i, (loss, params) in enumerate(res):
        losses[i,:] = loss
        parameters[i,:] = params
        csv[i,:] = params + loss
    # saving results in CSV file
    convert_csv.array_to_csv(csv, file_name)
    return losses, parameters

