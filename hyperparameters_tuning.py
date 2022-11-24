import numpy as np
import convert_csv
import concurrent.futures
from tqdm import tqdm
from typing import Tuple, Callable

def test_multiple_combs(testing_function: Callable,
                        lrs: list[float], 
                        max_rounds: list[int], 
                        batchsizes: list[int], 
                        n_hidden: list[int],
                        early_stopping: Tuple[bool, list[int], list[float]],
                        file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""Tests all combinations of the given hyperparameters.

    Args:
        - **lrs** (list[float]): List of learning rates to try.
        - **max_rounds** (list[int]): List of maximal number of training rounds to try.
        - **batchsizes** (list[int]): List of batchsizes to try.
        - **n_hidden** (list[int]): List of number of hidden layer neurons to try.
        - **early_stopping** (Tuple[bool, list[int], list[float]]): 
        
            - (bool): If True, use the early stopping regularization. Defaults to False.            
            - **patience** (list[int]): List of patience levels.
              At epoch :math:`n`, the :math:`(n-n_p)` -th epoch is compared pairwise with that
              of the last :math:`n_p` epochs. Defaults to :math:`50`.
            - **delta** (list[float]): List of tolerance thresholds. Training is stopped if the decrease between
              the elements of one of those pairs is lower than :math:`\delta`. Defaults to :math:`10^{-6}`.
        - **file_name** (str): Name of the CSV file in which the results will be saved. Each line of the CSV file corresponds to
          one hyperparameter combination. The first 4 elements of each line are the chosen learning rate, number of iterations, 
          batchsizes, number of hidden layer neurons. The next and last 2 elements of each line are the training and validation losses 
          for these hyperparameters.

    Returns:
        - Array of the training and validation losses for each combination of hyperparameters. It has shape 
          :math:`(N_{\text{comb}}, 2)` where :math:`N_{\text{comb}}` is the number of possible hyperparameters combinations.
        - Array of tested combinations of hyperparameters. It has shape :math:`(N_{\text{comb}}, n_{\text{params}})`.
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

