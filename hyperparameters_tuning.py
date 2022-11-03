import torch
import numpy as np
import convert_csv
import neuralnetwork
import concurrent.futures
import hyperparameters_test
from tqdm import tqdm
from typing import Tuple

def test_multiple_combs(lrs: list[float], 
                        max_rounds: list[int], 
                        batchsizes: list[int], 
                        n_hidden: int,
                        n_comps: int,
                        n_params: int,
                        train_data: Tuple[torch.tensor],
                        valid_data: Tuple[torch.tensor],
                        test_data: Tuple[torch.tensor],
                        mixture: str, 
                        file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tests all combinations of the given hyperparameters.

    Args:
        - **lrs** (list[float]): List of learning rates to try.
        - **max_rounds** (list[int]): List of maximal number of training rounds to try.
        - **batchsizes** (list[int]): List of batchsizes to try.
        - **n_hidden** (int): List of number of hidden layer neurons to try.
        - **n_comps** (int): Number of components of the mixture.
        - **n_params** (int): Number of CRN parameters.
        - **train_data** (Tuple[torch.tensor]): Training dataset.
        - **valid_data** (Tuple[torch.tensor]): Validation dataset.
        - **test_data** (Tuple[torch.tensor]): Testing dataset.
        - **mixture** (str): Type of mixture to compute.
        - **file_name** (str): Name of the CSV file in which the results will be saved. Each line of the CSV file corresponds to \
            one hyperparameter combinations. The first 4 elements of each line are the chosen learning rate, number of iterations, batchsizes, number of hidden layer neurons. \
            The next and last 3 elements of each line are the training, validation and testing loss for these hyperparameters.

    Returns:
        - Array of the training, validation and testing losses for each combination of hyperparameters. Shape :math:`(N_{comb}, 3)`
        - Array of tested combinations of hyperparameters. Shape :math:`(N_{comb}, 4)`.
    """                        
    # all combinations
    comb = np.meshgrid(lrs, max_rounds, batchsizes, n_hidden)
    length = len(lrs) * len(max_rounds) * len(batchsizes) * len(n_hidden)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = list(tqdm(executor.map(hyperparameters_test.test_comb,
                                     np.reshape(comb[0], length), 
                                     np.reshape(comb[1], length), 
                                     np.reshape(comb[2], length), 
                                     np.reshape(comb[3], length)), total=length, desc='Computing..'))
    losses = np.zeros((length, 3))
    parameters = np.zeros((length, 4))
    csv = np.zeros((length, 7))
    for i, (train_loss, valid_loss, test_loss, params) in enumerate(res):
        losses[i,:] = [train_loss, valid_loss, test_loss]
        parameters[i,:] = params
        csv[i,:] = params + [train_loss, valid_loss, test_loss]
    convert_csv.array_to_csv(csv, file_name)
    return losses, parameters