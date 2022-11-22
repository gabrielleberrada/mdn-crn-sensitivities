import neuralnetwork
import torch
from typing import Tuple

def test_comb(lr: float, 
            max_rounds: int, 
            batchsize: int, 
            n_hidden: int,
            train_data: Tuple[torch.tensor], 
            valid_data: Tuple[torch.tensor],
            n_params: int, 
            n_comps: int, 
            mixture='NB') -> Tuple[float, float, float, list]:
    """Trains a Mixture Density Network on the input data with the given combination of hyperparameters.

    Args:
        - :math:`l_r` (float): Initial learning rate.
        - **max_rounds** (int): Maximal number of training rounds.
        - **batchsize** (int): Number of elements in a batch.
        - **n_hidden** (int): Number of neurons in the hidden layer.
        - **train_data** (Tuple[torch.tensor]): Training dataset.
        - **valid_data** (Tuple[torch.tensor]): Validation dataset.
        - **n_params** (int): Number of CRN parameters in input, excluding time parameter.
        - **n_comps** (int): Number of components of the output mixture.
        - **mixture** (str, optional): Type of mixture to compute. Defaults to 'NB' for a Negative Binomial Mixture. \
            Can also be 'Poisson' for a Poisson mixture.

    Returns:
        - Losses for the training and validation datasets and a list of the hyperparameters used for the training.
    """    
    model = neuralnetwork.NeuralNetwork(n_comps=n_comps, n_params=n_params, n_hidden=n_hidden, print_info=False, mixture=mixture)
    # converts np.int64 in int
    batchsize = int(batchsize)
    train_losses, valid_losses = neuralnetwork.train_NN(model, 
                                                        train_data, 
                                                        valid_data, 
                                                        loss=neuralnetwork.loss_kldivergence, 
                                                        max_rounds=max_rounds, 
                                                        lr=lr, 
                                                        batchsize=batchsize,
                                                        print_results=False, 
                                                        print_info=False)
    train_loss = train_losses[-1]
    valid_loss = valid_losses[-1]
    return [train_loss, valid_loss], [lr, max_rounds, batchsize, n_hidden]
