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
            early_stopping: Tuple[bool, int, float],
            mixture: str ='NB',
            n_models: int =2) -> Tuple[list]:
    r"""Trains a Mixture Density Network on the input data with the given combination of hyperparameters.

    Args:
        - :math:`l_r` (float): Initial learning rate.
        - **max_rounds** (int): Maximal number of training rounds.
        - **batchsize** (int): Number of elements in a batch.
        - **n_hidden** (int): Number of neurons in the hidden layer.
        - **train_data** (Tuple[torch.tensor]): Training dataset.
        - **valid_data** (Tuple[torch.tensor]): Validation dataset.
        - **n_params** (int): Number of CRN parameters in input, excluding time parameter :math:`M_{\text{tot}}`.
        - **n_comps** (int): Number of components of the output mixture.
        - **early_stopping** (Tuple[bool, int, float]):  
        
            - (bool): If True, uses the early stopping regularization. Defaults to False.            
            - **patience** (int): Patience level :math:`n_p`.
              At epoch :math:`n`, the :math:`(n-n_p)` -th epoch is compared pairwise with that 
              of the last :math:`n_p` epochs. Defaults to :math:`50`.
            - **delta** (float): Tolerance threshold :math:`\delta`. Training is stopped if the decrease between 
              the elements of one of those pairs is lower than :math:`\delta`. Defaults to :math:`10^{-6}`.
        - **mixture** (str, optional): Type of mixture to compute. Defaults to "NB" for a Negative Binomial Mixture.
          Can also be "Poisson" for a Poisson mixture.
        - **n_models** (int): Number of models to train. The return losses are the mean of the losses computed for each model.

    Returns:
        - Losses for the training and validation datasets and a list of the hyperparameters used for the training.
    """
    train_loss = 0
    valid_loss = 0    
    # converts np.int64 in int
    batchsize = int(batchsize)
    for _ in range(n_models):
        model = neuralnetwork.NeuralNetwork(n_comps=n_comps, n_params=n_params, n_hidden=n_hidden, print_info=False, mixture=mixture)
        train_losses, valid_losses = neuralnetwork.train_NN(model, 
                                                            train_data, 
                                                            valid_data, 
                                                            loss=neuralnetwork.loss_kldivergence, 
                                                            max_rounds=max_rounds, 
                                                            lr=lr, 
                                                            batchsize=batchsize,
                                                            add_early_stopping = early_stopping,
                                                            print_results=False, 
                                                            print_info=False)
        train_loss += train_losses[-1]
        valid_loss += valid_losses[-1]
    if early_stopping[0]:
        return [train_loss/n_models, valid_loss/n_models], [lr, max_rounds, batchsize, n_hidden, early_stopping[1], early_stopping[2]]
    else:
        return [train_loss/n_models, valid_loss/n_models], [lr, max_rounds, batchsize, n_hidden]
