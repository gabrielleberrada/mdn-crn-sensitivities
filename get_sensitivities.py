import torch
import neuralnetwork
import numpy as np
from typing import Union, Tuple, Callable

def probabilities(inputs: torch.tensor,
                model: neuralnetwork.NeuralNetwork,
                length_output: int =200) -> torch.tensor:
    """Computes the probability mass functions for the `length_output` first elements.
    Output has shape (length_output).

    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **length_output** (int, optional): Length of the output. Defaults to :math:`200`.
    """    
    mat_k = torch.arange(length_output).repeat(1, model.n_comps, 1).permute([2, 0, 1])
    return neuralnetwork.mix_pdf(model, inputs, mat_k)

def sensitivities(inputs: torch.tensor, 
                model: neuralnetwork.NeuralNetwork, 
                length_output: int =200, 
                with_probs: bool =False) -> Union[torch.tensor, Tuple[torch.tensor]]:
    """Computes the sensitivities of probability mass functions with respect to the time and to the input parameters.
    Output has shape (length_output, 1 + n_total_params).

    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **length_output** (int, optional): Length of the output. Defaults to :math:`200`.
        - **with_probs** (bool, optional): If True, also returns the corresponding probability distribution. Defaults to False.
    """
    def f(inputs):
        return probabilities(inputs, model, length_output)
    if with_probs:
        return torch.squeeze(torch.autograd.functional.jacobian(f, inputs)), f(inputs).detach()
    return torch.squeeze(torch.autograd.functional.jacobian(f, inputs))

def identity(x):
    return x

def expected_val(inputs: torch.tensor,
                model: neuralnetwork.NeuralNetwork, 
                loss: Callable =identity, 
                length_output: int =200, 
                array: bool =True) -> Union[np.ndarray, torch.tensor]:
    """Computes the value of the loss function evaluated in the expectation of the density. Output is a scalar.

    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **loss** (Callable, optional): Loss function. Must be compatible with PyTorch. Defaults to `identity`.
        - **length_output** (int, optional): Length of the output. Defaults to :math:`200`.
        - **array** (bool, optional):If True, the output is a NumPy array. If False, it is a PyTorch tensor. Defaults to True.
    """
    expec = probabilities(inputs, model, length_output)[:,0] * torch.arange(length_output)
    if array:
        return loss(expec.sum()).detach().numpy() # shape 1, output in numpy
    return loss(expec.sum()) # shape 1, output in pytorch

def gradient_expected_val(inputs: torch.tensor, 
                        model: neuralnetwork.NeuralNetwork, 
                        loss: Callable =identity, 
                        length_output: int =200) -> np.ndarray:
    """Computes the gradient of the loss function evaluated in the expectation of the density. Output has shape (1 + n_total_params).

    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **loss** (Callable, optional): Loss function. Must be compatible with PyTorch. Defaults to `identity`.
        - **length_output** (int, optional): _description_. Defaults to :math:`200`.
    """    
    def expec(inputs):
        return expected_val(inputs, model, loss, length_output, array=False)
    gradient =  torch.squeeze(torch.autograd.functional.jacobian(expec, inputs))
    return gradient.detach().numpy()

