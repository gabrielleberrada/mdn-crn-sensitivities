import torch
import neuralnetwork
import numpy as np
from typing import Union, Tuple, Callable

def probabilities(inputs: torch.tensor,
                model: neuralnetwork.NeuralNetwork,
                length_output: int =200) -> torch.tensor:
    """Computes the probability mass functions for the `length_output` first elements.
    Output has shape (`length_output`).

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
    r"""Computes the gradient of the probability mass functions with respect to the time and to the input parameters.
    Output has shape (length_output, :math:`1 + M_{\text{tot}}`).

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
    r"""Computes the expectation of the probability mass function estimated by the MDN:

    .. math::

        E_{\theta,\xi}[X_t] = \sum_{k=1}^{\text{length_output}} k \ p(k;t,\theta,\xi)

    It can also compute :math:`\mathcal{L}\big(E_{\theta,\xi}[X_t]\big)` where :math:`\mathcal{L}`
    is a given function.

    Args:
        - **inputs** (torch.tensor): Input data in the form requested by the MDN model: 
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{q_1+q_2}, \xi_2^1, ..., \xi_L^{q_1+q_2}]`.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **loss** (Callable, optional): Loss function. Must be compatible with PyTorch. Defaults to the `identity` function.
        - **length_output** (int, optional): Upper bound of the truncated expectation. Defaults to :math:`200`.
        - **array** (bool, optional): If True, the output is a NumPy array. If False, it is a PyTorch tensor. Defaults to True.
    """
    expec = probabilities(inputs, model, length_output)[:,0] * torch.arange(length_output)
    if array:
        return loss(expec.sum()).detach().numpy() # shape 1, output in numpy
    return loss(expec.sum()) # shape 1, output in pytorch

def gradient_expected_val(inputs: torch.tensor, 
                        model: neuralnetwork.NeuralNetwork, 
                        loss: Callable =identity, 
                        length_output: int =200) -> np.ndarray:
    r"""Computes the gradient of the expectation estimated by the MDN:

    .. math::

        \nabla_{t, \theta, \xi} E_{\theta, \xi}[X_t] = \sum_{k=1}^{\text{length_output}} k \ \nabla_{t, \theta, \xi}p(k;t,\theta,\xi)

    It can also compute :math:`\nabla_{t, \theta, \xi} \mathcal{L}\big(E_{\theta,\xi}[X_t]\big)` where :math:`\mathcal{L}` is a given function.
    Output has shape :math:`(1 + M_{\text{tot}})`.

    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **loss** (Callable, optional): Loss function. Must be compatible with PyTorch. Defaults to `identity`.
        - **length_output** (int, optional): Upper bound of the truncated expectation. Defaults to :math:`200`.
    """
    def expec(inputs):
        return expected_val(inputs, model, loss, length_output, array=False)
    gradient =  torch.squeeze(torch.autograd.functional.jacobian(expec, inputs))
    return gradient.detach().numpy()

