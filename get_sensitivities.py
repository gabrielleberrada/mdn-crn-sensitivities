import torch
import neuralnetwork
from typing import Union, Tuple

def probabilities(inputs: torch.tensor,
                model: neuralnetwork.NeuralNetwork,
                length_output: int =200) -> torch.tensor:
    mat_k = torch.arange(length_output).repeat(1, model.n_comps, 1).permute([2, 0, 1])
    return neuralnetwork.mix_pdf(model, inputs, mat_k)

def sensitivities(inputs: torch.tensor, 
                model: neuralnetwork.NeuralNetwork, 
                length_output: int =200, 
                with_probs: bool =False) -> Union[torch.tensor, Tuple[torch.tensor]]:
    """Computes the sensitivities of probability mass functions with respect to the time and to the input parameters.

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

def expected_val(inputs: torch.tensor, model: neuralnetwork.NeuralNetwork, length_output=142) -> torch.tensor:
    """Computes the expected value of the sensitivities of probabilities with respect to the time and to the input parameters.
    Args:
        - **input** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Model to use.
        - **length_output** (int, optional): Length of the output. Defaults to 142.
    Returns:
        - A tensor whose elements are the expected value of the sensitivities of probabilities.
    """    
    sensitivity = sensitivities(inputs, model, length_output)
    expec = sensitivity.permute(1,0) * torch.arange(length_output)
    return expec.sum(dim=1)







