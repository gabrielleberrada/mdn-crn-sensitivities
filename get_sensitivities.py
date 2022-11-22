import torch
import neuralnetwork
from typing import Union, Tuple

def sensitivities(input: torch.tensor, 
                model: neuralnetwork.NeuralNetwork, 
                length_output: int =200, 
                with_probs: bool =False) -> Union[torch.tensor, Tuple[torch.tensor]]:
    """Computes the sensitivities of probability mass functions with respect to the time and to the input parameters.

    Args:
        - **input** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **length_output** (int, optional): Length of the output. Defaults to :math:`200`.
        - **with_probs** (bool, optional): If True, also returns the corresponding probability distribution. Defaults to False.
    """    
    def f(input, length_output=length_output):
        mat_k = torch.arange(length_output).repeat(1,model.n_comps,1).permute([2,0,1])
        return neuralnetwork.mix_pdf(model, input, mat_k)
    if with_probs:
        return torch.squeeze(torch.autograd.functional.jacobian(f, input)), f(input).detach()
    return torch.squeeze(torch.autograd.functional.jacobian(f, input))




