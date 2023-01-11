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
        - **loss** (Callable, optional): Loss function. Must be compatible with PyTorch. Defaults to identity.
        - **length_output** (int, optional): _description_. Defaults to 200.
    """    
    def expec(inputs):
        return expected_val(inputs, model, loss, length_output, array=False)
    gradient =  torch.squeeze(torch.autograd.functional.jacobian(expec, inputs))
    return gradient.detach().numpy()


if __name__ == '__main__':

    from CRN3_control import propensities_explosive_production as propensities
    import save_load_MDN
    import convert_csv
    import numpy as np
    
    model1 = save_load_MDN.load_MDN_model('CRN3_control/saved_models/CRN3_model1.pt')
    # model2 = save_load_MDN.load_MDN_model('CRN3_control/saved_models/CRN3_model2.pt')
    # model3 = save_load_MDN.load_MDN_model('CRN3_control/saved_models/CRN3_model3.pt')
    X_test = convert_csv.csv_to_tensor(f'CRN3_control/data/X_CRN3_control_test.csv')
    # y_test = convert_csv.csv_to_tensor(f'CRN3_control/data/y_CRN3_control_test.csv')

#     def expect_theta(t, params, init_state=5):
#         theta = params[0]+params[1]
#         return t*init_state*(np.exp(t*theta) - init_state)

#     print('exact\n', expect_theta(X_test[1_000, 0].numpy(), X_test[1_000, 1:].numpy()))
#     print('expectation of gradient\n', expectation_gradient(X_test[1_000,:], model1))
#     print(expectation_gradient(X_test[1_000,:], model2))
#     print(expectation_gradient(X_test[1_000,:], model3))

    
    # import time

    # time1 = 0
    # time2 = 0
    # for _ in range(1_000):
    #     start = time.time()
    #     extended_expected_val(X_test[1_000,:], model1)
    #     end = time.time()
    #     time1 += start - end
    #     start = time.time()
    #     expectation_gradient(X_test[1_000,:], model1)
    #     end = time.time()
    #     time2 += start - end
    # print("1 - Expectation of gradient", time1)
    # print("2 - Gradient of expectation", time2)






