import torch
import neuralnetwork
from typing import Union, Tuple, Callable

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

def expected_val(inputs: torch.tensor, model: neuralnetwork.NeuralNetwork, length_output: int =200) -> torch.tensor:
    """Computes the expected value of the sensitivities of mass functions with respect to the time and to the input parameters.
    Args:
        - **inputs** (torch.tensor): Input data.
        - **model** (neuralnetwork.NeuralNetwork): Mixture Density Network model.
        - **length_output** (int, optional): Length of the output. Defaults to :math:`200`.
    Returns:
        - A tensor whose elements are the expected value of the sensitivities of probabilities.
    """    
    stv = sensitivities(inputs, model, length_output)
    expec = stv.permute(1,0) * torch.arange(length_output)
    return expec.sum(dim=1)

def identity(x):
    return x

def extended_expected_val(inputs: torch.tensor, model: neuralnetwork.NeuralNetwork, f: Callable =identity, length_output: int =200) -> torch.tensor:
    stv = sensitivities(inputs, model, length_output) # shape (length_output, n_params) (200, 6)
    expec = stv.permute(1, 0) * f(torch.arange(length_output))
    return expec.sum(dim=1)

# def expectation_gradient(inputs: torch.tensor, model: neuralnetwork.NeuralNetwork, f: Callable =identity, length_output: int =200) -> torch.tensor:
#     def expec(inputs):
#         probs = probabilities(inputs, model, length_output)[:,0]
#         return probs * f(torch.arange(length_output))
#     gradient =  torch.squeeze(torch.autograd.functional.jacobian(expec, inputs))
#     return gradient.sum(dim=0)


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






