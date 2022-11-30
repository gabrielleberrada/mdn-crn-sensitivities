import numpy as np

def fisher_information_t(probs: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information at a single time point.

    In the case of a finite state-space, we can enumerate its elements as :math:`\{ x_t^1, x_t^2, ..., x_t^{N_{\max}} \}`. We then have:

    .. math::
        \forall (i,j) \in [\![1, M]\!]^2, 
        [\mathcal{I}_t^\theta]_{ij} = \sum_{\ell=1}^{N_{\max}} \frac{1}{p_\ell(t,\theta)}[S_t^\theta]_{\ell i}[S_t^\theta]_{\ell j}

    Args:
        - **probs** (np.ndarray): The probability vector of shape :math:`(N_{\max},)`.
        - **sensitivities** (np.ndarray): The sensitivities of probability matrix of shape :math:`(N_{\max}, M)`

    Returns:
        - The Fisher Information estimated by the model at a single time point. It has shape :math:`(M,M)`.
    """
    # probs[probs<1e-15]=0
    inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
    pS = np.zeros_like(sensitivities)
    for l, pl in enumerate(inversed_p):
        pS[l,:] = pl * sensitivities[l,:]
    return np.matmul(pS.T, sensitivities)

# def fisher_information_t2(input: torch.tensor, model: neuralnetwork.NeuralNetwork, length_output: int =200) -> np.ndarray:
#     def f(input):
#         output = torch.sqrt(get_sensitivities.probabilities(input, model, length_output))
#         return 2*output[torch.nonzero(output)[:-1,0],:]
#     root_stv = torch.autograd.functional.jacobian(f, input)[:, :, 1:]
#     print(root_stv.dtype)
#     # replaces nans with 0
#     root_stv = torch.nan_to_num(root_stv).detach().numpy()
#     root_stv2 = np.transpose(root_stv, [0, 2, 1])
#     res = np.matmul(root_stv2, root_stv)
#     return np.sum(res, axis=0)

# def fisher_information_t3(input: torch.tensor, model: neuralnetwork.NeuralNetwork, length_output: int =200) -> np.ndarray:
#     res = torch.zeros((len(input)-1, len(input)-1))
#     probs = get_sensitivities.probabilities(input, model, length_output)
#     for x in range(length_output):
#         def fx(params):
#             output = torch.log(get_sensitivities.probabilities(params, model, length_output))
#             return output[x, 0]
#         print(fx(input))
#         hessian = torch.autograd.functional.hessian(fx, input)[1:, 1:]
#         print(hessian)
#         res += hessian*probs[x, 0]
#     return - res



def fisher_information(ntime_samples: int, probs: np.ndarray, sensitivities:np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information Matrix at different time points.

    As defined in :cite:`fox2019fspfim`:

    .. math::
        \forall (i,j) \in [\![1, M]\!]^2, 
        [\mathcal{I}_t^\theta]_{ij} = \sum_{k=1}^{N_t} [\mathcal{I}_{t_k}^\theta]_{ij} \sum_{\ell=1}^{N_{\max}} \frac{1}{p_\ell(t_k,\theta)}[S_{t_k}^\theta]_{\ell i}[S_{t_k}^\theta]_{\ell j}

    Args:
        - **ntime_samples** (int): Number of time samples :math:`N_t`.
        - **probs** (np.ndarray): The probability vector of shape :math:`(N_t, N_{\max})`.
        - **sensitivities** (np.ndarray): The sensitivities of probability matrix of shape :math:`(N_t, N_{\max}, M)`.

    Returns:
        - The Fisher Information estimated by the model. It has shape :math:`(M, M)`.
    """
    f_inf = np.zeros((sensitivities.shape[-1], sensitivities.shape[-1]))
    for t in range(ntime_samples):
        f_inf += fisher_information_t(probs[t,:], sensitivities[t,:,:])
    return f_inf
