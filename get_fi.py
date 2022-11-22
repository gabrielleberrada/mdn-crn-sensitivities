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
        - The Fisher Information estimated by the model at different time points. It has shape :math:`(M, M)`.
    """
    f_inf = np.zeros((sensitivities.shape[-1], sensitivities.shape[-1]))
    for t in range(ntime_samples):
        f_inf += fisher_information_t(probs[t,:], sensitivities[t,:,:])
    return f_inf
