import numpy as np

def fisher_information_t(probs: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information at a single time point.

    As defined in :cite:`fox2019fspfim`:

    .. math::
        \mathcal{I}(\theta)_{ij} = \sum_{l=1}^N \frac{1}{p(x_l;\theta)}S_{li}S_{lj}

    Args:
        - **probs** (np.ndarray): The probability vector of dimension :math:`(N,)`.
        - **sensitivities** (np.ndarray): The sensitivities of probabilities matrix of dimensions :math:`(N, N_{\theta})`, where :math:`N` is the number of species and:math:`N_{\theta}` is the number of parameters.

    Returns:
        - The Fisher Information estimated by the model at a single time point. It has dimensions :math:`(N_{\theta}, N_{\theta})`.
    """
    inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
    pS = np.zeros_like(sensitivities)
    for l, pl in enumerate(inversed_p):
        pS[l,:] = pl * sensitivities[l,:]
    return np.matmul(pS.T, sensitivities)


def fisher_information(ntime_samples: int, probs: np.ndarray, sensitivities:np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information Matrix at different time points.

    As defined in :cite:`fox2019fspfim`:

    .. math::
        \mathcal{I}(\theta)_{ij} = \sum_{k=1}^{N_t} \sum_{l=1}^N \frac{1}{p(x_l;t_k,\theta)}S_{li}(t_k)S_{lk}(t_k)

    Args:
        - **ntime_samples** (int): Number of time samples :math:`N_t`.
        - **probs** (np.ndarray): The probability vector of dimension :math:`(N_t, N)`.
        - **sensitivities** (np.ndarray): The sensitivities of probabilities matrix of dimensions :math:`(N_t, N, N_{\theta})`, where :math:`N` is the number of species and:math:`N_{\theta}` is the number of parameters.

    Returns:
        - The Fisher Information estimated by the model for different time points. It has dimensions :math:`(N_{\theta}, N_{\theta})`.
    """
    f_inf = np.zeros((sensitivities.shape[-1], sensitivities.shape[-1]))
    for t in range(ntime_samples):
        f_inf += fisher_information_t(probs[t,:], sensitivities[t,:,:])
    return f_inf
