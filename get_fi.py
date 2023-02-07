import numpy as np

def fisher_information_t(probs: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information of a marginal probability mass function at a single time point.

    In the case of a finite state-space, we can enumerate its elements as :math:`\{ x_t^1, x_t^2, ..., x_t^{N_{\max}} \}`. We then have:

    .. math::
        \forall (i,j) \in [\![1, M_{\text{tot}}]\!]^2, 
        [\mathcal{I}_t^{\theta,\xi}]_{ij} = \sum_{\ell=1}^{N_{\max}} \frac{1}{p_\ell(t,\theta,\xi)}[S_t^{\theta,\xi}]_{\ell i}[S_t^{\theta,\xi}]_{\ell j}

    Args:
        - **probs** (np.ndarray): The probability vector. Has shape :math:`(N_{\max},)`.
        - **sensitivities** (np.ndarray): The sensitivity matrix. Has shape :math:`(N_{\max}, M_{\text{tot}})`.

    Returns:
        - The Fisher Information estimated by the model at a single time point. Has shape :math:`(M_{\text{tot}}, M_{\text{tot}})`.
    """
    inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
    pS = np.zeros_like(sensitivities)
    for l, pl in enumerate(inversed_p):
        pS[l,:] = pl * sensitivities[l,:]
    return np.matmul(pS.T, sensitivities)


def fisher_information(n_sampling_times: int, probs: np.ndarray, sensitivities:np.ndarray) -> np.ndarray:
    r"""Computes the Fisher Information Matrix at different time points.

    As defined in :cite:`fox2019fspfim`:

    .. math::
        \forall (i,j) \in [\![1, M_{\text{tot}}]\!]^2, 
        [\mathcal{I}_t^\theta]_{ij} = \sum_{k=1}^L [\mathcal{I}_{t_k}^{\theta,\xi}]_{ij} \sum_{\ell=1}^{N_{\max}} 
        \frac{1}{p_\ell(t_k,\theta,\xi)}[S_{t_k}^{\theta,\xi}]_{\ell i}[S_{t_k}^{\theta,\xi}]_{\ell j}

    Args:
        - **n_sampling_times** (int): Number of sampling times.
        - **probs** (np.ndarray): The probability vector. Has shape (n_sampling_times, :math:`N_{\max}`).
        - **sensitivities** (np.ndarray): The sensitivity of the likelihood matrix. Has shape (n_sampling_times, :math:`N_{\max}`, :math:`M_{\text{tot}}`).

    Returns:
        - The Fisher Information estimated by the model. Has shape :math:`(M_{\text{tot}}, M_{\text{tot}})`.
    """
    f_inf = np.zeros((sensitivities.shape[-1], sensitivities.shape[-1]))
    for t in range(n_sampling_times):
        f_inf += fisher_information_t(probs[t,:], sensitivities[t,:,:])
    return f_inf
