import scipy.sparse as sp
import numpy as np
from scipy.integrate import solve_ivp
from bidict import bidict
import math
import simulation
from typing import Tuple

class StateSpaceEnumeration:
    r"""State space enumeration as presented in :cite:`gupta2017projection`.

    Computes functions :math:`\Phi` and :math:`\Phi^{-1}` to project the state space on the set of integers and conversely.

    Args:
        - :math:`c_r` (int): Value such that :math:`(0, .., 0, c_r)` is the last value in the truncated space.
        - **dim** (int): Dimensions of the initial space.
    """   
    def __init__(self, cr: int, dim: int):     
        # we choose to always start at cl=0
        self.bijection = bidict()
        self.dim = dim
        self.cl = np.zeros(self.dim, dtype=int)
        self.cr = np.zeros(self.dim, dtype=int)
        self.cr[-1] = cr
        self.lb = self.phi(self.cl, self.dim)
        self.ub = self.phi(self.cr, self.dim)
        # number of states included in the truncated state space
        self.length = self.ub - self.lb + 1
    
    def phi(self, x: np.ndarray, n: int) -> int:
        r"""Recurrent function that computes the projection :math:`\Phi: \mathbb{N}^{dim} \rightarrow \mathbb{N}` as defined in :cite:`gupta2017projection`.

        Args:
            - **x** (np.ndarray): Vector in :math:`\mathbb{N}^n`.
            - **n** (int): Dimension of the space.

        Returns:
            - Result :math:`\Phi_n(z)` of the projection in :math:`\mathbb{N}`.
        """        
        if n < 2:
            return x[0]
        if n == 2:
            return int((x[0]+x[1])*(x[0]+x[1]+1)/2 + x[1])
        else:
            return self.phi(np.array([self.phi(x[:-1], n-1), x[-1]]), 2)

    def phi_inverse(self, z: int, n: int) -> Tuple[int]:
        r"""Recurrent function that computes the inversed projection :math:`\Phi^{-1}: \mathbb{N} \rightarrow \mathbb{N}^{dim}` as defined in :cite:`gupta2017projection`.

        Args:
            - **z** (int): Input.
            - **n** (int): Dimension of the space.

        Returns:
            - Result vector :math:`\Phi_n^{-1}(z)` in :math:`\mathbb{N}^n`.
        """        
        if n < 2:
            return (z,)
        elif n == 2:
            v = math.floor((math.sqrt(8*z+1)-1)/2)
            x2 = z - v*(v+1)/2
            return int(v - x2), int(x2)
        else:
            z1, z2 = self.phi_inverse(z, 2)
            z11 = self.phi_inverse(z1, n-1)
            return z11 + (z2,)
    
    def create_bijection(self):
        r"""Saves the bijection values in a dictionary.
        """        
        self.bijection[self.lb] = tuple(self.cl)
        self.bijection[self.ub] = tuple(self.cr)
        for z in range(self.lb, self.ub):
            self.bijection[z] = self.phi_inverse(z, self.dim)
        # print(self.bijection)


class SensitivitiesDerivation:
    r"""Class to compute the probability sensitivities and probabilities with the FSP method.
    Based on :cite:`fox2019fspfim`.
    
    Args:
        - **crn** (simulation.CRN): the CRN to study.
        - :math:`c_r` (int, optional): value such that :math:`(0, .., 0, c_r)` is the last value in the truncated space.. Defaults to 4.  
    """
    def __init__(self, crn: simulation.CRN, cr: int =4):     
        self.cr = cr
        self.crn = crn
        self.n_params = crn.n_params
        self.n_reactions = crn.n_reactions
        self.n_species = crn.n_species
        self.bijection = StateSpaceEnumeration(cr, dim=self.n_species)
        self.bijection.create_bijection()
        self.entries = self.bijection.bijection.values()
        self.n_states = len(self.entries)

    def create_B(self, index: int) -> np.ndarray:
        r"""Computes the :math:`B_i` matrix for the parameter n°index as defined in :cite:`fox2019fspfim`.

        Args:
            - **index** (int): index of the reaction occuring.

        Returns:
            - The rate matrix :math:`B_i` for the reaction n°index over the truncated state-space.
        """
        d = self.bijection.bijection.inverse
        n = self.n_states
        stoich_mat = self.crn.stoichiometric_mat[:,index]
        propensity = self.crn.propensities[index]
        # might contain negative elements
        outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
        # propensity parameter is 1
        data = np.array(list(map(lambda x: propensity(np.ones(self.n_params), x), self.entries)))
        #
        rows = np.array([d[entry] for entry in self.entries])
        get_index = lambda key: d[key] if key in d else -1
        # rows = np.array([get_index(entry) for entry in self.entries])
        #
        columns = np.array([get_index(output) for output in outputs])
        # columns = np.array([self.bijection.phi(output, self.n_species) for output in outputs])
        compute_diags = np.vectorize(lambda i: -data[rows==i].sum())
        diags = compute_diags(np.arange(n))
        # truncation
        mask = (columns >= 0) & (columns < n)
        rows = rows[mask]
        columns = columns[mask]
        data = data[mask]
        # according to the paper from Fox and Munsky
        B = sp.coo_matrix((data, (columns, rows)), shape=(n, n))
        B.setdiag(diags)
        B.eliminate_zeros()
        return B

    def create_A(self, params: np.ndarray) -> np.ndarray:
        r"""Computes the matrix **A** as defined in :cite:`fox2019fspfim`.

        Args:
            - **params** (np.ndarray): Propensity parameters.

        Returns:
            - The rate matrix **A** over the truncated state-space.
        """        
        # creates the rate matrix for each reaction
        create_Bs = np.vectorize(lambda i: self.create_B(i))
        #
        Bs = create_Bs(np.arange(self.n_reactions))
        # Bs = create_Bs(np.arange(self.n_params))
        return (Bs*params).sum()

    def constant_matrix(self, params: np.ndarray, index: int) -> np.ndarray:
        r"""Computes the matrix **C** as defined in :cite:`fox2019fspfim`.

        Args:
            - **params** (np.ndarray): Propensity parameters.
            - **index** (int): Index of the reaction occuring.

        Returns:
            - The constant matrix **C** in the ODEs as presented in equation (34) of :cite:`fox2019fspfim`.
        """        
        A = self.create_A(params)
        B = self.create_B(index)
        empty = sp.coo_matrix(A.shape)
        up = sp.hstack((A, empty))
        bottom = sp.hstack((B, A))
        return sp.vstack((up, bottom))

    def solve_ode(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, index: int, t_eval: list[float]):
        r"""Solves the set of linear ODEs (34) from :cite:`fox2019fspfim`.

        .. math::
            \frac{d}{dt}\begin{pmatrix} p(t) \\ S_i(t) \end{pmatrix} = \begin{pmatrix} A & 0 \\ B & A \end{pmatrix}
            \begin{pmatrix} p(t) \\ S_i(t) \end{pmatrix}

        Args:
            - **init_state** (np.ndarray): Initial state for probabilities and sensitivities. The length of the initial state must be 

            .. math::
                2(\frac{Cr(Cr+3)}{2}+1) \text{ if } n \leq 2 \text{, else } 2(Cr+1)

            - :math:`t_0` (float): Starting time.
            - :math:`t_f` (float): Final time.
            - **params** (np.ndarray): Propensity parameters.
            - **index** (int): Index of the reaction occuring.
            - :math:`t_{eval}` (list[float]): Times at which to store the computed solution

        Returns:
            - Bunch object as the output of the ``solve_ivp`` function applied to the set of linear ODEs.
        """        
        constant = sp.csr_matrix(self.constant_matrix(params, index))
        def f(t, x):
            return constant.dot(x)
        return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)


    def get_sensitivities(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, t_eval: list[float]) -> Tuple[np.ndarray]:
        """Computes probabilities and sensitivities with respect to each parameter of the CRN using FSP methods.

        Args:
            - **init_state** (np.ndarray): Array of dimensions :math:`(N_\\theta, 2N)` such that \
                            init_state[i,:] is the initial states for the probabilities and sensitivities of the i-th reaction.\
                            The length of each initial state vector must be :math:`2(\\frac{Cr(Cr+3)}{2}+1)` if :math:`n \\leq 2`, else :math:`2(Cr+1)`.
            - :math:`t_0` (float): Starting time.
            - :math:`t_f` (float): Final time.
            - **params** (np.ndarray): Propensity parameters.
            - :math:`t_{eval}` (list[float]): Times at which to store the computed solution.

        Returns:
            - The probabilities vector and probability sensitivities matrix for each time points.
        """
        sensitivities = []
        for i in range(self.n_reactions):
            solution = self.solve_ode(init_state[i,:], t0, tf, params, i, t_eval)['y']
            # print(solution[:self.n_states,:])
            sensitivities.append(solution[self.n_states:,:].T)
        # print(solution[:self.n_states,:][3], solution[:self.n_states, :][4], solution[:self.n_states,:][5])
        probs = solution[:self.n_states,:].T
        return probs, np.stack(sensitivities, axis=-1)




# Correction get_sensitivities

stoich_mat = np.array([[-2, 2], 
                        [2, -2],
                        [1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]]).T

def lambda1(params, x):
    return params[0]*x[0]*(x[0]-1)

def lambda2(params, x):
    return params[1]*x[1]*(x[1]-1)

def lambda3(params, x):
    return params[2]

def lambda4(params, x):
    return params[3]*x[0]

def lambda5(params, x):
    return params[4]

def lambda6(params, x):
    return params[5]*x[1]

# propensities = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6])
# crn = simulation.CRN(stoich_mat, propensities, 6)
# cr = 50
# stv_calculator = SensitivitiesDerivation(crn, cr)
# n_cr = int(cr*(cr+3)/2+1)
# init_state_p = np.zeros(2*n_cr)
# init_state_p[0] = 1
# init_state = np.stack([init_state_p]*crn.n_reactions)
# t = 0.1
# params = np.arange(6)
# probs, stv = stv_calculator.get_sensitivities(init_state, 0., t, params, t_eval=[t])
# f = get_fi.fisher_information_t(probs[0,:], stv[0,:,:])
# print(f)
# print(probs[0,:12])
# print(stv_calculator.bijection.bijection)