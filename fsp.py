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
        - :math:`C_r` (int): Value such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
        - **dim** (int): Dimensions of the initial state-space.
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
        self.n_states = self.ub - self.lb + 1
    
    def phi(self, x: np.ndarray, n: int) -> int:
        r"""Recurrent function which computes the projection :math:`\Phi: \mathbb{N}^n \rightarrow \mathbb{N}` 
        as defined in :cite:`gupta2017projection`.

        Args:
            - **x** (np.ndarray): Vector in :math:`\mathbb{N}^n`.
            - **n** (int): Dimension of the space.

        Returns:
            - Result :math:`\Phi_n(x)` of the projection in :math:`\mathbb{N}`.
        """        
        if n < 2:
            return x[0]
        if n == 2:
            return int((x[0]+x[1])*(x[0]+x[1]+1)/2 + x[1])
        else:
            return self.phi(np.array([self.phi(x[:-1], n-1), x[-1]]), 2)

    def phi_inverse(self, z: int, n: int) -> Tuple[int]:
        r"""Recurrent function which computes the inversed projection :math:`\Phi^{-1}: \mathbb{N} \rightarrow \mathbb{N}^n` 
        as defined in :cite:`gupta2017projection`.

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


class SensitivitiesDerivation:
    r"""Class to compute the sensitivities and probabilities with the FSP method.
    Based on :cite:`fox2019fspfim`.
    
    Args:
        - **crn** (simulation.CRN): the CRN to study.
        - :math:`C_r` (int, optional): value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. 
          Defaults to :math:`4`.  
    """
    def __init__(self, crn: simulation.CRN, cr: int =4):     
        self.cr = cr
        self.crn = crn
        self.n_fixed_params = crn.n_fixed_params
        self.n_control_params = crn.n_control_params
        self.n_params = self.n_fixed_params + self.n_control_params
        self.n_reactions = crn.n_reactions
        self.n_species = crn.n_species
        self.bijection = StateSpaceEnumeration(cr, dim=self.n_species)
        self.bijection.create_bijection()
        self.entries = self.bijection.bijection.values()
        self.n_states = len(self.entries)
        self.time = 0
        init_state = np.zeros(2*self.n_states)
        init_state[self.bijection.bijection.inverse[tuple(crn.init_state)]] = 1
        self.init_state = init_state
        self.current_state = self.init_state.copy()
        self.samples = np.empty((0, self.n_species))

    def reset(self):
        self.time = 0
        self.current_state = self.init_state.copy()
        self.sampling_states = np.empty((0, self.n_species))

    def create_B(self, index: int) -> np.ndarray:
        r"""Computes the matrix :math:`B_i` for the **index**-th reaction as defined in :cite:`fox2019fspfim`.

        Args:
            - **index** (int): index of the occuring reaction.

        Returns:
            - Rate matrix :math:`B_i` for the **index**-th reaction over the truncated state-space.
        """
        d = self.bijection.bijection.inverse
        n = self.n_states
        stoich_mat = self.crn.stoichiometry_mat[:,index]
        propensity = self.crn.propensities[index]
        # might contain negative elements
        outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
        # propensity parameter is 1
        data = np.array(list(map(lambda x: propensity(np.ones(self.n_params), x), self.entries)))
        #
        rows = np.array([d[entry] for entry in self.entries])
        get_index = lambda key: d[key] if key in d else -1
        columns = np.array([get_index(output) for output in outputs])
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
            - **params** (np.ndarray): Parameters of the propensity functions.
              Here, we assume that the i-th parameter corresponds to the i-th reaction.

        Returns:
            - The rate matrix **A** over the truncated state-space.
        """        
        # creates the rate matrix for each reaction
        create_Bs = np.vectorize(lambda i: self.create_B(i))
        Bs = create_Bs(np.arange(self.n_reactions))
        return (Bs*params).sum()

    def constant_matrix(self, params: np.ndarray, index: int) -> np.ndarray:
        r"""Computes the matrix **C** as defined in :cite:`fox2019fspfim`.

        Args:
            - **params** (np.ndarray): Parameters of the propensity functions.
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

    def solve_ode(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, index: int, t_eval: list, with_stv: bool =True):
        r"""Solves the set of linear ODEs (34) from :cite:`fox2019fspfim`.

        .. math::
            \\frac{d}{dt}\\begin{pmatrix} p(t) \\\ S_i(t) \end{pmatrix} = \\begin{pmatrix} A & 0 \\\ B & A \\end{pmatrix}
            \\begin{pmatrix} p(t) \\\ S_i(t) \\end{pmatrix}

        Args:
            - **init_state** (np.ndarray): Initial state for probabilities and sensitivities.
              The length of the initial state must be 2 times the number of states, 
              which is stored in the attribute `n_states`.

            .. math::
                2\big(\\frac{Cr(Cr+3)}{2}+1\big) \\text{ if } n = 2 \\text{, else } 2(Cr+1)

            - :math:`t_0` (float): Starting time.
            - :math:`t_f` (float): Final time.
            - **params** (np.ndarray): Parameters of the propensity functions.
            - **index** (int): Index of the occuring reaction.
            - :math:`t_{eval}` (list): Times to save the computed solution.
            - **with_stv** (bool): If True, computes the corresponding sensitivities. If False, only solves the first part 
              of the ODE to compute the probability mass function. Defaults to True.

        Returns:
            - Bunch object as the output of the ``solve_ivp`` function applied to the set of linear ODEs.
              To access the probability and sensitivities distribution, use the key 'y'. The result has shape 
              :math:`(n_{\text{states}}, N_t)` if `with_stv` is False, :math:`(2n_{\text{states}}n, N_t)`else, 
              with :math:`N_t` is the number of sampling times.
        """
        if with_stv:        
            constant = sp.csr_matrix(self.constant_matrix(params, index))
            def f(t, x):
                return constant.dot(x)
            return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)
        A = sp.csr_matrix(self.create_A(params))
        def f(t, x):
            return A.dot(x)
        return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)

    def solve_multiple_odes(self,
                            sampling_times,
                            time_windows,
                            parameters,
                            with_stv=True):
        distributions = []
        for i, t in enumerate(time_windows):
            params = np.concatenate((parameters[i, :self.n_fixed_params], parameters[i, self.n_fixed_params:])) # faux i...
            t_eval = sampling_times[(sampling_times > self.time) & (sampling_times <= t)]
            added_t = None
            if len(t_eval)==0 or t_eval[-1] != t:
                # to get state at time t
                t_eval = np.concatenate((t_eval, [t]))
                added_t = -1
            if with_stv and self.n_fixed_params:
                distribution = []
                # computes sensitivities for all fixed reactions
                for ind_reaction in range(self.n_fixed_params):
                    solution = self.solve_ode(init_state=self.current_state, 
                                            t0=self.time,
                                            tf=t, 
                                            params=params,
                                            index=ind_reaction,
                                            t_eval=t_eval,
                                            with_stv=True)['y']
                    # probabilities
                    if ind_reaction == 0:
                        distribution.append(solution[:self.n_states, :added_t])
                    # sensitivities
                    distribution.append(solution[self.n_states:, :added_t])
                # probabilities
                distribution = np.stack(distribution, axis=-1) # (n_states, N_t, M+1)
                distributions.append(distribution)
            else:
                solution = self.solve_ode(init_state=self.current_state[:self.n_states], 
                                        t0=self.time,
                                        tf=t, 
                                        params=params,
                                        index=0,
                                        t_eval=t_eval,
                                        with_stv=False)['y'] 
                distributions.append(solution[:, :added_t])
            self.current_state = solution[:,-1]
            self.time = t
        # shape (n_states, N_t) or (n_states, N_t, M+1)
        return np.concatenate(distributions, axis=1)
            
            

    def get_sensitivities(self, 
                        init_state: np.ndarray, 
                        t0: float, 
                        tf: float, 
                        params: np.ndarray, 
                        t_eval: list) -> Tuple[np.ndarray]:
        """Computes probabilities and sensitivities with respect to each parameter of the CRN using the FSP method.

        Args:
            - **init_state** (np.ndarray): Array of shape :math:`(N_\\theta, 2N)` such that \
                            init_state[i,:] is the initial state for the probabilities and sensitivities of the i-th reaction.\
                            The shape of each initial state vector is the number of states: :math:`2(\\frac{Cr(Cr+3)}{2}+1)` \
                            if :math:`n = 2`, else :math:`2(Cr+1)`.\
                            This value can also be found in attribute `n_states`.
            - :math:`t_0` (float): Starting time.
            - :math:`t_f` (float): Final time.
            - **params** (np.ndarray): Parameters of the propensity functions.
            - :math:`t_{eval}` (list): Times to svae the computed solution.

        Returns:
            - The probability vector and sensitivities of probability mass functions matrix for each time point.
        """
        sensitivities = []
        for i in range(self.n_reactions):
            solution = self.solve_ode(init_state[i,:], t0, tf, params, i, t_eval)['y']
            sensitivities.append(solution[self.n_states:,:].T)
        probs = solution[:self.n_states,:].T
        return probs, np.stack(sensitivities, axis=-1)


    def marginal(self, 
                ind_species: int, 
                init_state: np.ndarray, 
                time_samples: list, 
                params: np.ndarray, 
                t0: float =0) -> Tuple[np.ndarray]:
        """Computes marginal probabilities and marginal sensitivities of probability mass functions.

        Args:
            - **ind_species** (int): Index of the species of interest.
            - **init_state** (np.ndarray): Array of dimensions :math:`(N_\\theta, 2N)` such that \
                            init_state[i,:] is the initial state for the probabilities and sensitivities for the i-th reaction.\
                            The length of each initial state vector must be the number of states: :math:`2(\\frac{Cr(Cr+3)}{2}+1)` \
                            if :math:`n = 2`, else :math:`2(Cr+1)`.\
                            It can also be found in attribute `n_states`.
            - **time_samples** (list): Times to save the computed solution.
            - **params** (np.ndarray): Parameters of the propensity functions.
            - :math:`t_0` (float, optional): Initialization time. By default, 0.

        Returns:
            - (Tuple[np.ndarray]): The first element is marginal probability vector for the species of interest at each time, \
                of dimensions :math:`(N_t, \\frac{Cr(Cr+3)}{2}+1)`. The second element is the marginal sensitivities of probability mass \
                function for the species of interest at each time, of dimensions :math:`(N_t, \\frac{Cr(Cr+3)}{2}+1, M)`.
        """        
        marginal_probs = np.zeros((len(time_samples), self.cr+1))
        marginal_sensitivities = np.zeros((len(time_samples), self.cr+1, len(params)))
        probs, s = self.get_sensitivities(init_state, t0, time_samples[-1], params, t_eval=time_samples)
        for n, state in self.bijection.bijection.items():
            for i, _ in enumerate(time_samples):
                marginal_probs[i, state[ind_species]] += probs[i, n]
                marginal_sensitivities[i, state[ind_species], :] += s[i, n, :]
        return marginal_probs, marginal_sensitivities

    def marginals(self, ind_species: list, init_state: np.ndarray, time_samples: list, params: np.ndarray):
        """Computes marginal distributions for multiple species.

        Args:
            - **ind_species** (list): List of index of the species of interest.
            - **init_state** (np.ndarray): Array of dimensions :math:`(N_\\theta, 2N)` such that 
              init_state[i,:] is the initial states for the probabilities and sensitivities for the i-th reaction. 
              The length of each initial state vector must be the number of states: :math:`2(\\frac{Cr(Cr+3)}{2}+1)` 
              if :math:`n = 2`, else :math:`2(Cr+1)`. It can also be found in attribute `n_states`.
            - **time_samples** (list): Times to save the computed solution.
            - **params** (np.ndarray): Parameters of the propensity functions.

        Returns:
            - (dict): Each key of the dictionary is the index of one species. Its value is the marginal distribution as returned by \
                the function ``marginal`` for this species.
        """        
        marginal_probs = {}
        marginal_stv = {}
        for ind in ind_species:
            marginal_probs[ind], marginal_stv[ind] = self.marginal(ind, init_state, time_samples, params)
        return marginal_probs, marginal_stv