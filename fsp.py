import scipy.sparse as sp
import numpy as np
from scipy.integrate import solve_ivp
from bidict import bidict
import simulation
import generate_data
from typing import Tuple, Union, Callable
import collections.abc as abc

class StateSpaceEnumeration:
    r"""State space enumeration as presented in :cite:`gupta2017projection`.

    Computes functions :math:`\Phi_{\text{dim}}` and :math:`\Phi_{\text{dim}}^{-1}` to project the state space on the set of integers and conversely.

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
    
    def phi(self, x: Union[Tuple, np.ndarray], n: int) -> int:
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
            v = np.floor((np.sqrt(8*z+1)-1)/2)
            x2 = z - v*(v+1)/2
            return int(v - x2), int(x2)
        else:
            z1, z2 = self.phi_inverse(z, 2)
            z11 = self.phi_inverse(z1, n-1)
            return z11 + (z2,)
    
    def create_bijection(self):
        r"""Saves the bijection values in a dictionary.

        A dictionary requires unhashable keys. For this reason, we use tuples instead of arrays.
        """        
        self.bijection[self.lb] = tuple(self.cl)
        self.bijection[self.ub] = tuple(self.cr)
        for z in range(self.lb, self.ub):
            self.bijection[z] = self.phi_inverse(z, self.dim)
        # print(self.bijection)


class SensitivitiesDerivation:
    r"""Class to compute the sensitivities and probabilities with the FSP method.
    Based on :cite:`fox2019fspfim`.
    
    Args:
        - **crn** (simulation.CRN): The CRN to work on.
        - **n_time_windows** (int): Number of time windows L.
        - **index** (Tuple[list, int], optional): Index of the parameters of interest. Can either be a
            single integer value or a list of integer values. If the parameter is controlled, considers parameters
            for each time window. When None, computes the sensitivities for each parameter. Defaults to None.
        - :math:`C_r` (int, optional): Value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. 
          Defaults to :math:`4`.
    """
    def __init__(self, crn: simulation.CRN, n_time_windows: int, index: Tuple[list, int], cr: int =4):     
        self.cr = cr
        self.crn = crn # make sure to specify propensities_drv if we compute the sensitivities
        self.n_fixed_params = crn.n_fixed_params
        self.n_control_params = crn.n_control_params
        self.n_params = self.n_fixed_params + self.n_control_params
        self.n_total_params = self.n_fixed_params + self.n_control_params*n_time_windows
        self.n_time_windows = n_time_windows
        self.n_reactions = crn.n_reactions
        self.n_species = crn.n_species
        self.bijection = StateSpaceEnumeration(cr, dim=self.n_species)
        self.bijection.create_bijection()
        self.entries = self.bijection.bijection.values()
        self.n_states = len(self.entries)
        # parameters index to consider
        if index is None:
            self.index = np.arange(self.n_total_params)
        elif isinstance(index, abc.Hashable):
            if index < self.n_fixed_params:
                self.index = np.array([index])
            else:
                # control parameter
                ind = index-self.n_fixed_params
                self.index = np.array([self.n_fixed_params+i*self.n_control_params+ind%self.n_control_params for i in range(self.n_time_windows)])
        else:
            self.index = np.array(index)
            self.index = np.concatenate([index[index < self.n_fixed_params]] + # fixed parameters
                                    [(index[index >= self.n_fixed_params]) + self.n_control_params*i for i in range(self.n_time_windows)]) # control parameters
        self.time = 0
        self.current_time_window = 0
        # init_state has shape (n_states, len(index)+1)
        # first column corresponds to the probability distribution, 
        # (i+1)-th column corresponds to the sensitivities with respect to the i-th parameter distribution in the index list
        self.init_state = np.zeros((self.n_states, len(self.index)+1))
        self.init_state[self.bijection.bijection.inverse[tuple(crn.init_state)], 0] = 1
        self.current_state = self.init_state.copy()

    def reset(self):
        """Resets the class to the initial setting: sets the time to :math:`t=0`, the current time window to :math:`0`
        and the current state to the initial state.
        """        
        self.time = 0
        self.current_time_window = 0
        self.current_state = self.init_state.copy()

    def create_generator(self, params: np.ndarray) -> np.ndarray:
        r"""Computes the generator matrix :math:`A` as defined in :cite:`fox2019fspfim`.

        Args:
            - **params** (np.ndarray): Current parameters of the propensity functions.

        Returns:
            - Generator **A** in the general case of non-mass-action kinetics.
        """
        Bs = []
        # generator for each reaction
        for index in range(self.n_reactions):
            d = self.bijection.bijection.inverse
            n = self.n_states
            stoich_mat = self.crn.stoichiometry_mat[:,index]
            propensity = self.crn.propensities[index]
            # might contain negative elements
            outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
            # propensity parameter is 1
            data = np.array(list(map(lambda x: propensity(params, x), self.entries)))
            columns = np.array([d[entry] for entry in self.entries])
            get_index = lambda key: d[key] if key in d else -1
            rows = np.array([get_index(output) for output in outputs])
            compute_diags = np.vectorize(lambda i: -data[columns==i].sum())
            diags = compute_diags(np.arange(n))
            # truncation
            mask = (rows >= 0) & (rows < n)
            columns = columns[mask]
            rows = rows[mask]
            data = data[mask]
            B = sp.coo_matrix((data, (rows, columns)), shape=(n, n))
            B.setdiag(diags)
            B.eliminate_zeros()
            Bs.append(B)
        return sum(Bs)

    def create_gdrv(self, params: np.ndarray, ind: int) -> np.ndarray:
        r"""Computes :math:`\frac{\partial \hat{A}^\theta}{\partial \theta_{\text{ind}}}` in the
        case of non-mass-action kinetics.

        Requires the propensity derivatives to be explicitly defined.

        Args:
            - **params** (np.ndarray): Current parameters of the propensity functions.
            - **index** (int): Index of the parameter from which :math:`\hat{A}^\theta` is derived.
        """
        dA = []
        for index in range(self.n_reactions):
            d = self.bijection.bijection.inverse
            n = self.n_states
            stoich_mat = self.crn.stoichiometry_mat[:,index]
            propensity = self.crn.propensities_drv[index, ind]
            # might contain negative elements
            outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
            # propensity parameter is 1
            data = np.array(list(map(lambda x: propensity(params, x), self.entries)))
            columns = np.array([d[entry] for entry in self.entries])
            get_index = lambda key: d[key] if key in d else -1
            rows = np.array([get_index(output) for output in outputs])
            compute_diags = np.vectorize(lambda i: -data[columns==i].sum())
            diags = compute_diags(np.arange(n))
            # truncation
            mask = (rows >= 0) & (rows < n)
            columns = columns[mask]
            rows = rows[mask]
            data = data[mask]
            # according tox Fox and Munsky's paper
            dAi = sp.coo_matrix((data, (rows, columns)), shape=(n, n))
            dAi.setdiag(diags)
            dAi.eliminate_zeros()
            dA.append(dAi)
        return sum(dA)

    def create_gdrv_B(self, ind: int) -> np.ndarray:
        r"""Computes :math:`\frac{\partial \hat{A}^\theta}{\partial \theta_{\text{ind}}}` in the case of mass-action kinetics.
        In that case, :math:`\frac{\partial\hat{A}^\theta}{\partial \theta_{\text{ind}}} = \hat{B}_{\text{ind}}` 
        where the rate matrix :math:`B_i` is as defined in :cite:`fox2019fspfim`.

        Args:
            - **ind** (int): Index of the parameter from which :math:`\hat{A}^\theta` is derived.
        """
        d = self.bijection.bijection.inverse
        n = self.n_states
        stoich_mat = self.crn.stoichiometry_mat[:,ind]
        propensity = self.crn.propensities[ind]
        # might contain negative elements
        outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
        # propensity parameter is 1
        data = np.array(list(map(lambda x: propensity(np.ones(self.n_params), x), self.entries)))
        #
        columns = np.array([d[entry] for entry in self.entries])
        get_index = lambda key: d[key] if key in d else -1
        rows = np.array([get_index(output) for output in outputs])
        compute_diags = np.vectorize(lambda i: -data[columns==i].sum())
        diags = compute_diags(np.arange(n))
        # truncation
        mask = (rows >= 0) & (rows < n)
        columns = columns[mask]
        rows = rows[mask]
        data = data[mask]
        # according tox Fox and Munsky's paper
        B = sp.coo_matrix((data, (rows, columns)), shape=(n, n))
        B.setdiag(diags)
        B.eliminate_zeros()
        return B


    def create_generator_derivative(self, params: np.ndarray, ind: int) -> np.ndarray:
        r"""Computes :math:`\frac{\partial \hat{A}^\theta}{\partial \theta_{\text{ind}}}` in the general case.

        Args:
            - **params** (np.ndarray): Current parameters of the propensity functions.
            - **ind** (int): Index of the parameter from which :math:`\hat{A}^\theta` is derived.
        """
        # by default, mass-action kinetics
        if self.crn.propensities_drv is None: 
            return self.create_gdrv_B(ind)
        return self.create_gdrv(params, ind)

    def constant_matrix(self, params: np.ndarray) -> np.ndarray:
        r"""Computes the constant matrix **C** to solve the set of ODEs, possibly for several parameters at once.

        Let us introduce :math:`\alpha \in \mathbb{N}^*`, and let :math:`I=\{i_1, i_2, ..., i_{\alpha}\} \subset [\![1, M]\!]`
        be the set of parameters indices defined in `index`. The constant matrix **C** in the set of ODEs to solve the sensitivities 
        with respect to several parameters is given by: 

        .. math::
            C = \\begin{pmatrix} \hat{A}^\theta & 0 && ... && 0 \\\ \frac{\partial \hat{A}^\theta}{\partial \theta_{i_1}} & A && 0 ... & 0
            \\\ ... \\\ \frac{\partial \hat{A}^\theta}{\partial \theta_{i_\alpha}} & 0 && ... & 0 & A \\end{pmatrix}

        Args:
            - **params** (np.ndarray): Current parameters of the propensity functions. Has shape :math:`(n_{\text{params}})`.
        """
        n = len(self.index)
        A = self.create_generator(params)
        empty = sp.coo_matrix(A.shape)
        rows = [sp.hstack([A]+[empty]*n)]
        current_params = self.n_fixed_params + self.current_time_window*self.n_control_params #
        for i, ind in enumerate(self.index):
            if ind >= self.n_fixed_params and (current_params > ind or current_params + self.n_control_params <= ind):
                # the parameter n°ind has no action on the current time window
                B = sp.coo_matrix(A.shape)
            else:
                if ind > self.n_fixed_params:
                    # the parameter n°ind is controlled and is the one used on the current time window
                    ind = ind - self.current_time_window*self.n_control_params
                B = self.create_generator_derivative(params, ind)
            row = sp.hstack([B] + [empty]*i + [A] + [empty]*(n-i-1))
            rows.append(row)
        return sp.vstack(rows)

    def solve_ode(self, 
                init_state: np.ndarray, 
                t0: float, 
                tf: float, 
                params: np.ndarray,  
                t_eval: list,
                with_stv: bool =True):
        r"""Solves the following set of linear ODEs, which allows to solve the sensitivities with respect to several parameters at once:

        ..math::
            \\frac{d}{dt}\\begin{pmatrix} p(t) \\\ S_{i_1}(t) \\\ ... \\\ S_{i_\alpha} \\end{pmatrix} 
            = C
            \\begin{pmatrix} p(t) \\\ S_{i_1}(t) \\\ ... \\\ S_{i_\alpha} \\end{pmatrix}

        where :math:`I=\{i_1, i_2, ..., i_{\alpha}\} \subset [\![1, M]\!]` is the set of parameters indices defined in `index`
        and :math:`C` is as defined in the function `constant_matrix`.

        Args:
            - **init_state** (np.ndarray): Initial state for probabilities and sensitivities.
              Has shape (n_states*(n_fixed_params+1)).

            .. math::
                2\big(\\frac{Cr(Cr+3)}{2}+1\big) \\text{ if } n = 2, 2(Cr+1) \\text{ otherwise.}

            - :math:`t_0` (float): Starting time.
            - :math:`t_f` (float): Final time.
            - **params** (np.ndarray): Parameters of the propensity functions. Has shape (n_params).
            - :math:`t_{eval}` (list): Sampling times.
            - **with_stv** (bool): If True, computes the corresponding sensitivities. If False, only solves the first part 
              of the ODE to compute the probability mass function. Defaults to True.

        Returns:
            - Bunch object as the output of the ``solve_ivp`` function applied to the set of linear ODEs.
              To access the probability and sensitivities distribution, use the key 'y'. The result has shape 
              :math:`(n_{\text{states}}, L)` if `with_stv` is False, :math:`(2n_{\text{states}}, L)` otherwise, 
              :math:`L` being the number of sampling times.
        """
        # init_state is a 1D vector
        if with_stv:
            constant = sp.csr_matrix(self.constant_matrix(params))
            def f(t, x):
                return constant.dot(x)
            return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)
        A = sp.csr_matrix(self.create_generator(params))
        def f(t, x):
            return A.dot(x)
        return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)


        
    def solve_multiple_odes(self,
                            sampling_times: np.ndarray,
                            time_windows: np.ndarray,
                            parameters: np.ndarray,
                            with_stv: bool =True) -> np.ndarray:
        """Solves the set of ODEs for several parameters at once, taking into account varying parameters over time windows.

        Args:
            - **sampling_times** (np.ndarray): Sampling times.
            - **time_windows** (np.ndarray): Time windows during which the parameters do not vary.
              Its form is :math:`[t_1, ..., t_L]`, such that the considered time windows are 
              :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match with the final time 
              :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the propensity functions for each time window.
              Has shape (L, n_params).
            - **with_stv** (bool, optional): If True, computes the sensitivities of the mass function 
              with respect to the indices in `self.index. If False, computes only the probability distribution. 
              Defaults to True.

        Returns:
            - The probability and, if **with_stv** is True, the sensitivities distributions for each sampling time.
              Has shape (n_states, L, len(index)+1) if **with_stv** is True and (n_states, L, 1)
              if **with_stv** is False.
        """        
        distributions = []
        for i, t in enumerate(time_windows):
            if sampling_times[-1] < t:
                break
            # parameters has shape (L, n_params)
            params = parameters[i, :]
            t_eval = sampling_times[(sampling_times > self.time) & (sampling_times <= t)]
            added_t = None
            if len(t_eval) == 0 or t_eval[-1] != t:
                # to get state at time t to update the current state
                t_eval = np.concatenate((t_eval, [t]))
                added_t = -1
            if with_stv:
                # computes sensitivities for all fixed reactions
                solution = self.solve_ode(init_state=self.current_state.reshape(self.n_states*(len(self.index)+1), order='F'),
                                        t0=self.time,
                                        tf=t, 
                                        params=params,
                                        t_eval=t_eval,
                                        with_stv=True)['y']
                # reshaping the array
                solution = solution.reshape((self.n_states, len(self.index)+1, len(t_eval)), order='F')
                distributions.append(solution.transpose([0, 2, 1])[:,:added_t,:]) # shape (n_states, L, len(index)+1)
                self.current_state = solution[:,:,-1]
            else:
                solution = self.solve_ode(init_state=self.current_state[:,0], 
                                        t0=self.time,
                                        tf=t, 
                                        params=params,
                                        t_eval=t_eval,
                                        with_stv=False)['y']
                distributions.append(np.expand_dims(solution[:, :added_t], axis=-1)) # shape (n_states, L, 1)
                self.current_state[:,0] = solution[:,-1]
            self.time = t
            self.current_time_window += 1
        return np.concatenate(distributions, axis=1) # shape (n_states, L, 1) or (n_states, L, len(index)+1)


    def marginal(self,  
                sampling_times: np.ndarray,
                time_windows: np.ndarray,
                parameters: np.ndarray, 
                ind_species: int,
                with_stv: bool =True
                ) -> np.ndarray:
        r"""Computes marginal probability mass functions and marginal sensitivities of probability mass functions.

        Args:
            - **sampling_times** (np.ndarray): Sampling times.
            - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
              such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
              with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the propensity functions.
            - **ind_species** (int): Index of the species of interest.
            - **with_stv** (bool, optional): If True, computes the sensitivities of the mass function. 
              If False, computes only the probability distribution. Defaults to True.

        Returns:
            - The marginal probability and, if **with_stv** is True, the marginal sensitivities of probability mass function for each sampling time.
              Has shape (n_states, L, len(index)+1) if **with_stv** is True and (n_states, L, 1) otherwise.
        """
        if with_stv:
            marginal_distributions = np.zeros((self.cr+1, len(sampling_times), len(self.index)+1))
        else:
            marginal_distributions = np.zeros((self.cr+1, len(sampling_times), 1))
        solution = self.solve_multiple_odes(sampling_times, time_windows, parameters, with_stv)
        for n, state in self.bijection.bijection.items():
            for i, _ in enumerate(sampling_times):
                marginal_distributions[state[ind_species],i,:] += solution[n,i,:]
        return marginal_distributions # shape (n_states, L, 1) or (n_states, L, len(index)+1)

    def marginals(self, 
                sampling_times: np.ndarray, 
                time_windows: np.ndarray, 
                parameters: np.ndarray, 
                ind_species: list,
                with_stv: bool =True) -> dict:
        """Computes marginal probability mass functions and marginal sensitivities of probability mass functions for multiple species.

        Args:
            - **sampling_times** (np.ndarray): Sampling times.
            - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
              such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
              with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the propensity functions.
            - **ind_species** (list): List of index of the species of interest.
            - **with_stv** (bool, optional): If True, computes the sensitivities of the mass function. 
              If False, only computes the probability distribution. Defaults to True.

        Returns:
            - Each key of the dictionary is the index of one species. Its value is the marginal distribution as returned by
              the function ``marginal`` for this species.
        """
        marginal_distributions = {}
        for ind in ind_species:
            marginal_distributions[ind] = self.marginal(sampling_times, time_windows, parameters, ind, with_stv)
        return marginal_distributions

    def identity(self, x):
        return x

    def expected_val(self, 
                    sampling_times: np.ndarray, 
                    time_windows: np.ndarray, 
                    parameters: np.ndarray, 
                    ind_species: int,
                    loss: Union[Callable, list] =None) -> np.ndarray:
        r"""Computes the expected value of a loss function evaluated at an abundance random variable
        :math:`E_{\theta, \xi}[\mathcal{L}(X_t)]`.

        Args:
            - **sampling_times** (np.ndarray): Sampling times.
            - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
              such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
              with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the propensity functions.
            - **ind_species** (int): Index of the species of interest.
            - **loss** (Union[Callable, list], optional): Loss function. If it is a list, each element is the loss function for the
              corresponding time window. By defaults when None, the loss is set to the identity function. Defaults to None.
        """        
        if loss is None:
            # by default, identity function
            loss = self.identity
        marginal_distributions = self.marginal(sampling_times=sampling_times,
                                                time_windows=time_windows,
                                                parameters=parameters,
                                                ind_species=ind_species,
                                                with_stv=False)[:,:,0]
        self.reset()
        if isinstance(loss, abc.Hashable):
            # one loss for all time windows
            return loss(np.dot(marginal_distributions.transpose(), np.arange(self.cr+1))) # shape(Nt)
        else:
            res = []
            x = np.arange(self.cr + 1)
            for i, loss_f in enumerate(loss):
                res.append(loss_f(np.dot(marginal_distributions[:, i], x)))
            return np.concatenate(res)
        # else:
        #     scalar = np.zeros((self.n_time_windows, self.cr + 1))
        #     for i, loss_f in enumerate(loss):
        #         vectorized_loss_f = np.vectorize(loss_f)
        #         scalar[i, :] = vectorized_loss_f(np.arange(self.cr + 1))
        #     return np.diag(np.dot(scalar, marginal_distributions))

    def gradient_expected_val(self, 
                            sampling_times: np.ndarray, 
                            time_windows: np.ndarray, 
                            parameters: np.ndarray, 
                            ind_species: int,
                            loss: Union[Callable, list] =None,
                            grad_loss: Union[Callable, list] =None) -> np.ndarray:
        r"""Computes the gradient of the loss function evaluated at the expected value with respect to 
        one or several parameters defined in `index` 
        
        ..math::
            \nabla_{\theta, \xi} \mathcal{L}\big(E_{\theta, \xi}[X_t]\big) = \nabla_{x}\mathcal{L}(x) \nabla_{\theta, \xi}E_{\theta, \xi}[X_t]

        Args:
            - **sampling_times** (np.ndarray): Sampling times.
            - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
              such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
              with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the propensity functions.
            - **ind_species** (int): Index of the species of interest.
            - **loss** (Union[Callable, list], optional): Loss function. If it is a list, each element is the loss function for the
              corresponding time window. By defaults when None, the loss is set to the identity function. Defaults to None.
            - **grad_loss** (Union[Callable, list], optional):
        """       
        if loss is None:
            loss = self.identity
        marginal_distributions = self.marginal(sampling_times=sampling_times, 
                                                time_windows=time_windows, 
                                                parameters=parameters, 
                                                ind_species=ind_species,
                                                with_stv=True)
        self.reset()
        stv = marginal_distributions[:, :, 1:]
        return np.dot(np.transpose(stv, [1, 2, 0]), np.arange(self.cr+1)) # shape (Nt, len(index))
        # else:
        #     scalar = np.zeros((self.n_time_windows, self.cr + 1))
        #     for i, loss_f in enumerate(loss):
        #         vectorized_loss_f = np.vectorize(loss_f)
        #         scalar[i, :] = vectorized_loss_f(np.arange(self.cr + 1))
        #     pass
        #     # return np.diag(np.dot(scalar, marginal_distributions))


    def create_gradient(self, sampling_times, time_windows, parameters, ind_species, grad_loss):
        if grad_loss is None:
            def grad_identity(expected_val, gradient):
                return gradient
            grad_loss = grad_identity
        marginal_distributions = self.marginal(sampling_times=sampling_times,
                                                time_windows=time_windows,
                                                parameters=parameters,
                                                ind_species=ind_species,
                                                with_stv=True)
        self.reset()
        stv = marginal_distributions[:, :, 1:]
        probs = marginal_distributions[:, :, 0]
        gradient = np.dot(np.transpose(stv, [1, 2, 0]), np.arange(self.cr + 1))
        expected_val = np.dot(probs.transpose(), np.arange(self.c + 1))
        if isinstance(grad_loss, abc.Hashable):
            # one loss for all time windows
            return grad_loss(expected_val, gradient) * gradient # shape (L)
        else:
            gradient_loss = []
            for g in grad_loss:
                gradient_loss.append(g(expected_val, gradient))
            return np.concatenate(gradient_loss) # shape (L)
        

if __name__ == '__main__':

        # from CRN6_toggle_switch import propensities_toggle as propensities
        # parameters = np.arange(10)

        # crn = simulation.CRN(propensities.stoich_mat,
        #                     propensities.propensities,
        #                     propensities.init_state,
        #                     n_fixed_params=10,
        #                     n_control_params=0)
        
        # stv = SensitivitiesDerivation(crn, n_time_windows=4, index=None)


        from CRN4_control import propensities_bursting_gene as propensities
        parameters = np.arange(7)
        crn = simulation.CRN(propensities.stoich_mat,
                            propensities.propensities,
                            propensities.init_state,
                            n_fixed_params=3,
                            n_control_params=1)

        stv = SensitivitiesDerivation(crn, n_time_windows=4, index=None)


        
