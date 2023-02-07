import numpy as np
import random
from typing import Tuple, Callable, Union

class CRN:
    r"""Class to specify the CRN to work on.

    Args:
        - **stoichiometry_mat** (np.ndarray): Stoichiometry matrix. Has shape :math:`(N, M)`.
        - **propensities** (np.ndarray): Non-parameterised propensity functions.
        - **init_state** (np.ndarray): Initial state of the system.
        - **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
        - **n_control_params** (int): Number of varying parameters required to define the propensity functions :math:`q_1+q_2`. 
          Their values vary from a time window to another.
        - **propensities_drv** (np.ndarray, optional): Gradient functions of the propensities with respect to the parameters.
          Has shape :math:`(M, M_{\theta}+q_1+q_2)`. If None, the CRN is assumed to follow mass-action kinetics.
          Defaults to None.
        - **exact** (Tuple[bool, Tuple[Callable], Tuple[Callable]], optional): If the first element is True, 
          the exact distribution of the CRN is known. The second element then is the exact probability mass function. The third
          element is the exact sensitivities of the mass function. Defaults to (False, None, None).
    """    
    def __init__(self,
                stoichiometry_mat: np.ndarray, 
                propensities: np.ndarray, 
                init_state: np.ndarray,
                n_fixed_params: int,
                n_control_params: int =0,
                propensities_drv: np.ndarray =None,
                exact_distr: Tuple[bool, Tuple[Callable], Tuple[Callable]] =(False, None, None)):
        # stoichiometry_mat has shape (n_species, n_reactions)
        self.stoichiometry_mat = stoichiometry_mat
        # total number of reactions, including those whose parameters change
        self.n_species, self.n_reactions = np.shape(stoichiometry_mat)
        self.sampling_times = np.empty(0)
        self.sampling_states = np.empty((0, self.n_species))
        self.init_state = init_state
        self.time = 0
        self.current_state = self.init_state.copy()
        self.propensities = propensities
        self.n_fixed_params = n_fixed_params
        self.n_control_params = n_control_params
        self.exact = exact_distr[0]
        self.propensities_drv = propensities_drv
        if exact_distr[0]:
            self.exact_distr = exact_distr[1]
            self.exact_sensitivities_prob = exact_distr[2]

    def step(self, 
            init_state: np.ndarray, 
            params: np.ndarray, 
            sampling_times: np.ndarray, 
            t0: float,
            tf: float,
            method: str,
            complete_trajectory: bool): 
        """Computes a simulation for a time window during which all parameters are fixed.

        Args:
            - **init_state** (np.ndarray): Initial state of the CRN when starting the simulation.
            - **params** (np.ndarray): Parameters associated to the propensity functions.
            - **sampling_times** (np.ndarray): Times to sample.
            - :math:`t_0` (float): Time at which the simulation starts.
            - :math:`t_f` (float): Time to end the simulation.
            - **method** (str): Stochastic Simulation to compute.
            - **complete_trajectory** (bool):
        """
        set_parameters = np.vectorize(lambda f, params: (lambda x: f(params, x)), excluded=[1])          
        lambdas = set_parameters(self.propensities, params)
        simulations = StochasticSimulation(x0=init_state,
                                            t0=t0,
                                            tf=tf,
                                            sampling_times=sampling_times,
                                            propensities=lambdas,
                                            n_species=self.n_species,
                                            n_reactions=self.n_reactions,
                                            stoich_mat=self.stoichiometry_mat)
        if method == 'SSA':
            samples = simulations.SSA(complete_trajectory)
        else:
            samples = simulations.mNRM(complete_trajectory)
        self.sampling_times = np.concatenate((self.sampling_times, simulations.sampling_times))
        self.sampling_states = np.concatenate((self.sampling_states, samples))
        self.current_state = simulations.current_state
        self.time = tf

    def simulation(self, 
                sampling_times: np.ndarray, 
                time_windows: np.ndarray, 
                parameters: np.ndarray, 
                method: str ='SSA', 
                complete_trajectory: bool =False):
        r"""Computes a simulation between two time points.

        Args:
            - **sampling_times** (np.ndarray): Times to sample.
            - **time_windows** (np.ndarray): Time windows during which all parameters are constant. Its form is :math:`[t_1, ..., t_L]`,
              such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
              with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
            - **parameters** (np.ndarray): Parameters of the simulation, including fixed parameters for the whole simulation and varying
              parameters for each time window. Its form is :math:`[\theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, ..., \xi^{q_1+q_2}_1, \xi_2^1, ..., \xi_L^{q_1+q_2}]`.
            - **method** (str, optional): Stochastic Simulation to compute. Defaults to `SSA`.
            - **complete_trajectory** (bool, optional): If True, saves the complete trajectory of the simulation, ie the time of each jump and the
              corresponding abundance. Defaults to False.
        """       
        # time_windows [t1, ..., tf] with t0 = 0
        for i, t in enumerate(time_windows):
            self.step(init_state=self.current_state, 
                        params=parameters[i,:], 
                        sampling_times=sampling_times[(sampling_times > self.time) & (sampling_times <= t)],
                        t0=self.time,
                        tf=t,
                        method=method,
                        complete_trajectory=complete_trajectory)

    def reset(self):
        """Resets the CRN to the initial setting: sets the time to :math:`t=0`, the current state to the initial state and
        empties the sampling times and samples arrays.
        """
        self.time = 0
        self.current_state = self.init_state.copy()
        self.sampling_times = np.empty(0)                   
        self.sampling_states = np.empty((0, self.n_species))



class StochasticSimulation: 
    """
    Class to run a simulation between two time points of the same time window using the Stochastic Simulation Algorithm.
    
    Args:
        - :math:`x_0` (np.ndarray): Initial state.
        - :math:`t_0` (float): Initial time of the simulation.
        - :math:`t_f` (float): Final time of the simulation.
        - **sampling_times** (np.ndarray): Times to sample.
        - **propensities** (np.ndarray): Propensity functions (parameterised).
        - **n_species** (int): Number of species involved :math:`N`.
        - **n_reactions** (int): Number of reactions of the CRN.
        - **stoich_mat** (np.ndarray): Stoichiometry matrix.     
    """   
    def __init__(self,
                x0: np.ndarray,
                t0: float, 
                tf: float, 
                sampling_times: np.ndarray, 
                propensities: np.ndarray, 
                n_species: int, 
                n_reactions: int, 
                stoich_mat: np.ndarray):
        self.final_time = tf
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.time = t0
        self.samples = []
        self.sampling_times = sampling_times # = np.empty(0) when complete_trajectory=True
        self.current_state = x0
        self.propensities = propensities
        self.stoich_mat = stoich_mat


    def SSA(self, complete_trajectory: bool =False) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Computes the SSA.

        Args:
            - **complete_trajectory** (bool): If True, returns the complete jump process, ie the time
              of each jump and the corresponding abundance.

        Returns:
            - **samples**: Abundance samples at the sampling times.
            - **sampling_times**: If **complete_trajectory** is True, also returns the time of each jump.
        """
        while True:
            eval_propensities = np.vectorize(lambda f, x: f(x), excluded=[1], otypes=[np.ndarray])
            lambdas = eval_propensities(self.propensities, self.current_state)
            lambda0 = lambdas.sum()
            probabilities = np.cumsum(lambdas) / lambda0
            delta = np.random.exponential(1/lambda0)
            self.time += delta
            if self.time > self.final_time:
                if not(complete_trajectory):
                    # last samples
                    for _ in range(len(self.sampling_times) - len(self.samples)):
                        self.samples.append(list(self.current_state))
                else:
                    if len(self.samples) == 0:
                        self.sampling_times = np.concatenate((self.sampling_times, [self.final_time]))
                        self.samples.append(list(self.current_state))
                break
            # choosing which reaction occurs
            u = random.random()
            ind_reaction = np.searchsorted(probabilities, u, side='right') # the reaction n°ind_reaction occurs
            if not(complete_trajectory):
                # sampling if needed
                last_index = int(np.searchsorted(self.sampling_times, self.time - delta, side='left'))
                current_index = int(np.searchsorted(self.sampling_times, self.time, side='left'))
                for _ in range(current_index - last_index):
                    self.samples.append(list(self.current_state))
            # updating state
            self.current_state += self.stoich_mat[:, ind_reaction]
            if complete_trajectory:
                self.sampling_times = np.concatenate((self.sampling_times, [self.time]))
                self.samples.append(list(self.current_state))
        return np.array(self.samples)

    def mNRM(self, complete_trajectory=False):
        """Computes the mNRM.
        
        To be implemented.
        """
        pass


