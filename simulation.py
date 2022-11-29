import numpy as np
import random
from typing import Tuple, Callable

class CRN:
    """Class to specify the CRN to work on.

    Args:
        - **stoichiometric_mat** (np.ndarray): Stoichiometric matrix. It has shape (N, N_reactions).
        - **propensities** (np.ndarray): Propensity functions.
        - **n_params** (int): Number of parameters required to define the propensity functions.
        - **exact** (bool, optional): If True, the exact distribution of the CRN is known. Defaults to False.
        - **exact_distr** (Tuple[Callable], optional): The exact probability mass function, when known. Defaults to None.
        - **exact_sensitivities_prob** (Tuple[Callable], optional): The exact sensitivities of the mass function, when known. \
            Defaults to None.
    """    
    def __init__(self,
                stoichiometric_mat: np.ndarray, 
                propensities: np.ndarray, 
                n_params: int,
                init_state: np.ndarray,
                n_control_params: int =0,
                exact: bool =False, 
                exact_distr: Tuple[Callable] =None, 
                exact_sensitivities_prob: Tuple[Callable] =None):   
        # stoichiometric_mat has shape (n_species, n_reactions)
        self.stoichiometric_mat = stoichiometric_mat
        self.n_species, self.n_reactions = np.shape(stoichiometric_mat) # total number of reactions, including those whose parameters change
        self.sampling_times = np.empty(0)
        self.sampling_states = np.empty((0, self.n_species))
        self.init_state = init_state
        self.time = 0
        self.current_state = self.init_state.copy()
        self.propensities = propensities
        self.n_params = n_params
        self.n_control_params = n_control_params
        self.exact = exact
        if exact:
            self.exact_distr = exact_distr
            self.exact_sensitivities_prob = exact_sensitivities_prob

    def step(self, 
            init_state: np.ndarray, 
            params: np.ndarray, 
            sampling_times: np.ndarray, 
            t0: float,
            tf: float,
            method: str = 'SSA') -> Tuple[np.ndarray]: 
        """Simulates the specified CRN.

        Args:
            - **init_state** (np.ndarray): Initial state of the CRN when starting the simulation.
            - **params** (np.ndarray): Parameters associated to the propensity functions.
            - **sampling_times** (np.ndarray): Times to sample.
            - :math:`t_f` (float): Time to end the simulation.
            - **method** (str): Stochastic Simulation to compute. Defaults to 'SSA'.

        Returns:
            Results of the Stochastic Simulations.

            - **sampling_times** (np.ndarray): Times to sample.
            - **samples** (np.ndarray): Samples at the sampling times.
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
                                            stoich_mat=self.stoichiometric_mat)
        if method == 'SSA':
            samples = simulations.SSA()
        else:
            samples = simulations.mNRM()
        self.sampling_times = np.concatenate((self.sampling_times, sampling_times))
        self.sampling_states = np.concatenate((self.sampling_states, samples))
        self.current_state = simulations.current_state
        self.time = tf

    def simulation(self, sampling_times, time_slots, parameters, method):
        # time_slots [t1, ..., tf] with t0 = 0
        for i, t in enumerate(time_slots):
            self.step(init_state=self.current_state, 
                    params=parameters[i,:], 
                    sampling_times=sampling_times[(sampling_times > self.time) & (sampling_times <= t)],
                    t0=self.time,
                    tf=t,
                    method=method)

    def reset(self):
        self.time = 0
        self.current_state = self.init_state.copy()
        self.sampling_times = np.empty(0)                   
        self.sampling_states = np.empty((0, self.n_species))



class StochasticSimulation:
    """
    Class to simulate a CRN.
    
    Args:
        - :math:`x_0` (np.ndarray): Initial state.
        - :math:`t_f` (float): Final time.
        - **sampling_times** (list): Times to sample.
        - **propensities** (np.ndarray): Propensity functions.
        - **n_species** (int): :math:`N`, Number of species involved.
        - **n_reactions** (int): Number of reactions that can occur.
        - **stoich_mat** (np.ndarray): Stoichiometric matrix.     
    """    
    def __init__(self,
                x0: np.ndarray,
                t0: float, 
                tf: float, 
                sampling_times: list, 
                propensities: np.ndarray, 
                n_species: int, 
                n_reactions: int, 
                stoich_mat: np.ndarray):
        self.final_time = tf
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.time = t0
        self.samples = []
        self.sampling_times = sampling_times
        self.current_state = x0
        self.propensities = propensities
        self.stoich_mat = stoich_mat


    def SSA(self) -> Tuple[np.ndarray]:
        """Computes the SSA.

        Returns:
            - **sampling_times** (np.ndarray): Times to sample.
            - **samples** (np.ndarray): Samples at the sampling times.
        """        
        while True:
            eval_propensities = np.vectorize(lambda f, x: f(x), excluded=[1], otypes=[np.ndarray])
            lambdas = eval_propensities(self.propensities, self.current_state)
            lambda0 = lambdas.sum()
            probabilities = np.cumsum(lambdas) / lambda0
            delta = np.random.exponential(1/lambda0)
            self.time += delta
            if self.time > self.final_time:
                # last samples
                for _ in range(len(self.sampling_times) - len(self.samples)):
                    self.samples.append(list(self.current_state))
                break
            # choosing which reaction occurs
            u = random.random()
            ind_reaction = np.searchsorted(probabilities, u, side='right') # the reaction n°ind_reaction occurs
            # sampling if needed
            last_index = int(np.searchsorted(self.sampling_times, self.time - delta, side='left'))
            current_index = int(np.searchsorted(self.sampling_times, self.time, side='left'))
            for _ in range(current_index - last_index):
                self.samples.append(list(self.current_state))
            # updating state
            self.current_state += self.stoich_mat[:, ind_reaction]
        return np.array(self.samples)

    def mNRM(self):
        """Computes the mNRM.
        """
        pass


