import numpy as np
import random
from typing import Tuple, Callable

class CRN:
    """Class to specify the CRN to work on.

    Args:
        - **stoichiometric_mat** (np.ndarray[int]): Stoichiometric matrix of the CRN.
        - **propensities** (np.ndarray[Callable]): Propensities of the reactions.
        - :math:`n_{params}` (int): Number of parameters required to define the propensities.
        - **exact** (bool, optional): If the exact distribution of the CRN is known. Defaults to False.
        - **exact_distr** (Tuple[Callable], optional): The exact distribution function, when known. Defaults to None.
        - **exact_sensitivities** (Tuple[Callable], optional): The exact sensitivities of probabilities with respect to the parameters, when known. Defaults to None.
    """    
    def __init__(self, 
                stoichiometric_mat: np.ndarray[int], 
                propensities: np.ndarray[Callable], 
                n_params: int, 
                exact: bool =False, 
                exact_distr: Tuple[Callable] =None, 
                exact_sensitivities_prob: Tuple[Callable] =None):       
        # stoichiometric_mat has shape (n_species, n_reactions)
        self.stoichiometric_mat = stoichiometric_mat
        self.n_species, self.n_reactions = np.shape(stoichiometric_mat)
        self.propensities = propensities
        self.n_params = n_params
        self.exact = exact
        if exact:
            self.exact_distr = exact_distr
            self.exact_sensitivities_prob = exact_sensitivities_prob

    def step(self, 
            init_state: np.ndarray[int], 
            params: np.ndarray[float], 
            sampling_times: np.ndarray[float], 
            tf: float,
            method: str = 'SSA') -> Tuple[np.ndarray[float], np.ndarray[float]]: 
        """Simulate the specified CRN with stochastic simulations.

        Args:
            - **init_state** (np.ndarray[int]): Initial state of the CRN when starting the simulation
            - **params** (np.ndarray[float]): Parameters associated to the propensities.
            - **sampling_times** (np.ndarray[float]): Times at which to sample.
            - :math:`t_f` (float): Final time at which to end the simulation.

        Returns:
            - **sampling_times** (np.ndarray[float]): Times at which samplings were done.
            - **samples** (np.ndarray[float]): Samples at the sampling times.
        """
        self.state = init_state
        set_parameters = np.vectorize(lambda f, params: (lambda x: f(params, x)), excluded=[1])          
        lambdas = set_parameters(self.propensities, params)
        simulations = StochasticSimulation(init_state, tf, sampling_times, lambdas, self.n_species, self.n_reactions, self.stoichiometric_mat)
        if method == 'SSA':
            return simulations.SSA()
        else:
            return simulations.mNRM()



class StochasticSimulation:
    """
    Class to simulate a CRN using Stochastic Simulations.
    
    Args:
        - :math:`x_0` (np.ndarray[int]): Initial state.
        - :math:`t_f` (float): Final time.
        - **sampling_times** (list[float]): Times at which to sample.
        - **propensities** (np.ndarray[Callable]): Propensities of the CRN.
        - :math:`n_{species}` (int): Number of species involved.
        - :math:`n_{reactions}` (int): Number of reactions that can occur.
        - **stoich_mat** (np.ndarray[int]): Stoichiometric matrix of the CRN.     
    """    
    def __init__(self,
                x0: np.ndarray[int], 
                tf: float, 
                sampling_times: list[float], 
                propensities: np.ndarray[Callable], 
                n_species: int, 
                n_reactions: int, 
                stoich_mat: np.ndarray[int]):         
        self.final_time = tf
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.time = 0
        self.samples = []
        self.sampling_times = sampling_times
        self.current_state = x0
        self.propensities = propensities
        self.stoich_mat = stoich_mat


    def SSA(self) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """Computes the SSA until the final time.

        Returns:
            - **sampling_times** (np.ndarray[float]): Times at which samplings were done.
            - **samples** (np.ndarray[float]): Samples at the sampling times.
        """        
        while True:
            eval_propensities = np.vectorize(lambda f, x: f(x), excluded=[1], otypes=[np.ndarray])
            lambdas = eval_propensities(self.propensities, self.current_state)
            lambda0 = lambdas.sum()
            probabilities = np.cumsum(lambdas) / lambda0
            delta = np.random.exponential(1/lambda0)
            self.time += delta
            if (self.time > self.sampling_times[-1]) or (self.time > self.final_time):
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
        return self.sampling_times, self.samples

    def mNRM(self):
        """Computes the mNRM.
        """
        pass
