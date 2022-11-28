import numpy as np
import math
import scipy.stats.qmc as qmc
import time
import concurrent.futures
import simulation
from tqdm import tqdm
from typing import Tuple, Union

class CRN_Dataset:
    r"""Class to build a dataset of probability distributions for a specified CRN.

    Args: 
        - **crn** (simulation.CRN): CRN to work on.
        - **sampling_times** (list): Times to sample.
        - :math:`n_{\text{trajectories}}` (int, optional): Number of trajectories to compute. 
          Can also be defined when calling the ``generate_data`` function. Defaults to :math:`10^4`.
        - **ind_species** (int, optional): Index of the species of interest. 
          The distribution generated will be the one of that species. Defaults to :math:`0`.
        - **method** (str, optional): Stochastic Simulation to compute. Defaults to 'SSA'.
    """
    def __init__(self, 
            crn: simulation.CRN, 
            sampling_times: list, 
            n_trajectories: int =10**4, 
            ind_species: int =0,
            method: str ='SSA'):     
        self.crn = crn
        self.n_params = crn.n_params
        self.n_species = crn.n_species
        self.sampling_times = sampling_times
        self.n_trajectories = n_trajectories
        self.ind_species = ind_species
        self.initial_state = np.zeros(self.n_species)
        self.method = method


    def samples_probs(self, params: np.ndarray) -> Tuple[list, int]:
        r"""Runs :math:`n_{\text{trajectories}}` of Stochastic Simulations for the parameters in input and deducts the corresponding distribution 
        for the species indexed by **ind_species**.

        Args:
            - **params** (np.ndarray): Parameters associated to the propensity functions.

        Returns:
            - **samples**: List of the distributions for the corresponding species at sampling times. 
              This list begins with time and parameters: :math:`[t, \theta_1, ..., \theta_M, p_0(t,\theta), ...]`.
            - **max_value**: Maximum value reached during simulations + number of parameters + 1 (for time). 
              Used to standardize each data length to turn the data list into a tensor.
        """
        res = []
        for i in range(self.n_trajectories):
            _, samples = self.crn.step(self.initial_state.copy(),
                                            params, 
                                            self.sampling_times, self.sampling_times[-1], self.method)
            res.append(samples)
        res = np.array(res)
        max_value = int(np.max(res))
        # Counts of events for each species
        distr = np.empty((int(max_value) + 1, len(self.sampling_times), self.crn.n_species))
        for i in range(int(max_value)+1):
            distr[i,:,:] = np.count_nonzero((res == i), axis=0)
        # final output
        samples = []
        for i, t in enumerate(self.sampling_times):
            sample = [t] + list(params) + list(distr[:, i, self.ind_species])
            samples.append(sample)
        # + 1 to count the time
        return samples, max_value + self.n_params + 1

    def set_length(self, onedim_tab: np.ndarray, length: int) -> np.ndarray:
        """Adds enough zeros at the end of an array to adjust its length.

        Args:
            - **onedim_tab** (np.ndarray): Array to extend. In this case, 1D array.
            - **length** (int): Expected length of array.

        Returns:
            - The array in input with zeros at the end so that its length equals 'length'.
        """   
        return onedim_tab + [0] * max(length - len(onedim_tab), 0)

    def generate_data(self, 
                    data_length: int, 
                    n_trajectories: int =10**4, 
                    sobol_start: Union[float, list] =0.,
                    sobol_end: Union[float, list] =2.,
                    ind_species: Union[int, np.ndarray] =0,
                    initial_state: Tuple[bool, np.ndarray] =(False, None)) -> Tuple[np.ndarray]:
        r"""Generates a dataset which can be used for training, validation or testing.
        Uses multiprocessing to run multiple simulations in parallel.
        Parameters are generated from the Sobol Sequence (Low Discrepancy Sequence).

        Args:
            - **data_length** (int): Length of the expected output data.
            - :math:`n_{\text{trajectories}}` (int, optional): Number of trajectories to compute to estimate the distribution. 
              Defaults to :math:`10^4`.
            - **sobol_start** (Union[float, list], optional): Lower boundary of the parameters samples. Defaults to :math:`0`.
            - **sobol_end** (Union[float, list], optional): Upper boundary of the parameters samples. Defaults to :math:`2`.
            - **ind_species** (Union[int, np.ndarray], optional): Index of the species whose distribution is estimated. 
              Defaults to :math:`0`.
            - **initial_state** (Tuple[bool, np.ndarray], optional): Initial state of the species. Defaults to (False, None).

        Returns:
            - **(X, y)**:

                - Each entry of **X** is an input to the neural network of the form :math:`[t, \theta_1,..., \theta_M]`.
                - The corresponding entry of **y** is the estimated probability distribution for these parameters.
        """                    
        self.n_trajectories = n_trajectories
        self.ind_species = ind_species
        if initial_state[0]:
            self.initial_state = initial_state[1]
        else:
            self.initial_state = np.zeros(self.n_species, dtype=np.float32)
        n_params = self.crn.n_params
        start = time.time()
        sobol = qmc.Sobol(n_params)
        # generating parameters
        # sobol sequence requires a power of 2
        n_elts = 2**math.ceil(np.log2(data_length))
        params = sobol.random(n_elts)*(sobol_end-sobol_start)+sobol_start # array of n_elts of parameters set, each set of length n_params
        # to avoid all zeros
        params[np.count_nonzero(params, axis=1) == 0] = sobol.random()*(sobol_end-sobol_start)+sobol_start
        # using multithreading to process faster
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = list(tqdm(executor.map(self.samples_probs, params), total=len(params), desc='Generating data ...'))
        print('Simulations done.')
        distributions = []
        max_value = 0
        for distrs, value in res:
            for distr in distrs:
                distributions.append(distr)
            if value > max_value:
                max_value = value
        # shaping distributions to turn it into an array
        distributions = list(map(lambda d: self.set_length(d, max_value + 1), distributions))
        distributions = np.array(distributions)
        # split 'distributions' into input data and output data
        X = distributions[:, :1+n_params].copy()
        y = distributions[:, 1+n_params:].copy()/n_trajectories
        end=time.time()
        print('Total time: ', end-start)
        return X, y
