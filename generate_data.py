import numpy as np
import math
import scipy.stats.qmc as qmc
import time
import concurrent.futures
import simulation
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Union
# from scipy.stats import poisson
# import propensities
from scipy.stats import nbinom
from CRN2_birth_death import propensities_birth_death as propensities
import psutil

class CRN_Dataset:

    def __init__(self, 
            crn: simulation.CRN, 
            sampling_times: list[float], 
            n_trajectories: int =10**3, 
            ind_specy: int =0):
        """
        Initializes parameters to build our dataset.

        Inputs: 'crn': Chemical Reaction Network to work on.
                'sampling_times': Times to sample.
                'n_trajectories': Number of trajectories to compute. 
                                Can also be defined when calling the 'generate_data' function.
                'ind_specy': Index of the specy of interest. 
                            The distribution generated will be the one of that specy.
        """
        self.crn = crn
        self.n_params = crn.n_params
        self.n_species = crn.n_species
        self.sampling_times = sampling_times
        self.n_trajectories = n_trajectories
        self.ind_specy = ind_specy
        self.initial_state = np.zeros(self.n_species)


    def samples_probs(self, params: np.ndarray):
        """
        Runs n_trajectories of the SSA simulation for the parameters in input and deducts the corresponding distribution
        for the specy indexed by 'ind_specy'.

        Inputs: 'params': the parameters of the propensities.

        Outputs:
            'samples': list of the distributions for the corresponding specy at times time_samples.
                        This list begins with time and parameters: [t, param1, ..., paramK, distr]
                        in order to keep parameters linked to the distribution.
            'max_value': maximum value reached during simulations + number of parameters + 1 (for time).
                        Will be useful to standardize each data length to turn the data list into a tensor.
        """
        
        res = []
        # print(0)
        for _ in range(self.n_trajectories):
            # print(psutil.virtual_memory())
            # print(1)
            _, samples = self.crn.simulation(self.initial_state.copy(),
                                            params, 
                                            self.sampling_times, self.sampling_times[-1])
            # print(2)
            res.append(samples)
        res = np.array(res)
        max_value = int(np.max(res))
        # print(1)
        # Counts of events for each specy
        distr = np.empty((int(max_value) + 1, len(self.sampling_times), self.crn.n_species))
        for i in range(int(max_value)+1):
            distr[i,:,:] = np.count_nonzero((res == i), axis=0)
        # final output
        samples = []
        # print(2)
        for i, t in enumerate(self.sampling_times):
            sample = [t] + list(params) + list(distr[:, i, self.ind_specy])
            samples.append(sample)
        # + 1 for time
        # print(3)
        return samples, max_value + self.n_params + 1

    def set_length(self, onedim_tab: np.ndarray, length: int):
        """
        Inputs: 'onedim_tab': array to extend. In our case, 1D array.
                'length': wanted length of array.

        Output: An array of length 'length' equal to 'onedim_tab' array completed by zeros at the end.
        """
        return onedim_tab + [0] * max(length - len(onedim_tab), 0)

    def generate_data(self, 
                    data_length: int, 
                    n_trajectories: int =10**4, 
                    sobol_length: float =2., 
                    ind_specy: Union[int, np.ndarray] =0,
                    initial_state: Tuple[bool, np.ndarray] =(False, None)):
        """
        Generates a training, validation or test dataset.
        We use multiprocessing to run simulations in parallel to compute faster.
        Parameters are generated from the Sobol Sequence.

        Inputs: 'data_length': length of data expected in outputs.
                'n_trajectories': number of trajectories to compute to build the distribution.
                'sobol_length': upper bound of the hypercube [0, length]^N in which each set of parameters lie.
                'ind_specy': index of the specy of which to compute the distribution.
        
        Output: A tuple of arrays '(X, y)'. 
                Each entry of 'X' is an input to the neural network of the form '[t, params...]'
                The corresponding entry of 'y' is the expected probability distribution for these parameters.
        """
        self.n_trajectories = n_trajectories
        self.ind_specy = ind_specy
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
        params = sobol.random(n_elts)*sobol_length # array of n_elts of parameters set, each set of length n_params
        # to avoid all zeros
        params[np.count_nonzero(params, axis=1) == 0] = sobol.random()*sobol_length
        # using multithreading to process faster
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = list(tqdm(executor.map(self.samples_probs, params), total = len(params), desc='Generating data ...'))
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

# CRN1

# if __name__ == '__main__':

#     CRN_NAME = 'ø_S1'
#     datasets = {'train': 20, 'valid': 0, 'test': 0}
#     DATA_LENGTH = sum(datasets.values())

#     stoich_mat = np.array([1]).reshape(1,1)
#     crn = simulation.CRN(stoichiometric_mat=stoich_mat, propensities=np.array([propensities.lambda1]), n_params=1)
#     dataset = CRN_Dataset(crn=crn, sampling_times=np.array([0, 1, 5, 10, 15]))
#     X, y = dataset.generate_data(data_length=DATA_LENGTH)

#     # print(np.shape(y))
#     index = 4
#     print(X[index, :])
#     print(y[index, :])
#     t = X[index, 0]
#     lambd = X[index,1]
#     exact = [poisson.pmf(k, t*lambd) for k in np.arange(len(y[index,:]))]
#     plt.plot(y[index,:])
#     plt.plot(np.arange(len(y[index,:])), exact, marker = 'x', color = 'red')
#     plt.show()

# CRN2

if __name__ == '__main__':

    CRN_NAME = 'birth_death'
    datasets = {'train': 20, 'valid': 0, 'test': 0}
    DATA_LENGTH = sum(datasets.values())

    stoich_mat = propensities.stoich_mat.reshape(1, 2)
    crn = simulation.CRN(stoichiometric_mat=stoich_mat, propensities=np.array([propensities.lambda1, propensities.lambda2]), n_params=2)
    dataset = CRN_Dataset(crn=crn, sampling_times=np.array([5, 10, 15, 20]))
    X, y = dataset.generate_data(data_length=DATA_LENGTH, initial_state=(True, np.ones(1)))

    # print(np.shape(y))
    index = 4
    print(X[index, :])
    print(y[index, :])
    t = X[index, 0]
    lambd = X[index,1]
    exact = nbinom.pmf(np.arange(len(y[index,:])), 1, np.exp(-t*lambd))
    plt.plot(y[index,:])
    plt.plot(np.arange(len(y[index,:])), exact, marker = 'x', color = 'red')
    plt.show()