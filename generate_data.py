import numpy as np
import math
import scipy.stats.qmc as qmc
import time
import concurrent.futures


class CRN_Dataset:

    def __init__(self, crn, sampling_times, n_trajectories=10**3, ind_specy=0):
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
    
    def samples_probs(self, params):
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
        start_ = time.time()
        for _ in range(self.n_trajectories):
            _, samples = self.crn.simulation(np.zeros(self.n_species, dtype=np.float32), 
                                    params, self.sampling_times, self.sampling_times[-1])
            res.append(samples)
        end_ = time.time()
        print('Done, time needed:', end_ - start_)
        res = np.array(res)
        max_value = int(np.max(res))
        # Counts of events for each specy
        distr = np.empty((int(max_value) + 1, len(self.sampling_times), self.crn.n_species))
        for i in range(int(max_value)+1):
            distr[i,:,:] = np.count_nonzero((res == i), axis=0)
        # final output
        samples = []
        for i, t in enumerate(self.sampling_times):
            sample = [t] + list(params) + list(distr[:,i, self.ind_specy])
            samples.append(sample)
        # + 1 for time
        return samples, max_value + self.n_params + 1

    def set_length(self, onedim_tab, length):
        """
        Inputs: 'onedim_tab': array to extend. In our case, 1D array.
                'length': wanted length of array.

        Output: An array of length 'length' equal to 'onedim_tab' array completed by zeros at the end.
        """
        return onedim_tab + [0] * max(length - len(onedim_tab), 0)

    def generate_data(self, data_length, n_trajectories=10**3, sobol_length=10, ind_specy=0):
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
        n_params = self.crn.n_params
        start = time.time()
        sobol = qmc.Sobol(n_params, scramble=False)
        # to avoid 0.
        sobol.random()
        # generating parameters
        # sobol sequence requires a power of 2
        n_elts = 2**math.ceil(np.log2(data_length/len(self.sampling_times)))
        params = sobol.random(n_elts)*sobol_length # array of n_elts of set of parameters, each set of length n_params
        # using multithreading to process faster
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = executor.map(self.samples_probs, params)
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

