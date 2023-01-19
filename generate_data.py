import numpy as np
import math
import scipy.stats.qmc as qmc
import time
import concurrent.futures
import simulation
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from typing import Tuple, Union


class CRN_Dataset:
    r"""Class to build a dataset of probability distributions for a specified CRN.

    Args: 
        - **crn** (simulation.CRN): CRN to work on.
        - **sampling_times** (np.ndarray): Sampling times.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - :math:`n_{\text{trajectories}}` (int, optional): Number of trajectories to compute. 
          Can also be defined when calling the ``generate_data`` function. Defaults to :math:`10^4`.
        - **ind_species** (int, optional): Index of the species of interest. 
          The distribution generated will be the one of that species. Defaults to :math:`0`.
        - **method** (str, optional): Stochastic Simulation to compute. Defaults to 'SSA'.
    """
    def __init__(self, 
            crn: simulation.CRN, 
            sampling_times: np.ndarray, 
            time_windows: np.ndarray,
            n_trajectories: int =10**4, 
            ind_species: int =0,
            method: str ='SSA'):     
        self.crn = crn
        self.n_fixed_params = crn.n_fixed_params
        self.n_control_params = crn.n_control_params
        self.n_params = self.n_fixed_params + self.n_control_params
        self.n_time_windows = len(time_windows)
        # number of total parameters required to fully define the process
        self.total_n_params = self.n_fixed_params + self.n_time_windows*self.n_control_params
        self.n_species = crn.n_species
        self.sampling_times = sampling_times
        self.n_trajectories = n_trajectories
        self.ind_species = ind_species
        self.initial_state = crn.init_state
        self.method = method
        self.time_windows = time_windows


    def samples_probs(self, params: np.ndarray) -> Tuple[list, int]:
        r"""Runs :math:`n_{\text{trajectories}}` of Stochastic Simulations for the parameters in input and deducts the 
        corresponding distribution for the species indexed by **ind_species**.

        Args:
            - **params** (np.ndarray): Parameters associated to the propensity functions for each time window. Array of shape 
              (n_time_windows, n_params).
        Returns:
            - **samples**: List of the distributions for the corresponding species at sampling times. This list begins with time and 
              parameters: :math:`[t, \theta_1, ..., \theta_M, \xi_1^1, \xi_2^1, ..., \xi_{M'}^1, ..., \xi_{M'}^T, p_0(t,\theta), ...]`.
            - **max_value**: Maximum value reached during simulations + number of total parameters + 1 (for time). 
              Used to standardize each data length to turn the data list into a tensor.
        """
        res = []
        for i in range(self.n_trajectories):
            self.crn.simulation(sampling_times=self.sampling_times, 
                                time_windows=self.time_windows,
                                parameters=params, 
                                method=self.method,
                                complete_trajectory=False)
            res.append(self.crn.sampling_states)
            self.crn.reset()
        res = np.array(res)
        max_value = int(np.max(res))
        # Counts of events for each species
        distr = np.empty((int(max_value) + 1, len(self.sampling_times), self.crn.n_species))
        for i in range(int(max_value)+1):
            distr[i,:,:] = np.count_nonzero((res == i), axis=0)
        # final output
        samples = []
        control_parameters = []
        for i in range(self.n_time_windows):
            control_parameters.append(params[i, self.n_fixed_params:])
        control_parameters = list(np.concatenate(control_parameters))
        for i, t in enumerate(self.sampling_times):
            sample = [t] + list(params[0, :self.n_fixed_params]) + control_parameters + list(distr[:, i, self.ind_species])
            samples.append(sample)
        # + 1 to count the time
        return samples, max_value + self.total_n_params + 1

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
                    sobol_start: np.ndarray =None,
                    sobol_end: np.ndarray =None,
                    n_trajectories: int =10**4) -> Tuple[np.ndarray]:
        r"""Generates a dataset which can be used for training, validation or testing.
        Uses multiprocessing to run multiple simulations in parallel.
        Parameters are generated from the Sobol Sequence (Low Discrepancy Sequence).

        Args:
            - **data_length** (int): Length of the expected output data.
            - **sobol_start** (np.ndarray): Lower boundaries of the parameters samples. Shape :math:`(n_total_params)`.
              If None, an array of zeros. Defaults to None.
            - **sobol_end** (np.ndarray): Upper boundaries of the parameters samples. Shape :math:`(n_total_params)`.
              If None, an array of ones. Defaults to None.
            - :math:`n_{\text{trajectories}}` (int, optional): Number of trajectories to compute to estimate the distribution. 
              Defaults to :math:`10^4`.

        Returns:
            - **(X, y)**:

                - Each entry of **X** is an input to the neural network of the form :math:`[t, \theta_1,..., \theta_M, \xi_1^1, ..., \xi_{M'}^T]`.
                - The corresponding entry of **y** is the estimated probability distribution for these parameters.
        """
        if sobol_start is None:
            sobol_start = np.zeros(self.n_params)
        if sobol_end is None:
            sobol_end = np.ones(self.n_params)
        self.n_trajectories = n_trajectories
        start = time.time()
        # generating parameters theta_i
        sobol_theta = qmc.Sobol(self.n_fixed_params)
        # sobol sequence requires a power of 2
        n_elts = 2**math.ceil(np.log2(data_length))
        thetas = sobol_theta.random(n_elts)*(sobol_end[:self.n_fixed_params]-sobol_start[:self.n_fixed_params])+sobol_start[:self.n_fixed_params] 
        # to avoid all zeros
        thetas[np.count_nonzero(thetas, axis=1) == 0] = sobol_theta.random()*(sobol_end[:self.n_fixed_params]-sobol_start[:self.n_fixed_params])+sobol_start[:self.n_fixed_params]
        theta = np.stack([thetas]*self.n_time_windows, axis=1) # shape (n_elts, n_time_windows, n_fixed_params)
        # generating parameters xi_i
        sobol_xi = qmc.Sobol(self.n_control_params*self.n_time_windows)
        xi = sobol_xi.random(n_elts)
        # to avoid all zeros
        xi[np.count_nonzero(xi, axis=1)==0] = sobol_xi.random()
        xi = np.reshape(xi, (n_elts, self.n_time_windows, self.n_control_params)) # shape (n_elts, n_time_windows, n_control_params)
        # rescaling
        xi = xi*(sobol_end[self.n_fixed_params:] - sobol_start[self.n_fixed_params:]) + sobol_start[self.n_fixed_params:]
        params = np.concatenate((theta, xi), axis=-1) # shape (n_elts, n_time_windows, n_params)
        # using multithreading to process faster
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = list(tqdm(executor.map(self.samples_probs, params), total=n_elts, desc='Generating data ...'))
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
        # input data contains all the parameters used for the simulation, including the sets of parameters for each time window
        X = distributions[:, :1+self.total_n_params].copy()
        y = distributions[:, 1+self.total_n_params:].copy()/n_trajectories
        end=time.time()
        print('Total time: ', end-start)
        return X, y

class CRN_Simulations:
    r"""Class to run simulations over time and to estimate the abundance evolution of a species.

    Args:
        - **crn** (simulation.CRN): CRN to work on.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_T]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{T-1}, t_T]`. :math:`t_T` must match
          with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - :math:`n_{\text{trajectories}}` (int, optional): Number of trajectories to compute. Defaults to :math:`10^3`.
        - **ind_species** (int, optional): Index of the species of study. Defaults to :math:`0`.
        - **method** (str, optional): Stochastic Simulation to compute. Defaults to 'SSA'.
        - **complete_trajectory** (bool, optional): If True, computes the complete Jump Process. If False,
          computes the abundance of the species to study at the specified sampling times. Defaults to True.
        - **sampling_times** (np.ndarray, optional): Times to sample. Should not be specified when `complete_trajectory`is True.
          Defaults to np.empty(0).
    """     
    def __init__(self, 
            crn: simulation.CRN,
            time_windows: np.ndarray,
            n_trajectories: int =10**3, 
            ind_species: int =0,
            method: str ='SSA',
            complete_trajectory: bool =True,
            sampling_times: np.ndarray =np.empty(0)):
        self.crn = crn
        self.n_fixed_params = crn.n_fixed_params
        self.n_control_params = crn.n_control_params
        self.n_params = self.n_fixed_params + self.n_control_params
        self.n_time_windows = len(time_windows)
        # number of total parameters required to fully define the process
        self.total_n_params = self.n_fixed_params + self.n_time_windows*self.n_control_params
        self.n_species = crn.n_species
        self.n_trajectories = n_trajectories
        self.ind_species = ind_species
        self.initial_state = crn.init_state
        self.method = method
        self.complete_trajectory = complete_trajectory
        self.time_windows = time_windows
        self.sampling_times = sampling_times


    def run_simulations(self, params: np.ndarray) -> Union[Tuple[dict], Tuple[np.ndarray]]:
        r"""Runs :math:`n_{\text{trajectories}}` of Stochastic Simulations for the parameters in input and deducts the 
        corresponding distribution for the species indexed by **ind_species**.

        Args:
            - **params** (np.ndarray): Parameters associated to the propensity functions for each time window. Array of shape 
              (n_time_windows, n_params).

        Returns:
            When `complete_trajectory` is True:
                - **samples** (dict): Each key is the index of the corresponding computed trajectory. Its value is an array with the abundance 
                  values after each jump.
                - **times** (dict): Each key is the index of the corresponding computed trajectory. Its alue is an array with the times at which
                  each jump occured.
            When `complete_trajectory` is False:
                - **samples** (np.ndarray): Measured abundance for each trajectory at the sampling times. Shape (n_trajectories, n_sampling_times).
                - **times** (np.ndarray): Sampling times.
        """
        if self.complete_trajectory:
            samples = {}
            times = {}
        else:
            samples = np.zeros((self.n_trajectories, len(self.sampling_times)))
            times = self.sampling_times
        fixed_params = np.stack([params[:self.n_fixed_params]]*self.n_time_windows)
        control_params = np.reshape(params[self.n_fixed_params:], (self.n_time_windows, self.n_control_params))
        parameters = np.concatenate((fixed_params, control_params), axis=-1)
        for i in range(self.n_trajectories):
            self.crn.simulation(sampling_times=self.sampling_times, 
                                time_windows=self.time_windows,
                                parameters=parameters, 
                                method=self.method,
                                complete_trajectory=self.complete_trajectory)
            if self.complete_trajectory:
                times[i] = np.concatenate((np.array([0]), self.crn.sampling_times))
                samples[i] = np.concatenate((np.array([self.initial_state[self.ind_species]]), self.crn.sampling_states[:, self.ind_species]))
            else:
                if times[0] == 0:
                    samples[i,:] = np.concatenate((np.array([self.initial_state[self.ind_species]]), self.crn.sampling_states[:, self.ind_species]))
                else:
                    samples[i,:] = self.crn.sampling_states[:, self.ind_species]
            self.crn.reset()
        return samples, times

    def plot_simulations(self, 
                        params: np.ndarray,
                        targets: np.ndarray =None, 
                        save: Tuple[bool, str] =(False, None)):
        """Plots either all the simulated trajectories if `complete_trajectory` is False or 
        the mean evolution of the abundance if `complete_trajectory` is True`.

        Args:
            - **params** (np.ndarray): Parameters associated to the propensity functions for each time window. Array of shape 
              (n_time_windows, n_params).
            - **targets** (np.ndarray, optional): Target values. If None, no target value. Defaults to None.
            - **save** (Tuple[bool, str], optional): If the first argument is True, saves the plot. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
        """        
        samples, times = self.run_simulations(params)
        if self.complete_trajectory:
            for i in range(self.n_trajectories):
                edges = np.concatenate((times[i], self.time_windows[-1:]))
                plt.stairs(values=samples[i], edges=edges, baseline=None, orientation='vertical')
        else:
            data = pd.DataFrame(samples.transpose(), columns = [f'Abundance{i}' for i in range(self.n_trajectories)])
            data['id'] = data.index
            data['time'] = times
            data = pd.wide_to_long(data, ['Abundance'], i='time', j='id')
            seaborn.lineplot(data=data, x='time', y='Abundance')
        if targets is not None: # shape (n_targets,2)
            plt.scatter(x=targets[:,0], y=targets[:,1], marker='x', c='black', label='target values')
            plt.legend()
        plt.ylim(-0.1, plt.ylim()[1])
        if save[0]:
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

        
if __name__ == '__main__':

    from CRN2_control import propensities_production_degradation as propensities
    # from CRN4_control import propensities_bursting_gene as propensities

    crn = simulation.CRN(propensities.stoich_mat, propensities.propensities, init_state=propensities.init_state, n_fixed_params=1, n_control_params=1)
    sim = CRN_Simulations(crn, np.array([5, 10, 15, 20]), 10_000, propensities.ind_species, complete_trajectory=False, sampling_times=np.arange(21))
    sim.plot_simulations(np.array([2., 2.00432852, 0.99680416, 0.99743332, 0.6582418]), targets=np.array([[5., 1.], [10., 2.], [15., 2.], [20., 3.]]))
    # sim.plot_simulations(np.array([1., 2., 1., 0.9727892751672087, 1.000185943874426, 0.9999474975579944, 1.0006398944905717]), targets=np.array([[5., 1.], [10., 1.], [15., 1.], [20., 1.]]))
