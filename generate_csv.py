import simulation
import generate_data
import convert_csv
import numpy as np
# from CRN1_control import propensities_pure_production as propensities
# from CRN3_control import propensities_explosive_production as propensities
# from CRN2_control import propensities_production_degradation as propensities
# from CRN2_control import propensities_multiple_params as propensities
from CRN6_toggle_switch import propensities_toggle as propensities
from typing import Tuple

def generate_csv_datasets(crn_name: str,
                          datasets: dict,
                          n_fixed_params: int,
                          n_control_params: int,
                          stoich_mat: np.ndarray,
                          propensities: np.ndarray,
                          time_windows: np.ndarray,
                          sampling_times: np.ndarray,
                          ind_species: int,
                          n_trajectories: int,
                          sobol_start: np.ndarray,
                          sobol_end: np.ndarray,
                          initial_state: Tuple[bool, np.ndarray] =(False, None),
                          method: str ='SSA'):
    r"""Generates datasets from Stochastic Simulations and saves them in CSV files.

    Args:
        - **crn_name** (str): Name of the CRN to use for the files names.
        - **datasets** (dict): Dictionary whose keys are the names of the datasets and whose values are the corresponding lengths.
        - **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
        - **n_control_params** (int): Number of varying parameters required to define the propensity functions.
          Their values vary from a time window to another.
        - **stoich_mat** (np.ndarray): Stoichiometry matrix.
        - **propensities** (np.ndarray): Non-parameterized propensity functions.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_T` must match
          with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - **sampling_times** (np.ndarray): Sampling times.
        - **ind_species** (int): Index of the species of interest.
        - :math:`n_{trajectories}` (int): Number of trajectories to compute to estimate the distribution for each set of parameters.
        - **sobol_start** (np.ndarray): Lower boundaries of the parameters samples. Shape :math:`(n_total_params})`.
        - **sobol_end** (np.ndarray): Upper boundaries of the parameters samples. Shape :math:`(n_total_params)`.
        - **initial_state** (Tuple[bool, np.ndarray], optional): Initial state of the species. Defaults to (False, None). In this case,
          sets the initial state to :math:`0` for all species.
        - **method** (str): Stochastic Simulation to compute. Defaults to 'SSA'.
    """                         
    data_length = sum(datasets.values())
    n_times = len(sampling_times)
    if initial_state[0]:
        init_state = initial_state[1]
    else:
        init_state = np.zeros(np.shape(stoich_mat)[0], dtype=np.float32)
    crn = simulation.CRN(stoichiometry_mat=stoich_mat,
                        propensities=propensities, 
                        init_state=init_state,
                        n_fixed_params=n_fixed_params, 
                        n_control_params=n_control_params)
    dataset = generate_data.CRN_Dataset(crn=crn,
                                        time_windows=time_windows,
                                        sampling_times=sampling_times, 
                                        ind_species=ind_species, 
                                        method=method)
    X, y = dataset.generate_data(data_length=data_length, 
                                n_trajectories=n_trajectories, 
                                sobol_start=sobol_start, 
                                sobol_end=sobol_end)
    # writing CSV files
    somme = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[n_times*somme:n_times*(somme+value),:], f'X_{crn_name}_{key}')
        convert_csv.array_to_csv(y[n_times*somme:n_times*(somme+value),:], f'y_{crn_name}_{key}')
        somme += value


def generate_csv_simulations(crn_name: str,
                              n_fixed_params: int,
                              n_control_params: int,
                              stoich_mat: np.ndarray,
                              propensities: np.ndarray,
                              time_windows: np.ndarray,
                              sampling_times: np.ndarray,
                              ind_species: int,
                              n_trajectories: int,
                              params: np.ndarray,
                              initial_state: np.ndarray,
                              method: str ='SSA'):
    r"""Generates simulations of the abundance evolution of a species from Stochastic Simulations
    and saves them in CSV files.

    Args:
        - **crn_name** (str): Name of the CRN to use for the file name.
        - **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
        - **n_control_params** (int): Number of varying parameters required to define the propensity functions.
          Their values vary from a time window to another.
        - **stoich_mat** (np.ndarray): Stoichiometry matrix.
        - **propensities** (np.ndarray): Non-parameterized propensity functions.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_T]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{T-1}, t_T]`. :math:`t_T` must match
          with the final time :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - **sampling_times** (np.ndarray): Sampling times.
        - **ind_species** (int): Index of the species of interest.
        - :math:`n_{trajectories}` (int): Number of trajectories to compute to estimate the distribution for each set of parameters.
        - **params** (np.ndarray): Parameters used to run simulations.
        - **initial_state** (Tuple[bool, np.ndarray], optional): Initial state of the species. Defaults to (False, None). In this case,
          sets the initial state to :math:`0` for all species.
        - **method** (str): Stochastic Simulation to compute. Defaults to 'SSA'.
    """
    crn = simulation.CRN(stoichiometry_mat=stoich_mat,
                        propensities=propensities, 
                        init_state=initial_state,
                        n_fixed_params=n_fixed_params, 
                        n_control_params=n_control_params)
    dataset = generate_data.CRN_Simulations(crn=crn,
                                            time_windows=time_windows,
                                            n_trajectories=n_trajectories,
                                            ind_species=ind_species,
                                            complete_trajectory=False,
                                            sampling_times=sampling_times, 
                                            method=method)
    samples, _ = dataset.run_simulations(params=params)
    # writing CSV files
    convert_csv.array_to_csv(samples, f'Distributions_{crn_name}')


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'toggle'
    datasets = {'test': 16}
    # datasets = {'train1': 2464, 'train2': 2464, 'train3': 2464, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 500}
    # datasets = {'train1': 10656, 'train2': 10656, 'train3': 10656, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 500}
    N_PARAMS = 10
    generate_csv_datasets(crn_name=CRN_NAME,
                          datasets=datasets,
                          n_fixed_params=N_PARAMS-1,
                          n_control_params=1,
                          stoich_mat=propensities.stoich_mat, # shape (n_species, n_reactions)
                          propensities=propensities.propensities,
                          time_windows=np.array([5, 10, 15, 20]),
                          sampling_times=np.array([5, 10, 15, 20]),
                          ind_species=propensities.ind_species,
                          n_trajectories=10**4,
                          sobol_start=np.zeros(N_PARAMS),
                          sobol_end=np.array([1., 1., 1., 1., 1., 1., 3., 3., 1., 1.]),
                          initial_state=(True, propensities.init_state))

