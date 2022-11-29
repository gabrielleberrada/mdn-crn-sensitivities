import simulation
import generate_data
import convert_csv
import numpy as np
from CRN2_production_degradation import propensities_production_degradation as propensities
from typing import Union, Tuple

def generate_csv(crn_name: str,
                datasets: dict,
                n_params: int,
                n_control_params: int,
                stoich_mat: np.ndarray,
                propensities: np.ndarray,
                time_slots: list,
                sampling_times: list,
                ind_species: int,
                n_trajectories: int,
                sobol_start: Union[float, np.ndarray],
                sobol_end: Union[float, np.ndarray],
                initial_state: Tuple[bool, np.ndarray] =(False, None),
                method: str ='SSA'):
    """Generates datasets from Stochastic Simulations and save them in CSV files.

    Args:
        - **crn_name** (str): Name of the CRN to use for the filenames.
        - **datasets** (dict): Dictionary whose keys are the names of the datasets and whose values are the corresponding lengths.
        - **n_params** (int): Number of parameters for this CRN.
        - **stoich_mat** (np.ndarray): Stoichiometry matrix.
        - **propensities** (np.ndarray): Non-parameterized propensity functions.
        - **sampling_times** (list): Sampling times.
        - **ind_species** (int): Index of the species to study.
        - :math:`n_{trajectories}` (int): Number of trajectories to compute to estimate the distribution for each set of parameters.
        - **sobol_start** (Union[float, np.ndarray]): Lower boundary of the parameters samples.
        - **sobol_end** (Union[float, np.ndarray]): Upper boundary of the parameters samples.
        - **initial_state** (Tuple[bool, np.ndarray], optional): Initial state of the species. Defaults to (False, None).
        - **method** (str): Stochastic Simulation to compute. Defaults to 'SSA'.
    """                         
    data_length = sum(datasets.values())
    n_times = len(sampling_times)
    if initial_state[0]:
        init_state = initial_state[1]
    else:
        init_state = np.zeros(np.shape(stoich_mat)[0], dtype=np.float32)
    crn = simulation.CRN(stoich_mat, propensities, n_params, init_state, n_control_params)
    dataset = generate_data.CRN_Dataset(crn=crn,
                                        time_slots=time_slots,
                                        sampling_times=sampling_times, 
                                        ind_species=ind_species, 
                                        method=method)
    X, y = dataset.generate_data(data_length=data_length, 
                                n_trajectories=n_trajectories, 
                                sobol_start=sobol_start, 
                                sobol_end=sobol_end,
                                ind_species=ind_species)
    # writing CSV files
    somme = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[n_times*somme:n_times*(somme+value),:], f'X_{crn_name}_{key}')
        convert_csv.array_to_csv(y[n_times*somme:n_times*(somme+value),:], f'y_{crn_name}_{key}')
        somme += value
    print('done')


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'CRN2_0'
    datasets = {'test': 16}
    # datasets = {'train1': 5_094, 'train2': 5_095, 'train3': 5_095, 'valid1': 200, 'valid2': 200, 'valid3': 200, 'test': 500}
    N_PARAMS = 2
    generate_csv(crn_name=CRN_NAME,
                datasets=datasets,
                n_params=N_PARAMS,
                n_control_params=0,
                stoich_mat=np.expand_dims(propensities.stoich_mat, axis=0), # shape (n_species, n_reactions)
                propensities=propensities.propensities,
                time_slots=np.array([7, 15, 20]), # cannot be empty for now, needs to include final time
                sampling_times=np.array([5, 10, 15, 20]),
                ind_species=propensities.ind_species,
                n_trajectories=10**4,
                sobol_start=np.array([0., 0.]),
                sobol_end=np.array([2., 1.]),
                initial_state=(True, list(propensities.init_state)))

