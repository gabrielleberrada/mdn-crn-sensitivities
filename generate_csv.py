import simulation
import generate_data
import convert_csv
import numpy as np
from CRN5_isomeric_pd import propensities
from typing import Union, Tuple

def generate_csv(crn_name: str,
                datasets: dict,
                n_params: int,
                stoich_mat: np.ndarray,
                propensities: np.ndarray,
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
    crn = simulation.CRN(stoichiometry_mat=stoich_mat, propensities=propensities, n_params=n_params)
    dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=sampling_times, ind_species=ind_species, method=method)
    X, y = dataset.generate_data(data_length=data_length, 
                                n_trajectories=n_trajectories, 
                                sobol_start=sobol_start, 
                                sobol_end=sobol_end,
                                ind_species=ind_species, 
                                initial_state=initial_state)
    # writing CSV files
    somme = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[n_times*somme:n_times*(somme+value),:], f'X_{crn_name}_{key}')
        convert_csv.array_to_csv(y[n_times*somme:n_times*(somme+value),:], f'y_{crn_name}_{key}')
        somme += value
    print('done')


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'CRN5'
    datasets = {'test': 16}
    # datasets = {'train1': 5_094, 'train2': 5_095, 'train3': 5_095, 'valid1': 200, 'valid2': 200, 'valid3': 200, 'test': 502}
    N_PARAMS = 6
    generate_csv(crn_name=CRN_NAME,
                datasets=datasets,
                n_params=N_PARAMS,
                stoich_mat=propensities.stoich_mat, # shape (n_species, n_reactions)
                propensities=propensities.propensities,
                sampling_times=np.array([5, 10, 15, 20]),
                ind_species=propensities.ind_species,
                n_trajectories=10**4,
                sobol_start=np.array([0.]),
                sobol_end=np.array([1., 1., 1., 0.5, 2., 0.5]),
                initial_state=(True, list(propensities.init_state)))

