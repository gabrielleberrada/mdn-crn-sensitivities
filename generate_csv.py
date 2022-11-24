import simulation
import generate_data
import convert_csv
import numpy as np
from CRN4_bursting_gene import propensities_bursting_gene as propensities
from typing import Union, Tuple

def generate_csv(crn_name: str,
                datasets: dict,
                n_params: int,
                stoich_mat: np.ndarray,
                propensities: np.ndarray,
                sampling_times: list[float],
                ind_species: int,
                n_trajectories: int,
                sobol_start: Union[float, list[float]],
                sobol_end: Union[float, list[float]],
                initial_state: Tuple[bool, np.ndarray] =(False, None),
                method: str ='SSA'):
    """Generates datasets from Stochastic Simulations and save them in CSV files.

    Args:
        - **crn_name** (str): Name of the CRN to use for the filenames.
        - **datasets** (dict): Dictionary whose keys are the names of the datasets and whose values are the corresponding lengths.
        - **n_params** (int): Number of parameters for this CRN.
        - **stoich_mat** (np.ndarray): Stoichiometry matrix.
        - **propensities** (np.ndarray): Non-parameterized propensity functions.
        - **sampling_times** (list[float]): Sampling times.
        - **ind_species** (int): Index of the species to study.
        - :math:`n_{trajectories}` (int): Number of trajectories to compute to estimate the distribution for each set of parameters.
        - **sobol_start** (Union[float, list[float]]): Lower boundary of the parameters samples.
        - **sobol_end** (Union[float, list[float]]): Upper boundary of the parameters samples.
        - **initial_state** (Tuple[bool, np.ndarray], optional): Initial state of the species. Defaults to (False, None).
        - **method** (str): Stochastic Simulation to compute. Defaults to 'SSA'.
    """                         
    data_length = sum(datasets.values())
    n_times = len(sampling_times)
    crn = simulation.CRN(stoichiometric_mat=stoich_mat, propensities=propensities, n_params=n_params)
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

    CRN_NAME = 'CRN_bursting_gene'
    datasets = {'train1': 1_100, 'train2': 1_100, 'train3': 1_100, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 496}
    N_PARAMS = 4
    generate_csv(crn_name=CRN_NAME,
                datasets=datasets,
                n_params=N_PARAMS,
                stoich_mat=propensities.stoich_mat, # shape (n_species, n_reactions)
                propensities=propensities.propensities,
                sampling_times=np.array([5, 10, 15, 20]),
                ind_species=0,
                n_trajectories=10**4,
                sobol_start=[0.],
                sobol_end=[1., 3., 5., 0.05],
                initial_state=(True, np.array([1, 2])))

