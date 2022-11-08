import simulation
import generate_data
import convert_csv
import numpy as np
from CRN4_bursting_gene import propensities_bursting_gene as propensities
from typing import Union

def generate_csv(crn_name: str,
                datasets: dict,
                n_params: int,
                stoich_mat: np.ndarray,
                propensities: np.ndarray,
                sampling_times: list[float],
                ind_species: int,
                n_trajectories: int,
                sobol_up_bounds: Union[float, np.ndarray[float]],
                sobol_low_bounds: Union[float, np.ndarray[float]]):
    """Generate datasets from SSA simulations and save them in CSV files.

    Args:
        - **crn_name** (str): name of the CRN for the filenames.
        - **datasets** (dict): dictionary whose keys are the name of the datasets and whose values are the corresponding length.
        - **n_params** (int): number of parameters for this CRN.
        - **stoich_mat** (np.ndarray): stoichiometric matrix of the CRN.
        - **propensities** (np.ndarray): non-parameterized propensities of the CRN.
        - **sampling_times** (list[float]): sampling times.
        - **ind_species** (int): index of species to study
        - **n_trajectories** (int): number of trajectories to do for each set of parameters
        - **sobol_up_bounds** (Union[float, np.ndarray[float]]): upper boundaries of the generated parameters.
        - **sobol_low_bounds** (Union[float, np.ndarray[float]]): lower boundaries of the generated parameters.
    """                         
    data_length = sum(datasets.values())
    crn = simulation.CRN(stoichiometric_mat=stoich_mat, propensities=propensities, n_params=n_params)
    dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=sampling_times, ind_species=ind_species)
    X, y = dataset.generate_data(data_length=data_length, n_trajectories=n_trajectories, sobol_end=sobol_up_bounds, sobol_start=sobol_low_bounds, ind_species=ind_species)
    # writing CSV files
    somme = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[4*somme:4*(somme+value),:], f'X_{crn_name}_{key}')
        convert_csv.array_to_csv(y[4*somme:4*(somme+value),:], f'y_{crn_name}_{key}')
        somme += value
    print('done')


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'bursting_gene'
    datasets = {'train1': 1_100, 'train2': 1_100, 'train3': 1_100, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 496}
    # shape (n_species, n_reactions)
    stoich_mat = propensities.stoich_mat #.reshape(1, 1)
    N_PARAMS = 4
    generate_csv(crn_name=CRN_NAME,
                datasets=datasets,
                n_params=N_PARAMS,
                stoich_mat=stoich_mat,
                propensities=propensities.propensities,
                sampling_times=np.array([5, 10, 15, 20]),
                ind_species=1,
                n_trajectories=10**4,
                sobol_up_bounds=np.array([1., 3., 5., 0.05]),
                sobol_low_bounds=[0.],)

