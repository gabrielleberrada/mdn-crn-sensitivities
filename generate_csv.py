import simulation
import generate_data
import convert_csv
import numpy as np
from CRN2_production_degradation import propensities_production_degradation as propensities
from typing import Union

def generate_csv(crn_name: str,
                datasets: dict,
                n_params: int,
                init_state: np.ndarray[int],
                stoich_mat: np.ndarray,
                propensities: np.ndarray,
                sampling_times: list[float],
                ind_species: int,
                n_trajectories: int,
                sobol_up_bounds: Union[float, np.ndarray[float]],
                sobol_low_bounds: Union[float, np.ndarray[float]]):
    """Generate datasets from SSA simulations and save them in CSV files.

    Args:
        - **crn_name** (str): Name of the CRN for the filenames.
        - **datasets** (dict): Dictionary whose keys are the name of the datasets and whose values are the corresponding length.
        - **n_params** (int): Number of parameters for this CRN.
        - **init_state** (np.ndarray[int]): Initial state.
        - **stoich_mat** (np.ndarray): Stoichiometric matrix of the CRN.
        - **propensities** (np.ndarray): Non-parameterized propensities of the CRN.
        - **sampling_times** (list[float]): Sampling times.
        - **ind_species** (int): Index of species to study
        - **n_trajectories** (int): Number of trajectories to do for each set of parameters
        - **sobol_up_bounds** (Union[float, np.ndarray[float]]): Upper boundaries of the generated parameters.
        - **sobol_low_bounds** (Union[float, np.ndarray[float]]): Lower boundaries of the generated parameters.
    """
    data_length = sum(datasets.values())
    crn = simulation.CRN(init_state=init_state, stoichiometric_mat=stoich_mat, propensities=propensities, n_params=n_params)
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

    CRN_NAME = 'TEST'
    datasets = {'test': 4}
    # datasets = {'train1': 1_100, 'train2': 1_100, 'train3': 1_100, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 496}
    N_PARAMS = 2
    generate_csv(crn_name=CRN_NAME,
                datasets=datasets,
                n_params=N_PARAMS,
                init_state=propensities.init_state,
                stoich_mat=propensities.stoich_mat.reshape((1,2)), # shape (n_species, n_reactions)
                propensities=propensities.propensities,
                sampling_times=np.array([5, 10, 15, 20]),
                ind_species=0,
                n_trajectories=10**4,
                sobol_up_bounds=np.array([1., 2.]),
                sobol_low_bounds=0.)

