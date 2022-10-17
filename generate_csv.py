import simulation
import generate_data
import convert_csv
import numpy as np
from CRN3_birth import propensities_birth as propensities


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'birth'
    datasets = {'train1': 1_100, 'train2': 1_100, 'train3': 1_100, 'valid1': 100, 'valid2': 100, 'valid3': 100, 'test': 496}
    # datasets = {'train1': 900, 'valid1': 100, 'test': 24}
    DATA_LENGTH = sum(datasets.values())

    # shape (n_species, n_reactions)
    stoich_mat = propensities.stoich_mat.reshape(1, 1)
    crn = simulation.CRN(stoichiometric_mat= stoich_mat, propensities=np.array([propensities.lambda1]), n_params=1)
    dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=np.array([5, 10, 15, 20]))
    X, y = dataset.generate_data(data_length=DATA_LENGTH, n_trajectories=10**4, sobol_length=np.array([0.2]), initial_state=(True, 5*np.ones(1)))


    # writing CSV files
    somme = 0
    max_value = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[4*somme:4*(somme+value),:], f'X_{CRN_NAME}_{key}')
        convert_csv.array_to_csv(y[4*somme:4*(somme+value),:], f'y_{CRN_NAME}_{key}')
        somme += value
    print('done')

