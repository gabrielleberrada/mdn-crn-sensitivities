import simulation
import generate_data
import convert_csv
import numpy as np
from CRN2_birth_death import propensities_birth_death as propensities

# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'birth'
    datasets = {'train': 1280, 'valid': 128, 'test': 640}
    DATA_LENGTH = sum(datasets.values())

    stoich_mat = propensities.stoich_mat.reshape(2, 1)
    crn = simulation.CRN(stoichiometric_mat= stoich_mat, propensities=np.array([propensities.lambda1, propensities.lambda2]), n_params=2)
    dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=np.array([5, 10, 15, 20]))
    X, y = dataset.generate_data(data_length=DATA_LENGTH, n_trajectories=10**4, sobol_length=2., initial_state=(True, np.ones(1)))


    # writing CSV files
    somme = 0
    max_value = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[4*somme:4*(somme+value),:], f'X_{CRN_NAME}_{key}')
        convert_csv.array_to_csv(y[4*somme:4*(somme+value),:], f'y_{CRN_NAME}_{key}')
        somme += value

