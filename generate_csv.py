import simulation
import generate_data
import convert_csv
import numpy as np
from CRN4_bursting_gene import propensities_bursting_gene as propensities


# because we use multiprocessing
if __name__ == '__main__':

    CRN_NAME = 'bursting_gene'
    datasets = {'train1': 1_100, 'train2': 1_100, 'train3': 1_100, 'valid1': 100, 'valid2': 100, 'valid3': 100, 
                'train4': 1_100, 'train5': 1_100, 'train6': 1_100, 'valid4': 100, 'valid5': 100, 'valid6': 100, 'test': 992}
    # datasets = {'train1': 900, 'valid1': 100, 'test': 24}
    DATA_LENGTH = sum(datasets.values())
    N_PARAMS = 1

    # shape (n_species, n_reactions)
    stoich_mat = propensities.stoich_mat #.reshape(1, 1)
    crn = simulation.CRN(stoichiometric_mat= stoich_mat, propensities=propensities.propensities, n_params=N_PARAMS)
    dataset = generate_data.CRN_Dataset(crn=crn, sampling_times=np.array([5, 10, 15, 20]), ind_species = 1)
    X, y = dataset.generate_data(data_length=DATA_LENGTH, n_trajectories=10**4, sobol_end=np.array([20]), ind_species=1)

    # writing CSV files
    somme = 0
    max_value = 0
    for key, value in datasets.items():
        convert_csv.array_to_csv(X[4*somme:4*(somme+value),:], f'X_{CRN_NAME}_{key}')
        convert_csv.array_to_csv(y[4*somme:4*(somme+value),:], f'y_{CRN_NAME}_{key}')
        somme += value
    print('done')

