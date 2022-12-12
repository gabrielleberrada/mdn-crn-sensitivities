import numpy as np

def lambda1(params, x):
        return params[0]

stoich_mat = np.expand_dims(np.array([1.]), axis=0)
propensities = np.array([lambda1])
init_state = np.array([0.])
ind_species = 0