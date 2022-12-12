import numpy as np

def lambda1(params, x):
        return params[0]*x[0]

def lambda2(params, x):
        return params[1]*x[0]

stoich_mat = np.expand_dims(np.array([1., 1.]), axis=0)

propensities = np.array([lambda1, lambda2])

init_state = np.array([5.])

ind_species = 0