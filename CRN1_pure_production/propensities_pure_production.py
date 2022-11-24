import numpy as np

def lambda1(params, x):
        return params[0]

stoich_mat = np.array([1.])
propensities = np.array([lambda1])
init_state = (0,)
ind_species = 0