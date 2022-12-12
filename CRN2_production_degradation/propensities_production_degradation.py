import numpy as np

# birth R1
def lambda1(params, x):
    return params[0]

# death R2
def lambda2(params, x):
    return params[1]*x[0]

stoich_mat = np.expand_dims(np.array([1., -1.]), axis=0)
propensities = np.array([lambda1, lambda2])
init_state = np.array([0])
ind_species = 0