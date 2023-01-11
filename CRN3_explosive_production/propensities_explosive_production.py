import numpy as np

def lambda1(params, x):
        return params[0]*x[0]

def lambda1_drv0(params, x):
        return x[0]

stoich_mat = np.expand_dims(np.array([1.]), axis=0)

propensities = np.array([lambda1])

init_state = np.array([5.])

ind_species = 0

def zeros(params, x):
    return 0

N_REACTIONS = 1
N_PARAMS = 1
propensities_drv = np.array([zeros]*(N_REACTIONS*N_PARAMS)).reshape((N_REACTIONS, N_PARAMS))
propensities_drv[0, 0] = lambda1_drv0