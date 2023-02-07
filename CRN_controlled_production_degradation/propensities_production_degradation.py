import numpy as np

# birth R1
def lambda1(params, x):
    return params[0]

def lambda1_drv0(params, x):
    return 1

# death R2
def lambda2(params, x):
    return params[1]*x[0]

def lambda2_drv1(params, x):
    return x[0]

stoich_mat = np.expand_dims(np.array([1., -1.]), axis=0)
propensities = np.array([lambda1, lambda2])
init_state = np.array([0.])
ind_species = 0

def zeros(params, x):
    return 0

N_REACTIONS = 2
N_PARAMS = 2
propensities_drv = np.array([zeros]*(N_REACTIONS*N_PARAMS)).reshape((N_REACTIONS, N_PARAMS))
propensities_drv[0, 0] = lambda1_drv0
propensities_drv[1, 1] = lambda2_drv1