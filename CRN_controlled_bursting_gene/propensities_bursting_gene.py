import numpy as np

# R1
def lambda1(params, x):
    return params[0]*(1-x[0])

def lambda1_drv0(params, x):
    return 1-x[0]

# R2
def lambda2(params, x):
    return params[1]*x[0]

def lambda2_drv1(params, x):
    return x[0]

# R3
def lambda3(params, x):
    return params[2]*x[1]

def lambda3_drv2(params, x):
    return x[1]

# R4 (controlled)
def lambda4(params, x):
    return params[3]*x[0]

def lambda4_drv3(params, x):
    return x[0]


propensities = np.array([lambda1, lambda2, lambda3, lambda4])
stoich_mat = np.array([[1., 0.], [0., 1.], [0., -1.], [-1., 0.]]).T
init_state = np.array([0., 0.])
ind_species = 1

def zeros(params, x):
    return 0

N_REACTIONS = 4
N_PARAMS = 4
propensities_drv = np.array([zeros]*(N_REACTIONS*N_PARAMS)).reshape((N_REACTIONS, N_PARAMS))

propensities_drv[0, 0] = lambda1_drv0
propensities_drv[1, 1] = lambda2_drv1
propensities_drv[2, 2] = lambda3_drv2
propensities_drv[3, 3] = lambda4_drv3
