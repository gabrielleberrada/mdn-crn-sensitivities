import numpy as np

# Parameters
# [b_x, b_y, k_x, k_y, alpha_xy, alpha_yx, eta_xy, eta_yx, gamma_x, gamma_y]

def zeros(params, x):
    return 0

# R1
def lambda1(params, x):
    return params[0] + params[2]/(1+params[5]*x[1]**params[7])

def lambda1_drv0(params, x):
    return 1


def lambda1_drv2(params, x):
    return 1/(1+params[5]*x[1]**params[7])

def lambda1_drv5(params, x):
    return -params[2]*x[1]**params[7]/(1+params[5]*x[1]**params[7])**2

def lambda1_drv7(params, x):
    if x[1] == 0:
        return 0
    return -params[2]*params[5]*params[7]*x[1]**(params[7]-1)/(1+params[5]*x[1]**params[7])**2

# R2
def lambda2(params, x):
    return params[8]*x[0]

def lambda2_drv8(params, x):
    return x[0]

# R3
def lambda3(params, x):
    return params[1] + params[3]/(1 + params[4]*x[0]**params[6])

def lambda3_drv1(params, x):
    return 1

def lambda3_drv3(params, x):
    return 1/(1 + params[4]*x[0]**params[6])

def lambda3_drv4(params, x):
    return -params[3]*x[0]**params[6]/(1+params[4]*x[0]**params[6])**2

def lambda3_drv6(params, x):
    if x[0] == 0:
        return 0
    return -params[3]*params[4]*params[6]*x[0]**(params[6]-1)/(1+params[4]*x[0]**params[6])**2

# R4
def lambda4(params, x):
    return params[9]*x[1]

def lambda4_drv9(params, x):
    return x[1]

propensities = np.array([lambda1, lambda2, lambda3, lambda4])
stoich_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]).T
init_state = np.array([0, 0])
ind_species = 1


N_REACTIONS = 4
N_PARAMS = 10
propensities_drv = np.array([zeros]*(N_REACTIONS*N_PARAMS)).reshape((N_REACTIONS, N_PARAMS))
propensities_drv[0, 0] = lambda1_drv0
propensities_drv[0, 2] = lambda1_drv2
propensities_drv[0, 5] = lambda1_drv5
propensities_drv[0, 7] = lambda1_drv7

propensities_drv[1, 8] = lambda2_drv8

propensities_drv[2, 1] = lambda3_drv1
propensities_drv[2, 3] = lambda3_drv3
propensities_drv[2, 4] = lambda3_drv4
propensities_drv[2, 6] = lambda3_drv6

propensities_drv[3, 9] = lambda4_drv9