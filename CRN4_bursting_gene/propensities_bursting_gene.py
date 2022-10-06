import numpy as np

# R1
def lambda1(params, x):
    pass

# R2
def lambda2(params, x):
    pass

# R3
def lambda3(params, x):
    return params[2]

# R4
def lambda4(params, x):
    return params[3]*x[2]

# g_on, g_off, RNA
stoich_mat = np.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.], [0., 0., -1.]])