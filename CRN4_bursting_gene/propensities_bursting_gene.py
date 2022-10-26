import numpy as np

# x = (g_on, RNA)
# kon = 0.05*alpha
# koff = 0.15*alpha
# kr = 5
# gamma = 0.05

# R1
def lambda1(params, x):
    return 0.05*params[0]*(1-x[0])

# R2
def lambda2(params, x):
    return 0.15*params[0]*x[0]

# R3
def lambda3(params, x):
    return 5*x[0]

# R4
def lambda4(params, x):
    return 0.05*x[1]

propensities = np.array([lambda1, lambda2, lambda3, lambda4])
# g_on, g_off, RNA
stoich_mat = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]).T