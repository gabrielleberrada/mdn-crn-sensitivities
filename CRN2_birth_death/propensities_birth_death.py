import numpy as np

# birth R1
def lambda1(params, x):
    return params[0]

# death R2
def lambda2(params, x):
    return params[1]*x[0]

stoich_mat = np.array([1., -1.])