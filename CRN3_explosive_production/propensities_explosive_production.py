import numpy as np

def lambda1(params, x):
        return params[0]*x[0]

stoich_mat = np.array([1.])

propensities = np.array([lambda1])

init_state = np.array([5])