import scipy.sparse as sp
import numpy as np
from scipy.integrate import solve_ivp
from bidict import bidict
import math
import simulation
from operator import itemgetter
from typing import Tuple

class StateSpaceEnumeration:
    """
    State space enumeration as presented in the paper 'A FSP algorithm for the stationary solution of the CME'

    Inputs:
        cl
        cr
        dim
    

    """
    def __init__(self, cr: int, dim: int):
        # we choose to always start at cl=0
        self.bijection = bidict()
        self.dim = dim
        self.cl = np.zeros(self.dim, dtype=int)
        self.cr = np.zeros(self.dim, dtype=int)
        self.cr[-1] = cr
        self.lb = self.phi(self.cl, self.dim)
        self.ub = self.phi(self.cr, self.dim)
        self.length = self.ub - self.lb + 1
    
    def phi(self, x: np.ndarray, n: int):
        """
        n>1
        """
        if n < 2:
            return x[0]
        if n == 2:
            return int((x[0]+x[1])*(x[0]+x[1]+1)/2 + x[1])
        else:
            return self.phi(np.array([self.phi(x[:-1], n-1), x[-1]]), 2)

    def phi_inverse(self, z: np.ndarray, n: int):
        if n < 2:
            return (z,)
        if n == 2:
            v = math.floor((math.sqrt(8*z+1)-1)/2)
            x2 = z - v*(v+1)/2
            return int(v - x2), int(x2)
        else:
            z1, z2 = self.phi_inverse(z, 2)
            z11, z12 = self.phi_inverse(z1, n-1)
            return z11, z12, z2
    
    def create_bijection(self):
        self.bijection[self.lb] = tuple(self.cl)
        self.bijection[self.ub] = tuple(self.cr)
        for n in range(self.lb + 1, self.ub):
            self.bijection[n] = self.phi_inverse(n, self.dim)
        print(self.bijection)


class SensitivitiesDerivation:
    def __init__(self, crn: simulation.CRN, cr: int = 4):
        #cr = somme des max des valeurs atteignables
        self.crn = crn
        self.n_params = crn.n_params
        self.n_species = crn.n_species
        self.bijection = StateSpaceEnumeration(cr, dim=self.n_species)
        self.bijection.create_bijection()
        self.entries = self.bijection.bijection.values()

    def create_B(self, index: int):
        """
        Inputs: entries : list of all possible states
                index : index of the reaction occuring for B
        
        Ouput:  the rate matrix for this reaction over the truncated state-space
        """
        d = self.bijection.bijection.inverse
        n = len(self.entries)
        stoich_mat = self.crn.stoichiometric_mat[:,index]
        propensity = self.crn.propensities[index]
        outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
        # propensity parameter is 1
        data = np.array(list(map(lambda x: propensity(np.ones(self.n_params), x), self.entries)))
        get_index = lambda key: d[key] if key in d else -1
        rows = np.array([get_index(entry) for entry in self.entries])
        columns = np.array([self.bijection.phi(output, self.n_species) for output in outputs])
        # columns = np.array([get_index(output) for output in outputs])
        compute_diags = np.vectorize(lambda i: -data[rows==i].sum())
        diags = compute_diags(np.arange(n))
        mask = (columns >= 0) & (columns < n)
        rows = rows[mask]
        columns = columns[mask]
        data = data[mask]
        # according to the paper
        B = sp.coo_matrix((data, (columns, rows)), shape=(n, n))
        B.setdiag(diags)
        B.eliminate_zeros()
        return B

    def create_A(self, params: np.ndarray):
        """
        Inputs: entries : list of all possible states
                params: list of propensity parameters
        
        Ouput:  the rate matrix A over the truncated state-space
        """
        # check types
        create_Bs = np.vectorize(lambda i: self.create_B(i))
        Bs = create_Bs(np.arange(self.n_params))
        return (Bs*params).sum()

    def constant_matrix(self, params: np.ndarray, index: int):
        """
        Inputs: entries : list of all possible states
                params: list of propensity parameters
                index : index of the reaction occuring for B
        
        Ouput:  the constant matrix in the ODEs as presented in the equation (34) 
                of the paper 'FSP-FIM approach to estimate information and optimize single-cell experiments'
        
        """
        A = self.create_A(params)
        B = self.create_B(index)
        empty = sp.coo_matrix(A.shape)
        up = sp.hstack((A, empty))
        bottom = sp.hstack((B, A))
        return sp.vstack((up, bottom))

    def solve_ode(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, index: int, t_eval: list[float]):
        """
        Solves the set of linear ODEs (34).

        Inputs: init_state: initial state for probabilities and sensitivities
                t0: starting time
                tf: final time
                entries: list of all possible states
                params: list of propensity parameters
                index: index of the reaction occuring for B
                t_eval: times at which to store the computed solutin

        Outputs: 
        """
        constant = sp.csr_matrix(self.constant_matrix(params, index))
        print(constant.toarray())
        def f(t, x): 
            return constant.dot(x)
        return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)


# testing

# def propensity1(params, x):
#     return params[0]*x[1]

# def propensity2(params, x):
#     return params[1]*x[0]

# # stoich_mats = [stoich_mat1, stoich_mat2]
# stoich_mats = np.array([[1, 0], [0, -1]])
# propensities = [propensity1, propensity2]

# crn = simulation.CRN(stoich_mats, propensities, 2)
# sens = SensitivitiesDerivation(crn)
# print(len(sens.bijection.bijection.values()))
# print(sens.solve_ode(np.array([0., 0., 0.5, 0.5, 1., 2., 0., 1., 0., 0., 0.3, 0.1]), 0., 5., np.array([2., 3.]), 0, t_eval=np.arange(5)))

# 1 specy

# def propensity1(params, x):
#     return params[0]

# def propensity2(params, x):
#     return params[1]*x[0]

# stoich_mats = np.array([1, -1]).reshape(1, 2)
# propensities = [propensity1, propensity2]

# crn = simulation.CRN(stoich_mats, propensities, 2)
# sens = SensitivitiesDerivation(crn)
# print(sens.solve_ode(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 0., 5., np.array([2., 3.]), 0, t_eval=np.arange(5)))

# A+B -> C

# def propensity1(params, x):
#     return params[0]*x[0]*x[1]


# # stoich_mats = [stoich_mat1, stoich_mat2]
# stoich_mats = np.array([-1, -1, 1]).reshape(3,1)
# propensities = [propensity1]

# crn = simulation.CRN(stoich_mats, [propensity1], 1)
# sens = SensitivitiesDerivation(crn)
# print(sens.solve_ode(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 0., 5., np.array([2.]), 0, t_eval=np.arange(5)))


# Fisher information

def fisher_information_t(probs, sensitivities):
    # sensitivities (N, N_teta)
    # probs (N,)
    inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
    pS = np.zeros_like(sensitivities)
    for l, pl in enumerate(inversed_p):
        pS[l,:] = pl * sensitivities[l,:]
    print(pS)
    return np.matmul(pS.T, sensitivities)


def fisher_information(ntime_samples, probs, sensitivities):
    # sensitivities (Nt, N, N_teta)
    # probs (N,)
    f_inf = np.zeros((sensitivities.shape[-1], sensitivities.shape[-1]))
    for t in range(ntime_samples):
        f_inf += fisher_information_t(probs[t,:], sensitivities[t,:,:])
    return f_inf

# test

# probs = np.array([0., 0.4, 1.])
# sensitivities = np.arange((6.)).reshape(3,2)
# sensitivities[1,1] = 10
# print('sensitivities', sensitivities)
# print(fisher_information_t(probs, sensitivities))


