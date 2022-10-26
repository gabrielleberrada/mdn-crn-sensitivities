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

    def phi_inverse(self, z: int, n: int):
        if n < 2:
            return (z,)
        elif n == 2:
            v = math.floor((math.sqrt(8*z+1)-1)/2)
            x2 = z - v*(v+1)/2
            return int(v - x2), int(x2)
        else:
            z1, z2 = self.phi_inverse(z, 2)
            z11 = self.phi_inverse(z1, n-1)
            return z11 + (z2,)
    
    def create_bijection(self):
        self.bijection[self.lb] = tuple(self.cl)
        self.bijection[self.ub] = tuple(self.cr)
        for z in range(self.lb, self.ub):
            self.bijection[z] = self.phi_inverse(z, self.dim)
        # print(self.bijection)


class SensitivitiesDerivation:
    def __init__(self, crn: simulation.CRN, cr: int =4):
        #cr = somme des max des valeurs atteignables
        self.cr = cr
        self.crn = crn
        self.n_params = crn.n_params
        self.n_reactions = crn.n_reactions
        self.n_species = crn.n_species
        self.bijection = StateSpaceEnumeration(cr, dim=self.n_species)
        self.bijection.create_bijection()
        self.entries = self.bijection.bijection.values()
        self.n_states = len(self.entries)

    def create_B(self, index: int):
        """
        Inputs: entries : list of all possible states
                index : index of the reaction occuring for B
        
        Ouput:  the rate matrix for this reaction over the truncated state-space
        """
        d = self.bijection.bijection.inverse
        n = self.n_states
        stoich_mat = self.crn.stoichiometric_mat[:,index]
        propensity = self.crn.propensities[index]
        outputs = list(map(lambda entry: tuple(entry + stoich_mat), self.entries))
        # propensity parameter is 1
        data = np.array(list(map(lambda x: propensity(np.ones(self.n_params), x), self.entries)))
        get_index = lambda key: d[key] if key in d else -1
        rows = np.array([get_index(entry) for entry in self.entries])
        columns = np.array([self.bijection.phi(output, self.n_species) for output in outputs])
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
        # with np.printoptions(threshold=np.inf):
        #     print(index, B.toarray().T)        
        # if index == 0:
        #     with np.printoptions(threshold=np.inf):
        #         print(A.toarray().T)
        empty = sp.coo_matrix(A.shape)
        up = sp.hstack((A, empty))
        bottom = sp.hstack((B, A))
        return sp.vstack((up, bottom))

    def solve_ode(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, index: int, t_eval: list[float]):
        """
        Solves the set of linear ODEs (34).

        Inputs: init_state: initial state for probabilities and sensitivities.
                            The length of the initial state must be 2*(Cr*(Cr+3)/2+1) if n>= 2, else 2*(Cr+1)
                t0: starting time
                tf: final time
                entries: list of all possible states
                params: list of propensity parameters
                index: index of the reaction occuring for B
                t_eval: times at which to store the computed solution

        Outputs: 
        """
        constant = sp.csr_matrix(self.constant_matrix(params, index))
        # if index == 2:
        #     print(index, constant.toarray())
        def f(t, x):
            return constant.dot(x)
        return solve_ivp(f, (t0, tf), init_state, t_eval=t_eval)


    def get_sensitivities(self, init_state: np.ndarray, t0: float, tf: float, params: np.ndarray, t_eval: list[float]):
        """
        Inputs: init_state: array of dimensions N_tetax2N such that 
                            init_state[i,:] is the initial states for the probabilities and sensitivities of the i-th reaction.
                            The length of each initial state vector must be 2*(Cr*(Cr+3)/2+1) if n>= 2, else 2*(Cr+1)
                t0: starting time
                tf: final time
                entries: list of all possible states
                params: list of propensity parameters
                t_eval: times at which to store the computed solution

        Outputs: the probabilities vector and sensitivities matrix for each time points.
        """
        sensitivities = []
        for i in range(self.n_reactions):
            solution = self.solve_ode(init_state[i,:], t0, tf, params, i, t_eval)['y']
            # print(solution[:self.n_states,:])
            sensitivities.append(solution[self.n_states:,:].T)
        # print(solution[:self.n_states,:][3], solution[:self.n_states, :][4], solution[:self.n_states,:][5])
        probs = solution[:self.n_states,:].T
        return probs, np.stack(sensitivities, axis=-1)


# Correction get_sensitivities



# stoich_mat = np.array([[-2, 2], 
#                         [2, -2],
#                         [1, 0],
#                         [-1, 0],
#                         [0, 1],
#                         [0, -1]]).T

# print(stoich_mat[0, 2], stoich_mat[1, 2])

# def lambda1(params, x):
#     return params[0]*x[0]*(x[0]-1)

# def lambda2(params, x):
#     return params[1]*x[1]*(x[1]-1)

# def lambda3(params, x):
#     return params[2]

# def lambda4(params, x):
#     return params[3]*x[0]

# def lambda5(params, x):
#     return params[4]

# def lambda6(params, x):
#     return params[5]*x[1]

# propensities = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6])
# crn = simulation.CRN(stoich_mat, propensities, 6)
# cr = 3
# stv_calculator = SensitivitiesDerivation(crn, cr)
# n_cr = int(cr*(cr+3)/2+1)
# init_state_p = np.zeros(2*n_cr)
# init_state_p[0] = 1
# init_state = np.stack([init_state_p]*crn.n_reactions)
# t = 1
# params = np.arange(6)
# probs, _ = stv_calculator.get_sensitivities(init_state, 0., t, params, t_eval=[t])
# print(probs[0,0], probs[0,3], probs[0,5])
# print(stv_calculator.bijection.bijection)




# testing

# def propensity1(params, x):
#     return params[0]*x[1]

# def propensity2(params, x):
#     return params[1]*x[0]

# stoich_mats = [stoich_mat1, stoich_mat2]
# stoich_mats = np.array([[1, 0], [0, -1]])
# propensities = [propensity1, propensity2]

# crn = simulation.CRN(stoich_mats, propensities, 2)
# sens = SensitivitiesDerivation(crn, 3)
# print(sens.solve_ode(np.array([0., 0., 0.5, 0.5, 1., 2., 0., 1., 0., 0., 0.3, 0.1]), 0., 5., np.array([2., 3.]), 0, t_eval=np.arange(5)))
# print(sens.solve_ode(np.zeros(20), 0., 5., np.array([2., 3.]), 0, t_eval=np.arange(5)))

# 1 specy

# def propensity1(params, x):
#     return params[0]

# def propensity2(params, x):
#     return params[1]*x[0]

# stoich_mats = np.array([1, -1]).reshape(1, 2)
# propensities = [propensity1, propensity2]

# crn = simulation.CRN(stoich_mats, propensities, 2)
# sens = SensitivitiesDerivation(crn)
# print((sens.solve_ode(np.zeros(10), 0., 5., np.array([2., 3.]), 0, t_eval=np.arange(5)))['y'].shape)
# print(sens.get_sensitivities(np.zeros(10), 0., 5., np.array([2., 3.]), t_eval=np.arange(3))[0].shape)

# A+B -> C

# def propensity1(params, x):
#     return params[0]*x[0]*x[1]


# # stoich_mats = [stoich_mat1, stoich_mat2]
# stoich_mats = np.array([-1, -1, 1]).reshape(3,1)
# propensities = [propensity1]

# crn = simulation.CRN(stoich_mats, [propensity1], 1)
# sens = SensitivitiesDerivation(crn, cr=4)
# print(sens.solve_ode(np.zeros(14), 0., 5., np.array([2.]), 0, t_eval=np.arange(5)))






# Fisher information

def fisher_information_t(probs: np.ndarray, sensitivities: np.ndarray):
    '''
    Computes the Fisher Information Matrix at a single time point.

    Inputs: 'probs': the probability vector which has dimension N
            'sensitivities': the sensitivity matrix which has dimensions NxN_teta,
                            where N is the number of species and N_teta is the number of parameters
    
    Outputs: The Fisher Information Matrix for the model evaluated at a single time point.
    '''
    # sensitivities (N, N_teta)
    # probs (N,)
    inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
    pS = np.zeros_like(sensitivities)
    for l, pl in enumerate(inversed_p):
        pS[l,:] = pl * sensitivities[l,:]
    return np.matmul(pS.T, sensitivities)


def fisher_information(ntime_samples: int, probs: np.ndarray, sensitivities:np.ndarray):
    '''
    Computes the Fisher Information Matrix at a single time point.

    Inputs: 'ntime_samples': number of time samples Nt
            'probs': the probability vector which has dimension NtxN
            'sensitivities': the sensitivity matrix which has dimensions NtxNxN_teta,
                            where N is the number of species and N_teta is the number of parameters
    
    Outputs: The total Fisher Information Matrix for the model.
    '''
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

# # 1904.11583

# stoich_mat = np.array([[2, -2], 
#                         [-2, 2],
#                         [0, 1],
#                         [0, -1],
#                         [1, 0],
#                         [-1, 0]]).T

# def lambda1(params, x):
#     return params[0]*x[1]*(x[1]-1)

# def lambda2(params, x):
#     return params[1]*x[0]*(x[0]-1)

# def lambda3(params, x):
#     return params[2]

# def lambda4(params, x):
#     return params[3]*x[1]

# def lambda5(params, x):
#     return params[4]

# def lambda6(params, x):
#     return params[5]*x[0]

# crn = simulation.CRN(stoich_mat, np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]), 6)
# sensitivities = SensitivitiesDerivation(crn, cr=50)

# # initial states
# init_state_p = np.zeros(1326)
# init_state_p[0] = 1
# init_state_s = np.zeros(1326)
# init_state = np.stack([np.concatenate((init_state_p, init_state_s))]*crn.n_reactions)


# init_state = np.zeros(132)
# init_state[0]=1

# print(sensitivities.solve_ode(init_state, 0., 5., np.arange(6), 4, t_eval=[0,1,2])['y'][66:,1])
# print(sensitivities.solve_ode(init_state, 0., 5., np.arange(6), t_eval=[0,1,2])['y'][6:,1])
# probs, sens = sensitivities.get_sensitivities(init_state, 0, 1, np.arange(6), t_eval = [0, 1])
# print('probs', probs.shape, probs[0,:10], probs[1, :10])
# print(sens, sens.shape)
# # print(sens[1,:,2])
# print('\nsensitivities', sens[1,:10, 2], sens.shape)
# print('\nsensitivities', sens[1,:10,3], sens.shape)
# print('\nsensitivities', sens[1,:10,4], sens.shape)
# print('\nsensitivities', sens[1,:10,5], sens.shape)

# print('shapes', probs.shape, sens.shape)
# inversed_p = np.divide(np.ones_like(probs), probs, out=np.zeros_like(probs), where=probs!=0)
# print('0,0')
# res = 0
# for l in range(10):
#     res += inversed_p[1,l]*sens[1,l,0]*sens[1,l,0]
# print(res)
# print('1,1')
# res = 0
# for l in range(10):
#     res += inversed_p[1,l]*sens[1,l,1]*sens[1,l,1]
# print(res)
# print('3,3')
# res = 0
# for l in range(10):
#     res += inversed_p[1,l]*sens[1,l,3]*sens[1,l,3]
# print(res)
# print(fisher_information_t(probs[1,:], sens[1,:,:]))






