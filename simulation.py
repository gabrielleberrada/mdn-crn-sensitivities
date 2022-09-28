from re import L
import numpy as np
import torch
import random


class CRN:
    def __init__(self, stoichiometric_mat, propensities):
        self.n_species, self.n_reactions = np.shape(stoichiometric_mat)
        self.stoichiometric_mat = stoichiometric_mat
        self.propensities = propensities


class SSA:
    def __init__(self, x0, tf, crn, sampling_times, parameters):
        self.initial_state = x0 # necessary?
        self.final_time = tf
        self.CRN = crn
        self.time = 0
        self.samples = []
        self.sampling_times = sampling_times # to check
        self.current_state = x0
        self.parameters = parameters

    def params_propensities(self):
        # associate to each function its parameter
        lambdas = self.CRN.propensities
        set_parameters = np.vectorize(lambda f, params: (lambda x: f(params, x)), excluded=[1])
        self.param_propensities = set_parameters(lambdas, self.parameters)

    def step(self):
        self.params_propensities()
        sampling_times = self.sampling_times
        while True:
            eval_propensities = np.vectorize(lambda f, x: f(x), excluded=[1], otypes=[np.ndarray])
            lambdas = eval_propensities(self.param_propensities, self.current_state)
            if np.size(lambdas) > 1:
                lambdas = np.concatenate(lambdas)
                lambdas.reshape(self.CRN.n_species, self.CRN.n_reactions)
            lambda0 = lambdas.sum()
            probabilities = np.cumsum(lambdas) / lambda0
            delta = np.random.exponential(1/lambda0)
            self.time += delta
            if (self.time > self.sampling_times[-1]) or (self.time > self.final_time):
                # last samples
                for _ in range(len(self.sampling_times) - len(self.samples)):
                    self.samples.append(list(self.current_state))
                break
            # choosing which reaction occurs
            u = random.random()
            ind_reaction = np.searchsorted(probabilities, u, side='left') # the reaction n°ind_reaction occurs
            # updating state
            self.current_state += self.CRN.stoichiometric_mat[:, ind_reaction]
            # sampling if needed
            last_index = int(np.searchsorted(sampling_times, self.time - delta, side='right'))
            current_index = int(np.searchsorted(sampling_times, self.time, side='right'))
            for _ in range(current_index - last_index):
                self.samples.append(list(self.current_state))
        return self.sampling_times, self.samples
    
    def outputs(self):
        n_samples = len(self.sampling_times)
        n_params = len(self.parameters)
        t = np.expand_dims(self.sampling_times, axis=0).transpose()
        p = self.parameters * n_samples
        p = np.reshape(p, (n_samples, n_params))
        return np.concatenate((t,p), axis=1), self.samples


# testing
     
def lambda1(params, x):
    return params[0]

stoichiometric_mat = np.array([1]).reshape(1,1)

crn = CRN(stoichiometric_mat, np.array([lambda1]))

sampling_times = np.linspace(0, 1, 50)
ssa = SSA(np.array([0.]), 700, crn, sampling_times, [2.])

sampling_times, samples = ssa.step()

print(samples)

X, y = ssa.outputs()

print(y)
