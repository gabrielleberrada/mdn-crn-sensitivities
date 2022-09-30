import numpy as np
import random


class CRN:
    def __init__(self, stoichiometric_mat, propensities, n_params):
        self.stoichiometric_mat = stoichiometric_mat
        self.n_species, self.n_reactions = np.shape(stoichiometric_mat)
        self.propensities = propensities
        self.n_params = n_params

    def parameters_propensities(self, params):
        # associates to each function its parameter
        lambdas = self.propensities
        set_parameters = np.vectorize(lambda f, params: (lambda x: f(params, x)), excluded=[1])
        return set_parameters(lambdas, params)

    def simulation(self, init_state, params, sampling_times, tf):
        lambdas = self.parameters_propensities(params)
        ssa = SSA(init_state, tf, sampling_times, lambdas, self.n_species, self.n_reactions, self.stoichiometric_mat)
        return ssa.step()



class SSA:
    def __init__(self, x0, tf, sampling_times, propensities, n_species, n_reactions, stoich_mat):
        self.initial_state = x0 # necessary?
        self.final_time = tf
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.time = 0
        self.samples = []
        self.sampling_times = sampling_times # to check
        self.current_state = x0
        self.propensities = propensities
        self.stoich_mat = stoich_mat


    def step(self):
        while True:
            eval_propensities = np.vectorize(lambda f, x: f(x), excluded=[1], otypes=[np.ndarray])
            lambdas = eval_propensities(self.propensities, self.current_state)
            if np.size(lambdas) > 1:
                lambdas = np.concatenate(lambdas)
                lambdas.reshape(self.n_species, self.n_reactions)
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
            self.current_state += self.stoich_mat[:, ind_reaction]
            # sampling if needed
            last_index = int(np.searchsorted(self.sampling_times, self.time - delta, side='right'))
            current_index = int(np.searchsorted(self.sampling_times, self.time, side='right'))
            for _ in range(current_index - last_index):
                self.samples.append(list(self.current_state))
        return self.sampling_times, self.samples

