import numpy as np
import casadi
import matplotlib.pyplot as plt
import torch
import get_sensitivities
from typing import Callable


class ProjectedGradientDescent():
    """_summary_

    Args:
        - *grad_f** (Callable): Gradient of the function of interest.
        - *domain** (np.ndarray): Boundaries of the domain to project in. Shape :math:`(2, dim)`.
            `domain[0,:]` defines the lower boundaries for each dimension, `domain[1,:]` defines the 
            upper boundaries for each dimension.
        - *dim** (int): Number of dimensions of the considered space.
    """ 
    def __init__(self, grad_f: Callable, domain: np.ndarray, dim: int):
        self.grad_f = grad_f
        self.domain = domain
        self.dim = dim


    def projected_gradient_descent(self,
                                init: np.ndarray,
                                gamma: Callable,
                                ind_opt: dict ={},
                                n_iter: int =1_000,
                                tolerance: float = 1e-20,
                                tolerance_rounds: int =20,
                                norm: Callable =casadi.norm_2): # en argument, indices des paramètres à optimiser ?
        """_summary_

        Args:
            - **init** (np.ndarray): Initial state.
            - **gamma** (Callable): _description_
            - **nb_iter** (int, optional): Number of iterations for the gradient descent. 
              Defaults to :math:`1000`.
            - **tolerance** (float, optional): Tolerance rate. Defaults to :math:`10^{-20}`.
            - **tolerance_rounds** (int, optional): _description_. Defaults to :math:`20`.
            - **norm** (Callable, optional): _description_. Defaults to `casadi.norm_2`.

        Returns:
            _type_: _description_
        """   
        xt = [init]
        costs = []
        for i in range(n_iter):
            opti = casadi.Opti()
            costs.append(gamma(xt[-1]))
            y = xt[-1] - costs[-1]*self.grad_f(xt[-1])
            x = opti.variable(self.dim)
            opti.minimize(norm(x-y))
            for n in range(self.dim):
                opti.subject_to(x[n] > self.domain[0,n])
                opti.subject_to(x[n] < self.domain[1,n])
            # fixed parameters
            for n, val in ind_opt.items():
                opti.subject_to(x[n] == val)
            opti.solver('ipopt');
            sol = opti.solve();
            xt.append(sol.value(x))
            # to check
            if len(xt) >= tolerance_rounds:
                last_elts = np.array(xt[-tolerance_rounds:])
                last_elts[0] -= tolerance
                if np.argmin(last_elts) == 0:
                    break
        print('Number of iterations:', i+1)
        return np.array(xt), costs # xt has shape (n_iter, dim)

class ProjectedGradientDescent_MDN(ProjectedGradientDescent):
    def __init__(self, model, domain, n_fixed_params, n_control_params, n_time_windows, t, ind_time, f, length_output):
        self.model = model
        self.domain = domain
        self.n_fixed_params = n_fixed_params
        self.n_control_params = n_control_params
        self.n_time_windows = n_time_windows
        self.dim = n_fixed_params+n_control_params*n_time_windows
        self.time = t
        self.ind_time = ind_time
        self.create_gradient(self.time, f, length_output)

    def create_gradient(self, t, f, length_output):
        def grad_f(params):
            return get_sensitivities.expectation_gradient(torch.concat((t, params)), self.model, f, length_output)[1+self.ind_time*self.n_control_params:1+(self.ind_time+1)*self.n_control_params]
        self.grad_f = grad_f

    # def gradient(self, params, t, f=identity, length_output=200):
    #     return get_sensitivities.expectation_gradient(torch.concat((t, params)), self.model, f, length_output)[self.ind_param + 1]
    
    def gradient_descent(self, init, gamma, n_iter, tolerance, tolerance_rounds, norm):
        ind_start = self.n_fixed_params+self.n_time_windows*self.n_control_params
        ind_end = self.n_fixed_params+(self.n_time_windows+1)*self.n_control_params
        return self.projected_gradient_descent(init, gamma, np.arange(ind_start, ind_end), n_iter, tolerance, tolerance_rounds, norm)


class Optimisation_MDN():
    # torch ou np?
    def __init__(self, model, domain, fixed_params, init_control_params, time_windows, default_value =1e-20):
        self.model = model
        self.domain = domain
        self.n_fixed_params = len(fixed_params)
        self.n_control_params = len(init_control_params)
        self.n_time_windows = len(time_windows)
        control_params = np.ones((self.n_time_windows, self.n_control_params))*default_value # shape (N_t, M')
        control_params[0,:] = init_control_params
        self.params = np.concatenate((np.stack([fixed_params]*self.n_time_windows), control_params), axis=1) # shape (Nt, M+M')
        self.time_windows = time_windows


    def optimisation(self, gamma, n_iter, tolerance, tolerance_rounds, norm, f, length_output =200):
        self.buffer_control = np.zeros((self.n_time_windows, n_iter, self.n_control_params*self.n_time_windows))
        self.buffer_costs = np.zeros((self.n_time_windows, n_iter))
        for i, t in enumerate(self.time_windows):
            optimizer = ProjectedGradientDescent_MDN(self.model, self.domain, self.n_fixed_params, self.n_control_params, self.n_time_windows, t, i, f, length_output)
            xt, costs = optimizer.gradient_descent(self.params[i-1,:], gamma, n_iter, tolerance, tolerance_rounds, norm)
            self.params[i, self.n_fixed_params:] = xt[-1,self.n_fixed_params:]
            self.buffer_control[i,:,:] = xt[:, self.n_fixed_params:] # ?
            self.buffer_costs[i,:] = costs


    


# testing
if __name__ == '__main__':
    t_1 = 3
    t_2 = 7
    target = 20
    theta = 5

    def cost(params):
        xsi_1, xsi_2 = params[0], params[1]
        return (theta + xsi_1)**2 * t_1**3 / 3 + (1 - 2 * target) * (theta + xsi_1) / 2 * t_1**2 + target**2 * t_1 + ((theta + xsi_1) * t_1 + (theta + xsi_1)**2 * t_1**2) * (t_2 - t_1) + (2 * (xsi_1 + theta) * t_1 + 1) * (theta + xsi_2) * (t_2 - t_1)**2 / 2 + (theta + xsi_2)**2 * (t_2 - t_1)**3 / 3 - 2 * target * ((xsi_1 + theta) * t_1 * (t_2 - t_1) + (theta + xsi_2) * (t_2 - t_1)**2 / 2) + target**2 * (t_2 - t_1)

    def grad_f(params):
        xsi_1, xsi_2 = params[0], params[1]
        grad1 = t_1**3 / 3 * 2 * (theta + xsi_1) + (1 - 2 * target) * t_1**2 / 2 + (t_2 - t_1) * (t_1 + t_1**2 * 2 * (theta + xsi_1)) + (theta + xsi_2) * (t_2 - t_1)**2 * t_1 - 2 * target * t_1 * (t_2 - t_1)
        grad2 = (2 * (xsi_1 + theta) * t_1 + 1) * (t_2 - t_1)**2 / 2 + (t_2 - t_1)**3 / 3 * 2 * (xsi_2 + theta) - 2 * target * (t_2 - t_1)**2 / 2
        return np.array([grad1, grad2])

    domain = np.array([[0, 0], [5, 5]])

    pgd = ProjectedGradientDescent(grad_f, domain, 2)
    xt, yt = pgd.projected_gradient_descent(init=np.array([2, 6]), nb_iter=20, gamma=cost)
    xt_param1 = xt[:, 0]
    xt_param2 = xt[:, 1]


    print(xt_param1[-1], xt_param2[-1], yt[-1])
    plt.plot(xt_param1)
    plt.ylabel('Parameter 1')
    plt.xlabel('Iterations')
    plt.title('Gradient descent')
    plt.show()
    plt.plot(xt_param2, label='Parameter 2')
    plt.xlabel('Iterations')
    plt.ylabel('Parameter 2')
    plt.title('Gradient descent')
    plt.show()
    plt.plot(yt)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient descent')
    plt.show()




