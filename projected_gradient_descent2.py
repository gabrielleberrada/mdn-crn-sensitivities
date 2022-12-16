import numpy as np
import casadi
import matplotlib.pyplot as plt
import torch
import get_sensitivities
import neuralnetwork
import simulation
import generate_data
import fsp
from typing import Callable
import time
from tqdm import tqdm

class ProjectedGradientDescent():
    """Class to compute the Projected Gradient Descent

    Args:
        - *grad_f** (Callable): Gradient of the function of interest.
        - *domain** (np.ndarray): Boundaries of the domain to project in. Shape :math:`(dim, 2)`.
            `domain[:,0]` defines the lower boundaries for each dimension, `domain[:,1]` defines the 
            upper boundaries for each dimension.
        - *dim** (int): Number of dimensions of the considered space.
    """
    def __init__(self, grad_loss: Callable, domain: np.ndarray, dim: int):
        self.grad_loss = grad_loss
        self.domain = domain
        self.dim = dim


    def projected_gradient_descent(self,
                                init: np.ndarray,
                                gamma: float,
                                n_iter: int =1_000,
                                tolerance: float =1e-6,
                                tolerance_rounds: int =1_000,
                                norm: Callable =casadi.norm_2,
                                progress_bar: bool =True # pour afficher la progress bar
                                ):
        """_summary_

        Args:
            - **init** (np.ndarray): Initial state for the controlled parameters. Has shape N_t*n_control_params.
            - **gamma** (float): Step size.
            - **n_iter** (int, optional): Number of iterations for the gradient descent. 
              Defaults to :math:`1000`.
            - **tolerance** (float, optional): Tolerance rate. Defaults to :math:`10^{-20}`.
            - **tolerance_rounds** (int, optional): Number of rounds allowed without improvement before stopping
              the gradient descent. Defaults to :math:`20`.
            - **norm** (Callable, optional): Norm to use for the optimization. Defaults to `casadi.norm_2`.

        Returns:
            _type_: _description_
        """   
        xt = [init]
        losses = []
        if progress_bar:
            pbar = tqdm(total=n_iter, desc = 'Optimizing ...', position=0)
        for i in range(n_iter):
            if progress_bar:
                pbar.update(1)
            x = xt[-1] - gamma*self.grad_loss(xt[-1])
            for n in range(self.dim):
                x[n] = min(x[n], self.domain[n, 1])
                x[n] = max(x[n], self.domain[n, 0])
            xt.append(x)
            losses.append(self.loss(x))
            # to check
            # method 1
            # if i >= tolerance_rounds:
            #     last_elts = np.array(xt[-tolerance_rounds:])
            #     last_elts[0] -= tolerance
            #     if np.argmin(last_elts) == 0:
            #         break
            # method 2
            if i >= tolerance_rounds and np.all(np.abs(losses) <= tolerance):
            # if np.linalg.norm(self.grad_loss(xt[-1]))**2 <= tolerance:
                break
        if progress_bar:
            pbar.close()
        return np.array(xt), np.array(losses) # xt has shape (n_iter, dim)

class ProjectedGradientDescent_CRN(ProjectedGradientDescent):
    def __init__(self,
                domain: np.ndarray,
                fixed_params: np.ndarray,
                time_windows: np.ndarray,
                init_control_params: np.ndarray):
        self.domain = domain
        self.fixed_parameters = fixed_params
        self.n_fixed_params = len(fixed_params)
        self.init_control_params = init_control_params
        self.dim = len(init_control_params)
        self.time_windows = time_windows
        self.n_time_windows = len(time_windows)
        self.n_control_params = self.dim // self.n_time_windows

    def optimisation(self, gamma: float, n_iter: int =1_000, tolerance: float =1e-20, tolerance_rounds: int =20, norm: Callable =casadi.norm_2):
        """_summary_

        Args:
            - **gamma** (float): Step size.
            - **n_iter** (int): Number of iterations for the gradient descent. Defaults to :math:`1000`.
            - **tolerance** (int): Tolerance rate. Defaults to :math:`10^{-20}`.
            - **tolerance_rounds** (float): Number of rounds allowed without improvements before stopping the gradient descent.
              Defaults to :math:`20`.
            - **norm** (Callable): Norm to use for the optimisation. Defaults to `casadi.norm_2`.
        """        
        self.buffer_params, self.buffer_losses = self.projected_gradient_descent(self.init_control_params, gamma, n_iter, tolerance, tolerance_rounds, norm)
        return self.buffer_params[-1], self.buffer_losses[-1]
    
    def plot_control_values(self):
        edges = np.concatenate(([0], self.time_windows))
        plt.stairs(self.buffer_params[-1,:], edges, baseline=None)
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1])
        plt.ylabel('Parameter value')
        plt.xlabel('Time')
        plt.title('Control parameters')
        plt.show()
    
    def plot_losses_trajectory(self):
        plt.plot(self.buffer_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.title('Losses')
        plt.show()

    def plot_control_params_trajectory(self):
        for i in range(self.n_control_params):
            for j in range(self.n_time_windows):
                plt.plot(self.buffer_params[:,i+j*self.n_control_params], label=fr'$\xi_{i+1}^{j+1}$')
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1]+0.1)
        plt.legend()
        plt.show()




class ProjectedGradientDescent_MDN(ProjectedGradientDescent_CRN):
    """_summary_

    Args:
        - **model** (neuralnetwork.NeuralNetwork): Model used for the gradient descent.
        - **domain** (np.ndarray): Allowed interval for each controlled parameter.
        - **fixed_params** (np.ndarray): Values of the fixed parameters.
        - **time_windows** (np.ndarray): Time windows.
        - **cost** (Callable): Cost function.
        - **init_control_params** (np.ndarray): Initial values of the controlled parameters. Must all be non zeros.
        - **length_output** (int): 
        - **with_correction** (bool, optional): If True, sensitivities are set to zero when controlled parameters
            have no influence on that time window. If False, lets the computed sensitivities. Defaults to True.
    """ 
    def __init__(self, 
                model:neuralnetwork.NeuralNetwork, 
                domain: np.ndarray, 
                fixed_params: np.ndarray,
                time_windows: np.ndarray, 
                cost: Callable, 
                init_control_params: np.ndarray,
                length_output: int =200, 
                with_correction: bool =True):  
        super().__init__(domain, fixed_params, time_windows, init_control_params)     
        self.model = model
        self.create_gradient(cost, length_output, with_correction)
        self.create_loss(cost, length_output)

    def create_gradient(self, cost: Callable, length_output: int, with_correction: bool):
        """_summary_

        Args:
            - **time_windows** (np.ndarray): Time windows.
            - **cost** (Callable): Cost function.
            - **length_output** (int):
            - **with_correction** (bool): If True, sensitivities are set to zero when controlled parameters
              have no influence on that time window. If False, lets the computed sensitivities. Defaults to True.
        """        
        if with_correction:
            def grad_loss(control_params):
                res = np.zeros(self.dim)
                params = np.concatenate((self.fixed_parameters, control_params))
                for i, t in enumerate(self.time_windows):
                    inputs = np.concatenate(([t], params))
                    grad_exp = get_sensitivities.gradient_expected_val(torch.tensor(inputs, dtype=torch.float32), self.model, cost, length_output)[1+self.n_fixed_params:]
                    grad_exp[(i+1)*self.n_control_params:] = 0 # these parameters have no influence on this time window
                    res += grad_exp
                return res
        else:
            def grad_loss(control_params):
                res = 0
                params = np.concatenate((self.fixed_parameters, control_params))
                for t in self.time_windows:
                    inputs = np.concatenate(([t], params))
                    res += get_sensitivities.gradient_expected_val(torch.tensor(inputs, dtype=torch.float32), self.model, cost, length_output)[1+self.n_fixed_params:]
                return res
        self.grad_loss = grad_loss

    def create_loss(self, cost: Callable, length_output: int):
        def loss(control_params):
            params = np.concatenate((self.fixed_parameters, control_params))
            loss_value = 0
            for t in self.time_windows:
                inputs = np.concatenate(([t], params))
                loss_value += get_sensitivities.expected_val(torch.tensor(inputs, dtype=torch.float32), self.model, cost, length_output)
            return loss_value
        self.loss = loss


class ProjectedGradientDescent_FSP(ProjectedGradientDescent_CRN):
    def __init__(self,
                crn: simulation.CRN,
                ind_species: int,
                domain: np.ndarray,
                fixed_params: np.ndarray,
                time_windows: np.ndarray,
                cost: Callable,
                init_control_params: np.ndarray,
                cr: int):
        super().__init__(domain, fixed_params, time_windows, init_control_params)
        self.crn = crn
        self.ind_species = ind_species
        self.stv_calculator = fsp.SensitivitiesDerivation(self.crn, self.n_time_windows, index=None, cr=cr)
        self.create_gradient(cost)
        self.create_loss(cost)

    def create_loss(self, cost: Callable):
        def loss(control_params):
            fixed_parameters = np.stack([self.fixed_parameters]*self.n_time_windows)
            control_parameters = control_params.reshape(self.n_time_windows, self.n_control_params)
            params = np.concatenate((fixed_parameters, control_parameters), axis=1)
            res= self.stv_calculator.expected_val(sampling_times=self.time_windows, 
                                                    time_windows=self.time_windows, 
                                                    parameters=params, 
                                                    ind_species=self.ind_species, 
                                                    loss=cost)
            return res.sum()
        self.loss = loss

    def create_gradient(self, cost: Callable):
        def grad_loss(control_params):
            fixed_parameters = np.stack([self.fixed_parameters]*self.n_time_windows)
            control_parameters = control_params.reshape(self.n_time_windows, self.n_control_params)
            params = np.concatenate((fixed_parameters, control_parameters), axis=1)
            res= self.stv_calculator.gradient_expected_val(sampling_times=self.time_windows,
                                                            time_windows=self.time_windows,
                                                            parameters=params,
                                                            ind_species=self.ind_species,
                                                            loss=cost)[:,self.n_fixed_params:]
            return res.sum(axis=0)
        self.grad_loss = grad_loss

    
# testing for MDNs
if __name__ == '__main__':

    def identity(x):
        return x

    def cost(x):
        return (x-2)**2

    from CRN2_control import propensities_production_degradation as propensities
    crn = simulation.CRN(propensities.stoich_mat, propensities.propensities, propensities.init_state, 1, 1)
    domain = np.stack([np.array([1e-10, 3.])]*4)
    fixed_params = np.array([2.])
    init_control_params=np.array([0.9, 0.9, 0.9, 0.9])
    time_windows=np.array([5, 10, 15, 20])

    # MDN
    import save_load_MDN
    model = save_load_MDN.load_MDN_model('CRN2_control/saved_models/CRN2_model1.pt')

    optimizer = ProjectedGradientDescent_MDN(model=model, 
                                            domain=domain, 
                                            fixed_params=fixed_params, 
                                            init_control_params=init_control_params, 
                                            time_windows=time_windows,
                                            cost=cost)
    print('MDN')
    print(optimizer.loss(optimizer.init_control_params))
    print(optimizer.grad_loss(optimizer.init_control_params))

    # FSP
    # optimizer = ProjectedGradientDescent_FSP(crn=crn, 
    #                                         ind_species=propensities.ind_species, 
    #                                         domain=domain, 
    #                                         fixed_params=fixed_params, 
    #                                         time_windows=time_windows, 
    #                                         cost=cost, 
    #                                         init_control_params=init_control_params,
    #                                         cr=50)
    # print('FSP')
    # print(optimizer.loss(optimizer.init_control_params))
    # print(optimizer.grad_loss(optimizer.init_control_params))

    # Optimisation
    start = time.time()
    control_params, losses = optimizer.optimisation(gamma=0.01, n_iter=500, tolerance_rounds=20, tolerance=1e-2)
    end=time.time()
    print('time:', end-start)
    print('results', control_params, losses)
    optimizer.plot_control_values()
    optimizer.plot_losses_trajectory()
    optimizer.plot_control_params_trajectory()
    
    sim = generate_data.CRN_Simulations(crn, np.array([5, 10, 15, 20]), 1_000, 0, complete_trajectory=False, sampling_times=np.arange(21))
    sim.plot_simulations(np.concatenate((fixed_params, control_params)), targets=np.array([[5., 2.], [10., 2.], [15., 2.], [20., 2.]]))