import numpy as np
import matplotlib.pyplot as plt
import torch
import convert_csv
import get_sensitivities
import neuralnetwork
import simulation
import generate_data
import fsp
import collections.abc as abc
from typing import Callable, Union, Tuple
import time
from tqdm import tqdm

class ProjectedGradientDescent():
    """Class to compute the Projected Gradient Descent Algorithm.

    Args:
        - *grad_loss** (Callable): Gradient of the loss.
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
                                tolerance: float =1e-3,
                                tolerance_rounds: int =1_000,
                                clipping_value: float = 50.,
                                progress_bar: bool =True # pour afficher la progress bar
                                ) -> Tuple[np.ndarray]:
        """Computes the Projected Gradient Descent.

        Args:
            - **init** (np.ndarray): Initial state for the controlled parameters. Has shape N_t*n_control_params.
            - **gamma** (float): Step size.
            - **n_iter** (int, optional): Number of iterations for the gradient descent. 
              Defaults to :math:`1000`.
            - **tolerance** (float, optional): Tolerance rate. Defaults to :math:`10^{-3}`.
            - **tolerance_rounds** (int, optional): Number of rounds allowed without improvement before stopping
              the gradient descent. Defaults to :math:`20`.
            - **progress_bar** (bool, optional): If True, plots a progress bar during optimisation. Defaults to True.

        Returns:
            - **x**: Array of the controlled parameters values estimated at each iteration. Has shape (n_iter, dim).
            - **losses**: Array of the loss values estimated at each iteration. Has shape (n_iter).
        """   
        xt = [init]
        losses = []
        grads = []
        if progress_bar:
            pbar = tqdm(total=n_iter, desc = 'Optimizing ...', position=0)
        for i in range(n_iter):
            if progress_bar:
                pbar.update(1)
            # gradient clipping
            grad = self.grad_loss(xt[-1])
            grads.append(grad)
            if np.linalg.norm(grad) > clipping_value:
                # gamma /= 2
                grad = clipping_value / np.linalg.norm(grad) * grad
            x = xt[-1] - gamma*grad
            for n in range(self.dim):
                x[n] = min(x[n], self.domain[n, 1])
                x[n] = max(x[n], self.domain[n, 0])
            if np.linalg.norm(self.grad_loss(xt[-1]))**2 <= tolerance:
                break
            xt.append(x)
            losses.append(self.loss(x))
        if progress_bar:
            pbar.close()
        return np.array(xt), np.array(losses), np.array(grads)

class ProjectedGradientDescent_CRN(ProjectedGradientDescent):
    """Class to compute a Projected Gradient Descent for a CRN.

    Args:
        - *domain** (np.ndarray): Boundaries of the domain to project in. Shape :math:`(dim, 2)`.
            `domain[:,0]` defines the lower boundaries for each dimension, `domain[:,1]` defines the 
            upper boundaries for each dimension.
        - **fixed_params** (np.ndarray): Values of the fixed parameters.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. 
          Its form is :math:`[t_1, ..., t_T]`, such that the considered time windows are 
          :math:`[0, t_1], [t_1, t_2], ..., [t_{T-1}, t_T]`. :math:`t_T` must match with the final time 
          :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        """   
    def __init__(self,
                crn: simulation.CRN,
                domain: np.ndarray,
                fixed_params: np.ndarray,
                time_windows: np.ndarray,
                loss: Union[Callable, np.ndarray],
                weights: np.ndarray =None
                ):     
        self.domain = domain.copy()
        self.fixed_parameters = fixed_params
        self.n_fixed_params = len(fixed_params)
        self.init_control_params = np.random.uniform(self.domain[:, 0], self.domain[:, 1])
        self.dim = len(self.init_control_params)
        self.time_windows = time_windows
        self.n_time_windows = len(time_windows)
        self.n_control_params = self.dim // self.n_time_windows
        if weights is None:
            # All controlled parameters have the same weight.
            self.weights = np.ones(self.n_time_windows)
        else:
            self.weights = weights
        if isinstance(loss, abc.Hashable):
            self.loss_function = [loss]*self.n_time_windows
        else:
            self.loss_function = loss
        self.crn = crn

    def optimisation(self, gamma: float, n_iter: int =1_000, tolerance: float =1e-3, tolerance_rounds: int =20) -> Tuple[np.ndarray, float]:       
        """Computes the Projected Gradient Descent.

        Args:
            - **gamma** (float): Step size.
            - **n_iter** (int): Number of iterations for the gradient descent. Defaults to :math:`1000`.
            - **tolerance** (int): Tolerance rate. Defaults to :math:`10^{-3}`.
            - **tolerance_rounds** (float): Number of rounds allowed without improvements before stopping the gradient descent.
              Defaults to :math:`20`.
        """        
        self.buffer_params, self.buffer_losses, self.buffer_grads = self.projected_gradient_descent(self.init_control_params, gamma, n_iter, tolerance, tolerance_rounds)
        return self.buffer_params[-1], self.buffer_losses[-1]
    
    def plot_control_values(self, save=(False, None)):
        """Plots the optimised controlled parameters over time.
        """      
        edges = np.concatenate(([0], self.time_windows))
        plt.stairs(self.buffer_params[-1,:], edges, baseline=None)
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1])
        plt.ylabel('Parameter value')
        plt.xlabel('Time')
        plt.title('Controlled parameters')
        if save[0]:
            convert_csv.array_to_csv(self.buffer_params[-1, :], save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()
    
    def plot_losses_trajectory(self, save=(False, None)):
        """Plots the loss values over the iterations.
        """        
        plt.plot(self.buffer_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.title('Losses')
        if save[0]:
            convert_csv.array_to_csv(self.buffer_losses, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

    def plot_control_params_trajectory(self, save=(False, None)):
        """Plots the values of the controlled parameters over the iterations.
        """  
        for i in range(self.n_control_params):
            for j in range(self.n_time_windows):
                plt.plot(self.buffer_params[:,i+j*self.n_control_params], label=fr'$\xi_{i+1}^{j+1}$')
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1]+0.1)
        plt.legend()
        if save[0]:
            convert_csv.array_to_csv(self.buffer_params, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

    def plot_gradients_trajectory(self, save=(False, None)):
        for i in range(self.n_control_params):
            for j in range(self.n_time_windows):
                plt.plot(self.buffer_grads[:, i+j*self.n_control_params], label=fr'Gradient wrt $\xi_{i+1}^{j+1}$')
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1]+0.1)
        plt.legend()
        if save[0]:
            convert_csv.array_to_csv(self.buffer_grads, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

    
    def plot_abundances(self, ind_species, targets=None, rate=1_000, save=(False, None)):
        sim = generate_data.CRN_Simulations(self.crn, 
                                            optimizer.time_windows, 
                                            500, 
                                            ind_species=ind_species, 
                                            complete_trajectory=False, 
                                            sampling_times = optimizer.time_windows)
        res = []
        n_iter = self.buffer_params.shape[0]
        for i in range(n_iter//rate):
            parameters = np.concatenate((optimizer.fixed_parameters, self.buffer_params[i*rate,:]))
            samples, _ = sim.run_simulations(parameters)
            res.append(np.mean(samples, axis=0))
        res = np.array(res)
        for i, t in enumerate(optimizer.time_windows):
            plt.scatter(np.linspace(0, n_iter, n_iter//rate), res[:,i], marker = '+', label=f'$t={t}$')
        if targets is not None:
            for target in targets:
                plt.axhline(y = target, linestyle = 'dashed', color='gray')
        plt.legend()
        plt.title('Abundance evolution')
        if save[0]:
            convert_csv.array_to_csv(res, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()


    def plot_performance_index(self, ind_species, rate=200, save=(False, None)):
        sim = generate_data.CRN_Simulations(self.crn, 
                                            self.time_windows, 
                                            n_trajectories=10**4, 
                                            ind_species=ind_species, 
                                            complete_trajectory=False, 
                                            sampling_times = self.time_windows)
        n_iter = self.buffer_params.shape[0]
        performance_index = np.zeros(n_iter//rate)
        for i in range(n_iter//rate):
            parameters = np.concatenate((self.fixed_parameters, self.buffer_params[i*rate,:]))
            samples, _ = sim.run_simulations(parameters)
            expect = np.mean(samples, axis=0)
            res = 0
            for j in range(self.n_time_windows):
                res += self.weights[j] * self.loss_function[j](expect[j])
            performance_index[i] = res
        plt.plot(np.linspace(0, n_iter, n_iter//rate), performance_index)
        plt.title('Performance index')
        if save[0]:
            convert_csv.array_to_csv(performance_index, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()


class ProjectedGradientDescent_MDN(ProjectedGradientDescent_CRN):
    """Class to compute the Projected Gradient Descent based on a Mixture Density Network model.

    Args:
        - **model** (neuralnetwork.NeuralNetwork): Model used for the gradient descent.
        - **domain** (np.ndarray): Allowed interval for each controlled parameter.
        - **fixed_params** (np.ndarray): Values of the fixed parameters.
        - **time_windows** (np.ndarray): Time windows.
        - **loss** (Union[Callable, np.ndarray]): Loss function. When it is an array, each element is the 
          loss function for the corresponding time window.
        - **weights** (np.ndarray, optional): Weights for each target value. Defaults to None.
        - **length_output** (int): Length of the output. Defaults to :math:`200`.
        - **with_correction** (bool, optional): If True, sensitivities are set to zero when controlled parameters
            have no influence on that time window. If False, lets the computed sensitivities. Defaults to True.
    """ 
    def __init__(self, 
                crn: simulation.CRN,
                model:neuralnetwork.NeuralNetwork, 
                domain: np.ndarray, 
                fixed_params: np.ndarray,
                time_windows: np.ndarray, 
                loss: Union[Callable, list],
                weights: np.ndarray =None,
                length_output: int =200, 
                with_correction: bool =True):  
        super().__init__(crn, domain, fixed_params, time_windows, loss, weights)     
        self.model = model
        self.create_gradient(length_output, with_correction)
        self.create_loss(length_output)

    def create_gradient(self, length_output: int, with_correction: bool):
        r"""Computes the gradient function of the loss evaluated at the expected value with respect to all controlled parameters.

        .. math::

            \xi \mapsto \sum_{i=1}^{N_t} w_i \nabla_{\xi} \mathcal{L}_i(E[X_t^{\theta, \xi}])

        Usually, :math:`\mathcal{L}_i` is of the form :math:`\mathcal{L}_i(x) = (x-\text{target}_i)`. 

        Args:
            - **loss** (Union[Callable, np.ndarray): Loss function. When it is an array, each element is the loss function for the
              corresponding time window.
            - **length_output** (int): Length of the output.
            - **weights** (np.ndarray): Weights for each target value.
            - **with_correction** (bool): If True, sensitivities are set to zero when controlled parameters
              have no influence on that time window. If False, lets the computed sensitivities. Defaults to True.
        """
        if with_correction:
            def grad_loss(control_params):
                res = np.zeros(self.dim)
                params = np.concatenate((self.fixed_parameters, control_params))
                for i, t in enumerate(self.time_windows):
                    inputs = np.concatenate(([t], params))
                    grad_exp = get_sensitivities.gradient_expected_val(inputs=torch.tensor(inputs, dtype=torch.float32), 
                                                                        model=self.model, 
                                                                        loss=self.loss_function[i], 
                                                                        length_output=length_output)[1+self.n_fixed_params:]
                    grad_exp[(i+1)*self.n_control_params:] = 0 # these parameters have no influence on this time window
                    res += self.weights[i]*grad_exp
                return res
        else:
            def grad_loss(control_params):
                res = 0
                params = np.concatenate((self.fixed_parameters, control_params))
                for i, t in enumerate(self.time_windows):
                    inputs = np.concatenate(([t], params))
                    res += self.weights[i]*get_sensitivities.gradient_expected_val(inputs=torch.tensor(inputs, dtype=torch.float32), 
                                                                                    model=self.model, 
                                                                                    loss=self.loss_function[i], 
                                                                                    length_output=length_output)[1+self.n_fixed_params:]
                return res
        self.grad_loss = grad_loss

    def create_loss(self, length_output: int):
        """Computes the loss evaluated at the expected value.

        Args:
            - **loss** (Union[Callable, np.ndarray): Loss function. When it is an array, each element is the loss function for the
              corresponding time window.
            - **length_output** (int): Length of the output.
            - **weights** (np.ndarray): Weights for each target value.
        """ 
        def loss(control_params):
            params = np.concatenate((self.fixed_parameters, control_params))
            loss_value = 0
            for i, t in enumerate(self.time_windows):
                inputs = np.concatenate(([t], params))
                loss_value += self.weights[i]*get_sensitivities.expected_val(inputs=torch.tensor(inputs, dtype=torch.float32), 
                                                                            model=self.model, 
                                                                            loss=self.loss_function[i], 
                                                                            length_output=length_output)
            return loss_value
        self.loss = loss


class ProjectedGradientDescent_FSP(ProjectedGradientDescent_CRN):
    """Class to compute the Projected Gradient Descent based on the Finite State Projection method.

    Args:
        - **crn** (simulation.CRN): CRN to work on.
        - **ind_species** (int): Index of the species to study.
        - **domain** (np.ndarray): Allowed interval for each controlled parameter.
        - **fixed_params** (np.ndarray): Values of the fixed parameters.
        - **time_windows** (np.ndarray): Time windows.
        - **loss** (Union[Callable, np.ndarray]): Loss function. When it is an array, each element is the loss function for the
          corresponding time window.
        - **grad_loss** (Union[Callable, np.ndarray]): Gradient of the loss function. When it is an array, each element is the loss
          function for the corresponding time window.
        - **weights** (np.ndarray): Weights for each target value.
        - :math:`C_r` (int): value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. Defaults to :math:`50`.
    """       
    def __init__(self,
                crn: simulation.CRN, # make sure to specify propensities_drv
                ind_species: int,
                domain: np.ndarray,
                fixed_params: np.ndarray,
                time_windows: np.ndarray,
                loss: Union[Callable, np.ndarray],
                grad_loss: Union[Callable, np.ndarray],
                weights: np.ndarray =None,
                cr: int =50): 
        super().__init__(crn=crn, domain=domain, fixed_params=fixed_params, time_windows=time_windows, loss=loss, weights=weights)
        self.ind_species = ind_species
        self.stv_calculator = fsp.SensitivitiesDerivation(self.crn, self.n_time_windows, index=None, cr=cr)
        if isinstance(grad_loss, abc.Hashable):
            self.grad_loss_function = [grad_loss]*self.n_time_windows
        else:
            self.grad_loss_function = grad_loss
        self.create_gradient()
        self.create_loss()


    def create_loss(self):
        """Computes the loss function.

        Args:
            - **loss** (Union[Callable, np.ndarray]): Loss function. When it is an array, each element is the 
              loss function for the corresponding time window.
            - **weights** (np.ndarray): Weights for each target value. 
        """  
        def loss_function(control_params):
            fixed_parameters = np.stack([self.fixed_parameters]*self.n_time_windows)
            control_parameters = control_params.reshape(self.n_time_windows, self.n_control_params)
            params = np.concatenate((fixed_parameters, control_parameters), axis=1)
            res= self.stv_calculator.expected_val(sampling_times=self.time_windows, 
                                                    time_windows=self.time_windows, 
                                                    parameters=params, 
                                                    ind_species=self.ind_species, 
                                                    loss=self.loss_function) # shape (n_controlled_parameters)
            return np.dot(self.weights, res) # res.sum()
        self.loss = loss_function

    def create_gradient(self):
        """Computes the gradient of the loss function.

        Args:
            - **grad_loss** (Union[Callable, np.ndarray]): Gradient of the loss function. When it is an array, 
              each element is the loss function for the corresponding time window.
        """
        def gradient_loss(control_params):
            fixed_parameters = np.stack([self.fixed_parameters]*self.n_time_windows)
            control_parameters = control_params.reshape(self.n_time_windows, self.n_control_params)
            params = np.concatenate((fixed_parameters, control_parameters), axis=1)
            gradient = self.stv_calculator.gradient_expected_val(sampling_times=self.time_windows,
                                                            time_windows=self.time_windows,
                                                            parameters=params,
                                                            ind_species=self.ind_species)[:,self.n_fixed_params:]
            expec = self.stv_calculator.expected_val(sampling_times=self.time_windows,
                                                    time_windows=self.time_windows,
                                                    parameters=params,
                                                    ind_species=self.ind_species)
            res = np.zeros((self.n_time_windows, self.n_control_params*self.n_time_windows))
            for i, f in enumerate(self.grad_loss_function):
                res[i, :] = f(expec[i], gradient[i,:])
            return np.dot(self.weights, res)
        self.grad_loss = gradient_loss

def control_method(optimizer, 
                gamma, 
                n_iter, 
                tolerance_rounds, 
                tolerance, 
                ind_species, 
                targets,
                rate_performance=200,
                save=(True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results'])):
    start = time.time()
    control_params, loss_value = optimizer.optimisation(gamma=gamma, 
                                                        n_iter=n_iter,
                                                        tolerance_rounds=tolerance_rounds, 
                                                        tolerance=tolerance)
    end = time.time()
    print('Time: ', end-start)
    print('Control parameters: ', control_params)
    print('Final loss: ', loss_value)
    optimizer.plot_control_values(save=(save[0], save[1][0]))
    optimizer.plot_losses_trajectory(save=(save[0], save[1][1]))
    optimizer.plot_control_params_trajectory(save=(save[0], save[1][2]))
    optimizer.plot_gradients_trajectory(save=(save[0], save[1][3]))
    optimizer.plot_performance_index(ind_species=ind_species, rate=rate_performance, save=(save[0], save[1][4]))
    sim = generate_data.CRN_Simulations(optimizer.crn, 
                                        optimizer.time_windows, 
                                        10_000,
                                        ind_species=ind_species, 
                                        complete_trajectory=False, 
                                        sampling_times = np.arange(optimizer.time_windows[-1]+1))
    sim.plot_simulations(np.concatenate((optimizer.fixed_parameters, control_params)), targets=targets, save=(save[0], save[1][5]))
    return end-start, control_params, loss_value


# testing for MDNs
if __name__ == '__main__':

    # def identity(x):
    #     return x

    def loss(x):
        return (x-3)**2

    def loss1(x):
        return (x-3)**2

    def loss2(x):
        return (x-2)**2

    def loss3(x):
        return (x-1)**2

    def loss4(x):
        return x**2

    def grad_loss1(probs, gradient):
        return 2*gradient*(probs-3)

    def grad_loss2(probs, gradient):
        return 2*gradient*(probs-2)

    def grad_loss3(probs, gradient):
        return 2*gradient*(probs-1)

    def grad_loss4(probs, gradient):
        return 2*gradient*probs

    from CRN4_control import propensities_bursting_gene as propensities
    crn = simulation.CRN(propensities.stoich_mat, propensities.propensities, propensities.init_state, 3, 1)
    domain = np.stack([np.array([1e-2, 5.])]*4)
    fixed_params = np.array([1., 2., 1.])
    time_windows = np.array([5, 10, 15, 20])
    # from CRN2_control import propensities_production_degradation as propensities
    # crn = simulation.CRN(propensities.stoich_mat, propensities.propensities, propensities.init_state, 1, 1)
    # domain = np.stack([np.array([1e-10, 7.])]*4)
    # fixed_params = np.array([2.])
    # time_windows = np.array([5, 10, 15, 20])

    # MDN
    import save_load_MDN
    model = save_load_MDN.load_MDN_model('CRN4_control/saved_models/CRN4_model1.pt')

    optimizer = ProjectedGradientDescent_MDN(crn=crn,
                                            model=model, 
                                            domain=domain, 
                                            fixed_params=fixed_params,
                                            time_windows=time_windows,
                                            loss=loss2,
                                            weights=None)
    # print('MDN')
    # print(optimizer.loss(optimizer.init_control_params))
    # print(optimizer.grad_loss(optimizer.init_control_params))
    # plot.multiple_plots(to_pred=[torch.tensor([k, 2., 1.34, 1.34, 1.34, 1.34]) for k in [5., 10., 15., 20.]], 
    #                     models=[model], 
    #                     up_bound=4*[30],
    #                     time_windows=np.array([5, 10, 15, 20]),
    #                     n_comps=4,
    #                     index_names = ('Sensitivities', r'Abundance of species $S$'),
    #                     plot_exact_result=(False, None),
    #                     plot_fsp_result=(True, propensities.stoich_mat, propensities.propensities, 30, propensities.init_state, 0, 1, 1),
    #                     )
    # plt.show()

    # FSP
    # optimizer = ProjectedGradientDescent_FSP(crn=crn, 
    #                                         ind_species=propensities.ind_species, 
    #                                         domain=domain, 
    #                                         fixed_params=fixed_params, 
    #                                         time_windows=time_windows, 
    #                                         loss=loss3,
    #                                         grad_loss= grad_loss3,
    #                                         cr=50)
    # print('FSP')
    # print(optimizer.init_control_params)
    # print(optimizer.loss(optimizer.init_control_params))
    # print(optimizer.grad_loss(optimizer.init_control_params))

    # # Exact loss
    # def exact_gradient_loss(t, theta1, theta2):
    #     lambd = theta1/theta2*(1-np.exp(-theta2*t))
    #     # return lambd*(t*np.exp(-theta2*t) - 1/theta2)*(2*lambd - 2*2+1)
    #     return 2*(lambd-2)*theta1/theta2*(-(t-np.exp(-theta2*t))/theta2 + t*theta1/theta2*np.exp(-theta2*t))

    # def exact_loss(t, theta1, theta2):
    #     lambd = theta1/theta2*(1-np.exp(-theta2*t))
    #     # return lambd*(lambd+1-2*2) + 2**2
    #     return (lambd -2)**2

    # def f(theta1, theta2):
    #     res = 0
    #     for t in time_windows:
    #         res += exact_gradient_loss(t, theta1, theta2)
    #     return res

    # # stv = fsp.SensitivitiesDerivation(crn, 1, 1, cr=50)
    # # x1 = []
    # x2 = []
    # x3 = []
    # for elt in np.linspace(0.2, 2, 50):
    #     fixed_parameters = np.stack([np.array([2.])]*4)
    #     control_parameters = np.array([elt]*4).reshape(4,1)
    #     params = np.concatenate((fixed_parameters, control_parameters), axis=1)
    #     # x1.append(stv.expected_val(np.array([5]), np.array([20]), params, 0, loss=loss))
    #     x3.append(optimizer.grad_loss(np.array([elt]*4))[2])
    #     # res = 0
    #     # for t in time_windows:
    #     #     res += get_sensitivities.gradient_expected_val(torch.tensor([t, 2., elt, elt, elt, elt]).float(), model, loss=loss)[2]
    #     # x3.append(res)
    #     # x3.append(get_sensitivities.gradient_expected_val(torch.tensor([5, 2., elt, elt, elt, elt]).float(), model, loss=loss)[2])
    #     # x2.append(f(2., elt))
    #     x2.append(f(2., elt))
    # print(np.argmin(np.abs(x2)), np.argmin(np.abs(x3)))#, np.argmin(np.abs(x1)))

    # plt.scatter(np.linspace(0.1, 2, 50), x2, marker='x', label='exact')
    # # plt.scatter(np.linspace(0.1, 2, 50), x1, marker='+', label='FSP')
    # plt.scatter(np.linspace(0.1, 2., 50), x3, marker='+', label='MDN')
    # plt.legend()
    # plt.show()

    # Optimisation

    # loss 1
    # control_method(optimizer, gamma=0.0001, n_iter=30_000, tolerance_rounds=15, tolerance=1e-5, crn=crn, ind_species=propensities.ind_species, targets=np.array([[5., 3.], [10., 3.], [15., 3.], [20., 3.]]))

    # loss 2
    control_method(optimizer, 
                gamma=0.0001, 
                n_iter=5_000,
                tolerance_rounds=15,
                tolerance=1e-5,
                ind_species=propensities.ind_species,
                targets=np.array([[5., 2.], [10., 2.], [15., 2.], [20., 2.]]), 
                save=(False, [None]*5))

    # loss 3
    # fixed_params = np.array([1., 1., 1.])
    # control_method(optimizer, gamma=0.005, n_iter=20_000, tolerance_rounds=20, tolerance=1e-5, crn=crn, ind_species=propensities.ind_species, targets=np.array([[5., 1.], [10., 1.], [15., 1.], [20., 1.]]))

    
    