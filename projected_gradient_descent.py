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
        - **grad_loss** (Callable): Gradient function of the loss.
        - **domain** (np.ndarray): Boundaries of the domain in which to project. Has hape :math:`(dim, 2)`.
            **domain[:,0]** defines the lower boundaries for each dimension, **domain[:,1]** defines the 
            upper boundaries for each dimension.
        - **dim** (int): Dimension of the projection space.
    """
    def __init__(self, grad_loss: Callable, domain: np.ndarray, dim: int):
        self.grad_loss = grad_loss
        self.domain = domain
        self.dim = dim


    def projected_gradient_descent(self,
                                init: np.ndarray,
                                gamma: float,
                                n_iter: int =20_000,
                                eps: float =1e-5,
                                clipping_value: float = 50.,
                                progress_bar: bool =False
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Computes the Projected Gradient Descent.

        Args:
            - **init** (np.ndarray): Initial state for the controlled parameters. 
              Has shape :math:`L \times n_control_params`.
            - **gamma** (float): Step size.
            - :math:`n_{\text{iter}}` (int, optional): Maximal number of iterations allowed for the gradient descent. 
              Defaults to :math:`20000`.
            - **eps** (float, optional): Tolerance rate. The algorithm halts when the squared norm of the gradient of 
              the loss value is smaller than **eps**. Defaults to :math:`10^{-5}`.
            - **clipping_value** (float, optional): Maximal gradient norm value. Used to avoid explosing gradients.
              Defaults to :math:`50`.
            - **progress_bar** (bool, optional): If True, plots a progress bar during optimisation. Defaults to False.

        Returns:
            - Array of the controlled parameters values estimated at each iteration. Has shape :math:`(n, dim)`.
            - Array of the loss values estimated at each iteration. Has shape :math:`(n)`.
            - Array of the gradient values estimated at each iteration. Has shape :math:`(n, dim)`.
            - :math:`n`: number of iterations performed by the algorithm.
        """   
        xt = [init]
        losses = []
        grads = []
        print('Optimizing...')
        if progress_bar:
            pbar = tqdm(total=n_iter, desc = 'Optimizing ...', position=0)
        i = 0
        while i < n_iter and np.linalg.norm(self.grad_loss(xt[-1]))**2 > eps:
            if progress_bar:
                pbar.update(1)
            # gradient clipping
            grad = self.grad_loss(xt[-1])
            grads.append(grad)
            if np.linalg.norm(grad) > clipping_value:
                grad = clipping_value / np.linalg.norm(grad) * grad
            x = xt[-1] - gamma*grad
            # projection
            for n in range(self.dim):
                x[n] = min(x[n], self.domain[n, 1])
                x[n] = max(x[n], self.domain[n, 0])
            xt.append(x)
            losses.append(self.loss(x))
            i += 1
        if progress_bar:
            pbar.close()
        return np.array(xt), np.array(losses), np.array(grads), i

class ProjectedGradientDescent_CRN(ProjectedGradientDescent):
    """Class to compute a Projected Gradient Descent for a CRN.

    Args:
        - **crn** (simulation.CRN): CRN to compute the PGD on.
        - *domain** (np.ndarray): Boundaries of the domain in which to project. Has shape :math:`(dim, 2)`.
          **domain[:,0]** defines the lower boundaries for each dimension, **domain[:,1]** defines the 
          upper boundaries for each dimension.
        - **fixed_params** (np.ndarray): Selected values for the fixed parameters.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. 
          Its form is :math:`[t_1, ..., t_L]`, such that the considered time windows are 
          :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match with the final time 
          :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - **loss** (Union[Callable, np.ndarray]): Loss function used for the gradient descent.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L)`. If None, all targets
          have the same weight. Defaults to None.
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

    def optimisation(self, 
                    gamma: float, 
                    n_iter: int =1_000, 
                    eps: float =1e-3) -> Tuple[np.ndarray, float]:       
        """Computes the Projected Gradient Descent.

        Args:
            - **gamma** (float): Step size.
            - **n_iter** (int): Number of iterations for the gradient descent. Defaults to :math:`1000`.
            - **eps** (int): Tolerance rate. Defaults to :math:`10^{-3}`.
        """        
        self.buffer_params, self.buffer_losses, self.buffer_grads, i = self.projected_gradient_descent(self.init_control_params, 
                                                                                                        gamma, 
                                                                                                        n_iter, 
                                                                                                        eps)
        return self.buffer_params[-1], self.buffer_losses[-1], i
    
    def plot_control_values(self, 
                            save: Tuple[bool, str] =(False, None)):
        """Plots the parameters :math:`\xi_i` values over the iterations.

        Args:
            - **save** (Tuple[bool, str]): If the first argument is True, saves the file. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
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
    
    def plot_losses_trajectory(self, 
                            save: Tuple[bool, str] =(False, None)):
        """Plots the loss values over the iterations.

        Args:
            - **save** (Tuple[bool, str]):
        """        
        plt.plot(self.buffer_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss value')
        plt.title('Losses')
        if save[0]:
            convert_csv.array_to_csv(self.buffer_losses, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

    def plot_control_params_trajectory(self, 
                                    save: Tuple[bool, str] =(False, None)):
        """Plots the values of the controlled parameters at each iteration.

        Args:
            - **save** (Tuple[bool, str]): If the first argument is True, saves the file. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
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

    def plot_gradients_trajectory(self, 
                                save: Tuple[bool, str] =(False, None)):
        """Plots the values of the gradients at each iteration.

        Args:
            - **save** (Tuple[bool, str]): If the first argument is True, saves the file. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
        """        
        for i in range(self.n_control_params):
            for j in range(self.n_time_windows):
                plt.plot(self.buffer_grads[:, i+j*self.n_control_params], label=fr'Gradient wrt $\xi_{i+1}^{j+1}$')
        plt.ylim(plt.ylim()[0]-0.1, plt.ylim()[1]+0.1)
        plt.legend()
        if save[0]:
            convert_csv.array_to_csv(self.buffer_grads, save[1])
            plt.savefig(f'{save[1]}.pdf')
        plt.show()

    
    def plot_abundances(self, 
                        ind_species: int, 
                        targets: np.ndarray =None, 
                        rate: int =1_000, 
                        save: Tuple[bool, str] =(False, None)):
        """Plots the mean evolution of the abundance over time based on the SSA method.

        Args:
            - **ind_species** (int): Index of the species of interest.
            - **targets** (np.ndarray, optional): Target values at each time point.
              If None, no target is plotted. Defaults to None.
            - **rate** (int, optional): Plotting rate. Defaults to :math:`1000`.
            - **save** (Tuple[bool, str]): If the first argument is True, saves the file. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
        """        
        sim = generate_data.CRN_Simulations(self.crn, 
                                            self.time_windows, 
                                            n_trajectories=10**4, 
                                            ind_species=ind_species, 
                                            complete_trajectory=False, 
                                            sampling_times=self.time_windows)
        res = []
        n_iter = self.buffer_params.shape[0]
        for i in range(n_iter//rate):
            parameters = np.concatenate((self.fixed_parameters, self.buffer_params[i*rate,:]))
            samples, _ = sim.run_simulations(parameters)
            res.append(np.mean(samples, axis=0))
        res = np.array(res)
        for i, t in enumerate(self.time_windows):
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


    def plot_performance_index(self, 
                            ind_species: int, 
                            rate: int=200, 
                            save: Tuple[bool, str] =(False, None)):
        """Plots the accurate values of the losses over the iterations.

        Args:
            - **ind_species** (int): Index of the species of interest.
            - **rate** (int, optional): Plotting rate. Defaults to :math:`200`.
            - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. The second argument 
              is the name of the file under which to save the plot. Defaults to (False, None).
        """        
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
        - **crn** (simulation.CRN): CRN to compute the PGD on.
        - **model** (neuralnetwork.NeuralNetwork): MDN used for the gradient descent.
        - **domain** (np.ndarray): Boundaries of the domain in which to project. Has shape :math:`(dim, 2)`.
          **domain[:,0]** defines the lower boundaries for each dimension, **domain[:,1]** defines the 
          upper boundaries for each dimension.
        - **fixed_params** (np.ndarray): Selected values for the fixed parameters.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. 
          Its form is :math:`[t_1, ..., t_L]`, such that the considered time windows are 
          :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match with the final time 
          :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - **loss** (Union[Callable, list]): Loss function used for the gradient descent. If it is a list, each element
          is the loss function for the corresponding time window.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L)`. If None, all targets
          have the same weight. Defaults to None.
        - **length_output** (int, optional): Length of the output of the MDN prediction. Defaults to :math:`200`.
        - **with_correction** (bool, optional): If True, sensitivities are set to zero when controlled parameters
            have no influence on that time window. If False, works with the computed sensitivities. Defaults to True.
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
        r"""Computes the gradient function of the loss evaluated at the expected value with respect to 
        all controlled parameters.

        .. math::

            \xi \mapsto \sum_{i=1}^L w_i \nabla_{\xi} \mathcal{L}_i(E[X_t^{\theta, \xi}])

        Usually, we set :math:`\mathcal{L}_i` of the form :math:`\mathcal{L}_i(x) = (x-\text{target}_i)^2`. 

        Args:
            - **length_output** (int): Length of the output.
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
        """Computes the loss function evaluated at the expected value.

        Args:
            - **length_output** (int): Length of the output.
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
        - **domain** (np.ndarray): Boundaries of the domain in which to project. Has shape :math:`(dim, 2)`.
          **domain[:,0]** defines the lower boundaries for each dimension, **domain[:,1]** defines the 
          upper boundaries for each dimension.
        - **fixed_params** (np.ndarray): Selected values for the fixed parameters.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. 
          Its form is :math:`[t_1, ..., t_L]`, such that the considered time windows are 
          :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match with the final time 
          :math:`t_f`. If there is only one time window, it should be defined as :math:`[t_f]`.
        - **loss** (Union[Callable, list]): Loss function used for the gradient descent. If it is a list, each element
          is the loss function for the corresponding time window.
        - **grad_loss** (Union[Callable, list]): Gradient of the loss function. If it is a list, each element is the gradient
          of the loss function for the corresponding time window.
        - **weights** (np.ndarray, optional):  If True, sensitivities are set to zero when controlled parameters
            have no influence on that time window. If False, works with the computed sensitivities. Defaults to True.
        - :math:`C_r` (int, optional): Value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. 
          Defaults to :math:`50`.
    """       
    def __init__(self,
                crn: simulation.CRN,
                ind_species: int,
                domain: np.ndarray,
                fixed_params: np.ndarray,
                time_windows: np.ndarray,
                loss: Union[Callable, list],
                grad_loss: Union[Callable, list],
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
        """Computes the loss function evaluated at the expected value.
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
        """Computes the gradient function of the loss evaluated at the expected value with respect to 
        all controlled parameters.
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

def control_method(optimizer: ProjectedGradientDescent_CRN, 
                    gamma: float, 
                    n_iter: int,
                    eps: float, 
                    ind_species: int, 
                    targets: np.ndarray,
                    save: Tuple[bool, list] =(True, ['control_values', 
                                                    'experimental_losses', 
                                                    'parameters', 
                                                    'gradients_losses', 
                                                    'real_losses', 
                                                    'exp_results'])) -> Tuple[float, np.ndarray, float]:
    """Computes the PGD, plots the results and saves the algorithm and results data.

    Args:
        - **optimizer** (ProjectedGradientDescent_CRN): Either of type ProjectedGradientDescent_MDn or
          ProjectedGradientDescent_FSP.
        - **gamma** (float): Step size.
        - :math:`n_{\text{iter}}** (int): Maximal number of iterations allowed for the gradient descent.
        - **eps** (float): Tolerance rate. The algorithm halts when the gradient of the loss value is smaller than **eps**.
        - **ind_species** (int): Index of the species of interest.
        - **targets** (np.ndarray): Target values at each time point.
          If None, no target is plotted.
        - **save** (Tuple[bool, list], optional): If the first argument is True, saves the plots. The second argument 
          is a list of the names of the files under which to save the plots. 
          Defaults to (True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results']).

    Returns:
        - Running time of the algorithm.
        - Final values for the controlled parameters.
        - Final loss value.
    """    
    start = time.time()
    control_params, loss_value, i = optimizer.optimisation(gamma=gamma, 
                                                        n_iter=n_iter,
                                                        eps=eps)
    end = time.time()
    print('Time: ', end-start)
    print('Number of iterations: ', i)
    print('Control parameters: ', control_params)
    print('Final loss: ', loss_value)
    optimizer.plot_control_values(save=(save[0], save[1][0]))
    optimizer.plot_losses_trajectory(save=(save[0], save[1][1]))
    optimizer.plot_control_params_trajectory(save=(save[0], save[1][2]))
    optimizer.plot_gradients_trajectory(save=(save[0], save[1][3]))
    optimizer.plot_performance_index(ind_species=ind_species, rate=i//10, save=(save[0], save[1][4]))
    sim = generate_data.CRN_Simulations(optimizer.crn, 
                                        optimizer.time_windows, 
                                        10_000,
                                        ind_species=ind_species, 
                                        complete_trajectory=False, 
                                        sampling_times = np.arange(optimizer.time_windows[-1]+1))
    sim.plot_simulations(np.concatenate((optimizer.fixed_parameters, control_params)), targets=targets, save=(save[0], save[1][5]))
    return end-start, control_params, loss_value

    