import numpy as np
import simulation
import projected_gradient_descent as pgd
import neuralnetwork
from datetime import datetime
from typing import Tuple, Callable, Union


# FSP gradient descent

def pgdFSP(crn: simulation.CRN, 
            ind_species: int, 
            domain: np.ndarray, 
            fixed_params: np.ndarray, 
            time_windows: np.ndarray, 
            loss: Union[Callable, list], 
            grad_loss: Union[Callable, list], 
            cr: int, 
            gamma: float, 
            n_iter: int, 
            eps: float, 
            targets: np.ndarray,
            crn_name: str, 
            weights: np.ndarray =None,
            directory: str ="",
            save: Tuple[bool, list] =(True, ['control_values', 
                                            'experimental_losses', 
                                            'parameters', 
                                            'gradients_losses', 
                                            'real_losses', 
                                            'exp_results'])):
    r"""Performs the PGD with the FSP method, saves the selected parameters for the algorithm and the results in a .txt file,
    plots the results and saves them in CSV files. 

    Args:
        - **crn** (simulation.CRN): CRN to work on. Make sure to specify the derivatives of the propensities if it does not follow
          mass-action kinetics.
        - **ind_species** (int): Index of the species of interest.
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
        - :math:`C_r` (int): Value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. 
          Defaults to :math:`50`.
        - **gamma** (float): Step size.
        - :math:`n_iter` (int): Maximal number of iterations allowed for the gradient descent.
        - **eps** (float): Tolerance rate. The algorithm halts when the squared norm of the gradient of the loss value is smaller than **eps**.
        - **targets** (np.ndarray): Target values at each time point.
        - **crn_name** (str): Name of the CRN to use for the files.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L)`. If None, all targets
          have the same weight. Defaults to None.
        - **directory** (str, optional): Name of the directory under which to save the files. Defaults to "", which means no directory.
        - **save** (Tuple[bool, list], optional): If the first argument is True, saves the file. The second argument is the name of the file under 
          which to save the plot. Defaults to (True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results']).
    """
    optimizerFSP = pgd.ProjectedGradientDescent_FSP(crn=crn,
                                                    ind_species=ind_species,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    grad_loss=grad_loss,
                                                    weights=weights,
                                                    cr=cr)
    final_time, control_params, loss_value = pgd.control_method(optimizer=optimizerFSP,
                                                                gamma=gamma,
                                                                n_iter=n_iter,
                                                                eps=eps,
                                                                ind_species=ind_species,
                                                                targets=targets,
                                                                save=save)
    with open(f'{directory}data_pgdMDN_{crn_name}_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt', 'w') as f:
        f.write(f'domain: {domain}\n')
        f.write(f'fixed parameters: {fixed_params}\n')
        f.write(f'time_windows: {time_windows}\n')
        if isinstance(loss, list):
            for k, l in enumerate(loss):
                f.write(f'loss n°{k}: {l(0)}, {l(1)}, {l(2)}\n')
        else:
            f.write(f'loss: {loss(0)}, {loss(1)}, {loss(2)}\n')
        f.write(fr'$c_r$: {cr}\n')
        f.write(f'gamma: {gamma}\n')
        f.write(f'n_iter: {n_iter}\n')
        f.write(f'targets: {targets}\n')
        f.write(f'PGD time: {final_time}\n')
        f.write(f'Final parameters: {control_params}\n')
        f.write(f'Final loss: {loss_value}')

# MDN gradient descent

def pgdMDN(crn: simulation.CRN, 
            model: neuralnetwork.NeuralNetwork,
            ind_species: int,
            domain: np.ndarray, 
            fixed_params: np.ndarray, 
            time_windows: np.ndarray, 
            loss: Union[Callable, list],
            gamma: float, 
            n_iter: int,
            eps: float,
            targets: np.ndarray,
            crn_name: str, 
            weights: np.ndarray =None,
            directory: str ="",
            save: Tuple[bool, list] =(True, ['control_values', 
                                            'experimental_losses', 
                                            'parameters', 
                                            'gradients_losses', 
                                            'real_losses', 
                                            'exp_results'])):
    r"""Performs the PGD using a MDN, saves the selected parameters for the algorithm and the results in a .txt file,
    plots the results and saves them in CSV files. 

    Args:
        - **crn** (simulation.CRN): CRN to work on. Make sure to specify the derivatives of the propensities if it does not follow
          mass-action kinetics.
        - **model** (neuralnetwork.NeuralNetwork): MDN model to use for the PGD.
        - **ind_species** (int): Index of the species of interest.
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
        - **gamma** (float): Step size.
        - :math:`n_iter` (int): Maximal number of iterations allowed for the gradient descent.
        - **eps** (float): Tolerance rate. The algorithm halts when the squared norm of the gradient of the loss value is smaller than **eps**.
        - **targets** (np.ndarray): Target values at each time point.
        - **crn_name** (str): Name of the CRN to use for the files.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L)`. If None, all targets
          have the same weight. Defaults to None.
        - **directory** (str, optional): Name of the directory under which to save the files. If not "", must end with "/". Defaults to "", which means no directory.
        - **save** (Tuple[bool, list], optional): If the first argument is True, saves the file. The second argument is the name of the file under 
          which to save the plot. Defaults to (True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results']).
    """
    optimizerMDN = pgd.ProjectedGradientDescent_MDN(crn=crn,
                                                    model=model,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    weights=weights)
    final_time, control_params, loss_value = pgd.control_method(optimizer=optimizerMDN,
                                                                gamma=gamma,
                                                                n_iter=n_iter,
                                                                eps=eps,
                                                                ind_species=ind_species,
                                                                targets=targets,
                                                                save=save)
    with open(f'{directory}data_pgdMDN_{crn_name}_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt', 'w') as f:
        f.write(f'domain: {domain}\n')
        f.write(f'fixed parameters: {fixed_params}\n')
        f.write(f'time_windows: {time_windows}\n')
        if isinstance(loss, list):
            for k, l in enumerate(loss):
                f.write(f'loss n°{k}: {l(0)}, {l(1)}, {l(2)}\n')
        else:
            f.write(f'loss: {loss(0)}, {loss(1)}, {loss(2)}\n')
        f.write(f'gamma: {gamma}\n')
        f.write(f'n_iter: {n_iter}\n')
        f.write(f'targets: {targets}\n')
        f.write(f'PGD time: {final_time}\n')
        f.write(f'Final parameters: {control_params}\n')
        f.write(f'Final loss: {loss_value}')



if __name__ == '__main__':

    from CRN4_control import propensities_bursting_gene as propensities
    import save_load_MDN

    def loss3(x):
        return (x-3)**2

    def loss2(x):
        return (x-2)**2

    def loss1(x):
        return (x-1)**2

    def loss4(x):
        return (x-0.5)**2

    def grad_loss3(probs, gradient):
        return 2*gradient*(probs-3)

    def grad_loss2(probs, gradient):
        return 2*gradient*(probs-2)

    def grad_loss1(probs, gradient):
        return 2*gradient*(probs-1)

    def grad_loss4(probs, gradient):
        return 2*gradient*(probs-0.5)

    crn = simulation.CRN(stoichiometry_mat=propensities.stoich_mat, 
                        propensities=propensities.propensities, 
                        propensities_drv=None, 
                        init_state=propensities.init_state, 
                        n_fixed_params=3, 
                        n_control_params=1)
    domain = np.stack([np.array([1e-5, 5.])]*4)
    fixed_params = np.array([1., 2., 1.])
    time_windows = np.array([5, 10, 15, 20])


    # pgdFSP(crn=crn,
    #         ind_species=propensities.ind_species,
    #         domain=domain,
    #         fixed_params=fixed_params,
    #         time_windows=time_windows,
    #         loss=loss1,
    #         grad_loss=grad_loss1,
    #         cr=50,
    #         gamma=0.1,
    #         n_iter=1_000,
    #         eps=1e-7,
    #         targets=np.array([[5., 1.], [10., 1.], [15., 1.], [20., 1.]]),
    #         crn_name='CRN4')

    model = save_load_MDN.load_MDN_model('CRN4_control/saved_models/CRN4_model2.pt')

    pgdMDN(crn=crn,
            ind_species=propensities.ind_species,
            model=model,
            domain=domain,
            fixed_params=fixed_params,
            time_windows=time_windows,
            loss=loss4,
            gamma=0.01,
            n_iter=10_000,
            eps=1e-7,
            targets=np.array([[5., 0.5], [10., 0.5], [15., 0.5], [20., 0.5]]),
            crn_name='CRN4')

