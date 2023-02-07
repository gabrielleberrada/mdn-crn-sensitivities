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
    r"""Performs the PGD with the FSP method, saves the selected parameters for the algorithm and the results in a ``.txt`` file,
    plots the results and saves them in CSV files.

    Args:
        - **crn** (simulation.CRN): CRN to work on. Make sure to specify the derivatives of the propensities if the CRN does not follow
          mass-action kinetics.
        - **ind_species** (int): Index of the species of interest.
        - **domain** (np.ndarray): Boundaries of the domain in which to project. Has shape :math:`(\text{dim}, 2)`.
          **domain[:,0]** defines the lower boundaries for each dimension, **domain[:,1]** defines the 
          upper boundaries for each dimension.
        - **fixed_params** (np.ndarray): Selected values for the fixed parameters.
        - **time_windows** (np.ndarray): Time windows during which all parameters are fixed. 
          Its form is :math:`[t_1, ..., t_L]`, such that the considered time windows are 
          :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match with the final time 
          :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **loss** (Union[Callable, list]): Loss function used for the gradient descent. If it is a list, each element
          is the loss function for the corresponding time window.
        - **grad_loss** (Union[Callable, list]): Gradient of the loss function. If it is a list, each element is the gradient
          of the loss function for the corresponding time window.
        - :math:`C_r` (int): Value such that :math:`(0, .., 0, C_r)` is the last value in the truncated space. 
          Defaults to :math:`50`.
        - **gamma** (float): Step size :math:`\gamma`.
        - :math:`n_{\text{iter}}` (int): Maximal number of iterations allowed for the gradient descent.
        - **eps** (float): Tolerance rate :math:`\varepsilon`. The algorithm halts when the squared norm of the gradient of the loss value is smaller than :math:`\varepsilon`.
        - **targets** (np.ndarray): Target values at each time point.
        - **crn_name** (str): Name of the CRN to use for the files.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L,)`. If None, all targets
          have the same weight. Defaults to None.
        - **directory** (str, optional): Name of the directory under which to save the files. Must end with "/". Defaults to "", which means no directory.
        - **save** (Tuple[bool, list], optional): If the first argument is True, saves the file. The second argument is the name of the file under 
          which to save the plot. Defaults to (True, ["control_values", "experimental_losses", "parameters", "gradients_losses", "real_losses", "exp_results"]).
    """
    optimiserFSP = pgd.ProjectedGradientDescent_FSP(crn=crn,
                                                    ind_species=ind_species,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    grad_loss=grad_loss,
                                                    weights=weights,
                                                    cr=cr)
    final_time, control_params, loss_value = pgd.control_method(optimiser=optimiserFSP,
                                                                gamma=gamma,
                                                                n_iter=n_iter,
                                                                eps=eps,
                                                                ind_species=ind_species,
                                                                targets=targets,
                                                                plot_performance=False,
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
        f.write(f'c_r: {cr}\n')
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
    r"""Performs the PGD using a MDN, saves the selected parameters for the algorithm and the results in a `.txt` file,
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
          :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **loss** (Union[Callable, list]): Loss function used for the gradient descent. If it is a list, each element
          is the loss function for the corresponding time window.
        - **gamma** (float): Step size :math:`\gamma`.
        - :math:`n_iter` (int): Maximal number of iterations allowed for the gradient descent.
        - **eps** (float): Tolerance rate :math:`\varepsilon`. The algorithm halts 
          when the squared norm of the gradient of the loss value is smaller than :math:`\varepsilon`.
        - **targets** (np.ndarray): Target values at each time point.
        - **crn_name** (str): Name of the CRN to use for the files.
        - **weights** (np.ndarray, optional): Weights of each target. Has shape :math:`(L,)`. If None, all targets
          have the same weight. Defaults to None.
        - **directory** (str, optional): Name of the directory under which to save the files. Must end with "/".
          If not "", must end with "/". Defaults to "", which means no directory.
        - **save** (Tuple[bool, list], optional): If the first argument is True, saves the file. The second argument is the name of the file under 
          which to save the plot. Defaults to (True, ["control_values", "experimental_losses", "parameters", "gradients_losses", "real_losses", "exp_results"]).
    """
    optimiserMDN = pgd.ProjectedGradientDescent_MDN(crn=crn,
                                                    model=model,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    weights=weights)
    final_time, control_params, loss_value = pgd.control_method(optimiser=optimiserMDN,
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
        f.write(f'epsilon: {eps}\n')
        f.write(f'n_iter: {n_iter}\n')
        f.write(f'targets: {targets}\n')
        f.write(f'PGD time: {final_time}\n')
        f.write(f'Final parameters: {control_params}\n')
        f.write(f'Final loss: {loss_value}\n')



if __name__ == '__main__':

  from CRN4_control import propensities_bursting_gene as propensities
  # from CRN6_toggle_switch import propensities_toggle as propensities
  # from CRN2_control import propensities_production_degradation as propensities

  crn = simulation.CRN(stoichiometry_mat=propensities.stoich_mat,
                    propensities=propensities.propensities,
                    init_state=propensities.init_state,
                    propensities_drv=None,
                    n_fixed_params=3,
                    n_control_params=1)
  domain = np.stack([np.array([1e-10, 5.])]*4)
  fixed_params = np.array([1., 2., 1.])
  time_windows=np.array([5, 10, 15, 20])

  def loss05(x):
    return (x-0.5)**2

  def grad_loss05(x, grad_x):
    return 2*grad_x*(x-0.5)

  def loss175(x):
    return (x-1.75)**2

  def grad_loss175(x, grad_x):
    return 2*grad_x*(x-1.75)

  def loss15(x):
    return (x-1.5)**2

  def grad_loss15(x, grad_x):
    return 2*grad_x*(x-1.5)

  def loss1(x):
    return (x-1)**2

  def grad_loss1(x, grad_x):
    return 2*grad_x*(x-1)


  pgdFSP(crn=crn,
        ind_species=propensities.ind_species,
        domain=domain,
        fixed_params=fixed_params,
        time_windows=time_windows,
        loss=loss05,
        grad_loss=grad_loss05,
        gamma=0.01,
        n_iter=20_000,
        eps=2.7e-6,
        targets=np.array([[5., 0.5], [10., 0.5], [15., 0.5], [20., 0.5]]),
        crn_name='ctrl_bg',
        cr=100,
        directory = 'CRN4_control/gradient_descent/FSP/target05/',
        save=(True, ['CRN4_control/gradient_descent/FSP/target05/control_values', 
                    'CRN4_control/gradient_descent/FSP/target05/experimental_losses', 
                    'CRN4_control/gradient_descent/FSP/target05/parameters', 
                    'CRN4_control/gradient_descent/FSP/target05/gradients_losses', 
                    'CRN4_control/gradient_descent/FSP/target05/real_losses', 
                    'CRN4_control/gradient_descent/FSP/target05/exp_results']))

  # import save_load_MDN

  # model2 = save_load_MDN.load_MDN_model('CRN2_control/saved_models/CRN2_model2.pt')

  # pgdMDN(crn=crn,
  #       model=model2,
  #       domain=domain,
  #       fixed_params=fixed_params,
  #       time_windows=time_windows,
  #       loss=loss1,
  #       gamma=0.01,
  #       n_iter=20,
  #       eps=1e-7,
  #       ind_species=propensities.ind_species,
  #       targets=np.array([[5., 1.], [10., 1.], [15., 1.], [20., 1.]]),
  #       crn_name='ctrl_pureprod',
  #       save=(False, ['gradient_descent/MDN/constant_targets/target1/control_values', 
  #                   'gradient_descent/MDN/constant_targets/target1/experimental_losses', 
  #                   'gradient_descent/MDN/constant_targets/target1/parameters', 
  #                   'gradient_descent/MDN/constant_targets/target1/gradients_losses', 
  #                   'gradient_descent/MDN/constant_targets/target1/real_losses', 
  #                   'gradient_descent/MDN/constant_targets/target1/exp_results']))