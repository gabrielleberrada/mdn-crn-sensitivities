import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import neuralnetwork
import get_sensitivities
import seaborn
import math
import fsp
import get_fi
import simulation
import time
from typing import Callable, Tuple

# Plots the probability mass function or the sensitivity of the likelihood
def plot_model(to_pred: torch.tensor,
            models: list, 
            up_bound: int,
            time_windows: np.ndarray,
            n_comps: int,
            index_names: Tuple[str, str] =("Probabilities", r"Abundance of species $S$"), 
            plot_test_result: Tuple[bool, torch.tensor] =(False, None), 
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, Tuple[int], int, int, int] = (False, None),
            plot: Tuple[str, int] =("probabilities", None),
            save: Tuple[bool, str] =(False, None)):
    r"""Plots distributions estimated with various methods for a single set of time and parameters and for a specified CRN. 

    Args:
        - **to_pred** (torch.tensor): Time and parameters in the form requested by the Mixture Density Networks: 
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **models** (list): Mixture Density Network models to use.
        - **up_bound** (int): Upper boundary of the predicted distribution.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **n_comps** (int): Number of components of the predicted mixture.
        - **index_names** (Tuple[str, str], optional): Labels of x-axis and y-axis. Defaults to ("Probabilities", "Abundance of species S").
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If the first argument is True, plots the expected results 
          from the datasets for the chosen set of parameters. The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If the first argument is True, plots the exact results 
          for the chosen set of parameters. The second argument is the function that computes the exact results. Defaults to (False, None).        
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray], optional): If the first argument is True, 
          plots the estimated results with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the distribution with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Integer such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`. 
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from one time window to another.
                
        - **plot** (Tuple[str, int], optional): The first argument is either "probabilities" to plot a probability distribution, or "sensitivities"
          to plot a sensitivity of the likelihood distribution. If it is "sensitivities", the second argument is the index of the parameter 
          such that it plots the sensitivities with respect to this parameter. Defaults to ("probabilities", None).
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. The second argument is the name of the file 
          under which to save the plot. Defaults to (False, None).
    """            
    # prediction
    x = torch.arange(up_bound).repeat(1, n_comps,1).permute([2,0,1])
    preds = []
    ymin = 0
    ymax = 0
    for i, model in enumerate(models):
        if plot[0] == 'probabilities':
            y_pred = neuralnetwork.mix_pdf(model, to_pred, x)
        elif plot[0] == 'sensitivities':
            y_pred = get_sensitivities.sensitivities(to_pred, model, length_output=up_bound)[:, plot[1]+1]
        y_pred = y_pred.detach().numpy()
        ymin = min(ymin, y_pred.min())
        ymax = max(ymax, y_pred.max())
        pred = pd.DataFrame([np.squeeze(y_pred), np.arange(up_bound)], index = index_names).transpose()
        pred['Model'] = f'training{i+1}'
        preds.append(pred)
    if plot_test_result[0]:
        result = plot_test_result[1]
        if torch.is_tensor(result):
            result = result.detach().numpy()
        test_result = pd.DataFrame([np.squeeze(result), np.arange(up_bound)], index = index_names).transpose()
        test_result['Model'] = 'SSA simulation'
        preds.append(test_result)
    if plot_fsp_result[0]:
        n_time_windows = len(time_windows)
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[5],
                            n_fixed_params=plot_fsp_result[7],
                            n_control_params=plot_fsp_result[8],
                            propensities_drv=plot_fsp_result[3])
        stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=plot[1], cr=plot_fsp_result[4])
        # for now, time_window = [0, t], parameters has shape (n_params + 1)
        fixed_parameters = np.stack([to_pred[1:plot_fsp_result[7]+1].numpy()]*n_time_windows)
        control_parameters = to_pred[plot_fsp_result[7]+1:].numpy().reshape(n_time_windows, plot_fsp_result[8])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        if plot[0] == 'probabilities':
            results_fsp = stv_calculator.marginal(to_pred[:1].numpy(), 
                                                time_windows, parameters, 
                                                ind_species=plot_fsp_result[6],
                                                with_stv=False)[:,0,0]
        if plot[0] == 'sensitivities':
            results_fsp = stv_calculator.marginal(to_pred[:1].numpy(), 
                                                time_windows, parameters, 
                                                ind_species=plot_fsp_result[6], 
                                                with_stv=True)
            if plot[1] < crn.n_fixed_params:
                results_fsp = results_fsp[:,0,1]
            elif crn.n_control_params < 2:
                results_fsp = results_fsp[:, 0, 1+plot[1]-crn.n_fixed_params]
            else:
                results_fsp = results_fsp[:, 0, 1+(plot[1]-crn.n_fixed_params)%crn.n_control_params]
        length = min(up_bound, plot_fsp_result[4])
        ymin = min(ymin, results_fsp.min())
        ymax = max(ymax, results_fsp.max())
        fsp_result = pd.DataFrame([results_fsp[:length], np.arange(length)], index=index_names).transpose()
        fsp_result['Model'] = 'FSP estimation'
        preds.append(fsp_result)
    if plot_exact_result[0]:
        parameters = []
        for tens in to_pred:
            parameters.append(tens.numpy())
        exact_result = pd.DataFrame([[plot_exact_result[1](k, parameters) for k in range(up_bound)],
                                    np.arange(up_bound)], index = index_names).transpose()
        exact_result['Model'] = 'exact result'
        preds.append(exact_result)
    data = pd.concat(preds, ignore_index=True)
    # params = [np.round(param.numpy(), 2) for param in to_pred]
    fig = seaborn.relplot(data=data, x=index_names[1], y=index_names[0], hue='Model', style='Model', aspect=1.5, kind='line',
        dashes={'training1': '', 'training2': '', 'training3': '', 'exact result': (5, 5), 'FSP estimation': (1, 1), 'SSA simulation': (1, 1)}) #.set(title=fr'{plot[0]} plot for {crn_name} with $t=${params[0]}, $\theta=${params[1:]}')
    fig._legend.remove()
    plt.legend(loc='best')
    plt.ylim(ymin-0.05, ymax+0.05)
    if save[0]:
        plt.savefig(save[1])
    plt.show()

def multiple_plots(to_pred: list,
            models: list,
            up_bound: int,
            time_windows: np.ndarray,
            n_comps: int,
            index_names: Tuple[str] =('Probabilities', r'Abundance of species $S$'),
            plot_test_result: Tuple[bool, torch.tensor] =(False, None),
            plot_exact_result: Tuple[bool, Callable] =(False, None),
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            plot: Tuple[str, int] =('probabilities', None),
            n_col: int =2,
            save: Tuple[bool, str] =(False, None)):
    r"""Plots distributions estimated with various methods for multiple sets of time and parameters and for a specified CRN.

    Args:
        - **to_pred** (list): Time and parameters in the form requested by the Mixture Density Networks:
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **models** (list): Mixture Density Network models to use.
        - **up_bound** (int): Upper boundary of the predicted distributions.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **n_comps** (int): Number of components of the predicted mixture.
        - **index_names** (Tuple[str], optional): Labels of x-axis and y-axis. Defaults to ("Probabilities", "Abundance of species S").
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If the first argument is True, plots the expected results 
          from the datasets for the chosen set of parameters. The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If the first argument is True, plots the exact results for the
          chosen set of parameters. The second argument is the function that computes the exact results. Defaults to (False, None).
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): If the first argument is True, 
          plots the estimated results with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the distribution with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Value such that the projection of :math:`(0, ..., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (np.ndarray): Initial state.
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from one time window to another.

        - **plot** (Tuple[str, int], optional): The first argument is either "probabilities" to plot a probability distribution, or "sensitivities" 
          to plot a sensitivity of the likelihood distribution. If it is "sensitivities", second argument is the index of the parameter 
          such that it plots the sensitivities with respect to this parameter. Defaults to ("probabilities", None).
        - **n_col** (int, optional): Number of columns to plot. Defaults to :math:`2`.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. Second element is the name of the file 
          in which to save the plot. Defaults to (False, None).
    """          
    n = len(to_pred)
    if n == 1:
            plot_model(to_pred[0], models, up_bound, n_comps, index_names, plot_test_result, plot_exact_result, plot_fsp_result, plot, save)
    else:
        _, axes = plt.subplots(math.ceil(n/n_col), n_col, figsize=(3*n,3*n))
        ymin = 0
        ymax = 0
        # in case there is only one row
        axes = np.reshape(axes, (-1, n_col))
        for k, to_pred_ in enumerate(to_pred):
            x = torch.arange(up_bound[k]).repeat(1, n_comps,1).permute([2,0,1])
            preds = []
            for i, model in enumerate(models):
                if plot[0] == 'probabilities':
                    y_pred = neuralnetwork.mix_pdf(model, to_pred_, x)
                elif plot[0] == 'sensitivities':
                    y_pred = get_sensitivities.sensitivities(to_pred_, model, length_output=up_bound[k])[:, plot[1]+1]
                y_pred = y_pred.detach().numpy()
                ymin = min(ymin, y_pred.min())
                ymax = max(ymax, y_pred.max())
                pred = pd.DataFrame([np.squeeze(y_pred), np.arange(up_bound[k])], index = index_names).transpose()
                pred['Model'] = f'training{i+1}'
                preds.append(pred)
            if plot_test_result[0]:
                result = plot_test_result[1][k]
                if torch.is_tensor(result):
                    result = result.detach().numpy()
                test_result = pd.DataFrame([np.squeeze(result), np.arange(up_bound[k])], index = index_names).transpose()
                test_result['Model'] = 'SSA simulation'
                preds.append(test_result)
            if plot_fsp_result[0]:
                n_time_windows = len(time_windows)
                crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                                    propensities=plot_fsp_result[2], 
                                    init_state=plot_fsp_result[5],
                                    n_fixed_params=plot_fsp_result[7],
                                    n_control_params=plot_fsp_result[8],
                                    propensities_drv=plot_fsp_result[3])
                stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=plot[1], cr=plot_fsp_result[4])
                fixed_parameters = np.stack([to_pred_[1:plot_fsp_result[7]+1].numpy()]*n_time_windows)
                control_parameters = to_pred_[plot_fsp_result[7]+1:].numpy().reshape(n_time_windows, plot_fsp_result[8])
                parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
                if plot[0] == 'probabilities':
                    results_fsp = stv_calculator.marginal(to_pred_[:1].numpy(), 
                                                        time_windows, 
                                                        parameters, 
                                                        ind_species=plot_fsp_result[6],
                                                        with_stv=False)[:,0,0]
                if plot[0] == 'sensitivities':
                    results_fsp = stv_calculator.marginal(to_pred_[:1].numpy(), 
                                                        time_windows, parameters, 
                                                        ind_species=plot_fsp_result[6],
                                                        with_stv=True)
                    if plot[1] < crn.n_fixed_params:
                        results_fsp = results_fsp[:,0,1]
                    elif crn.n_control_params < 2:
                        results_fsp = results_fsp[:, 0, 1+plot[1]-crn.n_fixed_params]
                    else:
                        results_fsp = results_fsp[:, 0, 1+(plot[1]-crn.n_fixed_params)%crn.n_control_params]
                length = min(up_bound[k], plot_fsp_result[4])
                ymin = min(ymin, results_fsp.min())
                ymax = max(ymax, results_fsp.max())
                fsp_result = pd.DataFrame([results_fsp[:length], np.arange(length)], index=index_names).transpose()
                fsp_result['Model'] = 'FSP estimation'
                preds.append(fsp_result)
            if plot_exact_result[0]:
                parameters = []
                for tens in to_pred_:
                    parameters.append(tens.numpy())
                exact_result = pd.DataFrame([[plot_exact_result[1](j, parameters) for j in range(up_bound[k])], 
                                            np.arange(up_bound[k])], index = index_names).transpose()
                exact_result['Model'] = 'exact result'
                preds.append(exact_result)
            data = pd.concat(preds, ignore_index=True)
            seaborn.lineplot(ax=axes[k//n_col, k%n_col], data=data, x=index_names[1], y=index_names[0], hue='Model', style='Model',
                dashes={'training1': '', 'training2': '', 'training3': '', 'exact result': (5, 5), 'FSP estimation': (1, 1), 'SSA simulation': (1, 1)})
            # axes[k//n_col, k%n_col].annotate(f'({k})', xy=(length*0.9, ymax*0.8), xycoords='data', fontsize=11)
        plt.setp(axes, ylim=(ymin-0.05, ymax+0.05))
        plt.subplots_adjust(hspace=0.01)
        # fig.suptitle(f'{plot[0]} plot for params {params[1:]}')
        if save[0]:
            plt.savefig(save[1])
        plt.show()




# Plots the Fisher information

def fi_table(time_samples: np.ndarray, 
            params: np.ndarray, 
            ind_param: int, 
            time_windows: np.ndarray,
            models: Tuple[bool, list, int] =(False, None, 4),
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            up_bound: int =200,
            out_of_bounds_index: int =None,
            save: Tuple[bool, str] =(False, None)):
    r"""Plots a table of the diagonal element of the Fisher Information estimated by various methods at various times.

    Args:
        - **time_samples** (list): Sampling time.
        - **params** (list): Parameters of the propensity functions in the form requested by the Mixture Density Networks:
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **ind_param** (int): Index of the estimated Fisher Information diagonal value.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the Fisher Information 
          with MDN models. Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the Fisher Information with MDN models.
                2. **models_list** (list): List of MDN models to use.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to compute the exact value of the Fisher Information. 
          Defaults to (False, None).
                
                1. **exact_value** (bool): If True, computes the exact value of the Fisher Information.
                2. **fisher_information_function** (Callable): Exact function of the Fisher Information value.

        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the Fisher Information with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Value such that the projection of :math:`(0, ..., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (np.ndarray): Initial state.
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from one time window to another.

        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to :math:`200`.
        - **out_of_bounds_index** (int, optional): Index of the first time out of the training range in **time_samples**. If None, all
          times are within the training range. Defaults to None.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file under which to save the plot. Defaults to (False, None).
    """            
    rows = [fr'$t={t}$' for t in time_samples]
    n_rows = len(time_samples)
    # compute probabilities and sensitivities with the neural networks
    if models[0]:
        probabilities_m = np.zeros((len(time_samples),up_bound))
        stv_m = np.zeros((len(time_samples),up_bound, len(params)))
        for model in models[1]:
            for i, t in enumerate(time_samples):
                to_pred = torch.concat((torch.tensor([t]), torch.tensor(params)))
                sens, probs = get_sensitivities.sensitivities(to_pred, model, length_output=up_bound, with_probs=True)
                probabilities_m[i,:] += probs[:,0].numpy()
                stv_m[i,:,:] += sens[:,1:].numpy()
        probabilities_m /= len(models[1])
        stv_m /= len(models[1])
        # compute FIM
        predicted_fi = np.zeros(n_rows)
        for i in range(n_rows):
            fim_m = get_fi.fisher_information_t(probabilities_m[i,:], stv_m[i,:,:])
            predicted_fi[i] = fim_m[ind_param, ind_param]
    # compute probabilities and sensitivities of probabilities with the FSP
    if plot_fsp_result[0]:
        n_time_windows = len(time_windows)
        start = time.time()
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[5],
                            n_fixed_params=plot_fsp_result[7],
                            n_control_params=plot_fsp_result[8],
                            propensities_drv=plot_fsp_result[3])
        stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=ind_param, cr=plot_fsp_result[4])
        fixed_parameters = np.stack([params[:plot_fsp_result[7]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[7]:].reshape(n_time_windows, plot_fsp_result[8])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        length = min(up_bound, plot_fsp_result[4])
        results_fsp = stv_calculator.marginal(time_samples, 
                                            time_windows, 
                                            parameters, 
                                            plot_fsp_result[6],
                                            with_stv=True)[:length,:,:]
        end = time.time()
        print(end-start)
        fsp_fi = np.zeros(n_rows)
        if ind_param < crn.n_fixed_params:
            index = 1
        elif crn.n_control_params < 2:
            index = 1 + ind_param - crn.n_fixed_params
        else:
            index = 1 + (ind_param - crn.n_fixed_params) % crn.n_control_params
        for i in range(n_rows):
            fsp_fi[i] = get_fi.fisher_information_t(results_fsp[:,i,0], results_fsp[:,i,index:])[0,0]
    columns = []
    data = []
    # gathering data
    if models[0]:
        columns.append('Predicted with MDN (mean)')
        data.append(np.round(predicted_fi,3))
    if plot_fsp_result[0]:
        columns.append('Estimated with FSP')
        data.append(np.round(fsp_fi, 3))
    if plot_exact_result[0]:
        columns.append('Exact value')
        exact_fi = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            exact_fi[i] = plot_exact_result[1](t, params)
        data.append(np.round(exact_fi,3))
    if len(data)==1:
        data = np.array(data).T
    else:
        data = np.stack(data, axis=-1)
    if out_of_bounds_index != None:
        data = np.insert(data.astype('str'), out_of_bounds_index, '...', axis=0)
        rows.insert(out_of_bounds_index, '...')
    #plot
    fig, ax = plt.subplots(figsize=(10,3))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = plt.table(cellText=data, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center', colWidths=[1.]*len(columns))
    table.set_fontsize(14)
    table.scale(0.4,1.6)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    if save[0]:
        plt.savefig(save[1])
    plt.show()


# Plot Fisher information bars

def fi_barplots(time_samples: np.ndarray, 
            params: np.ndarray, 
            ind_param: int, 
            time_windows: np.ndarray,
            models: Tuple[bool, list, int] =(False, None, 4),
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            up_bound: int =200,
            save: Tuple[bool, str] =(False, None),
            colors: list =["blue", "darkorange", "forestgreen"],
            mean: bool =True):
    r"""Plots rectangular bars to visualize the diagonal element of the Fisher Information estimated by various methods at various times.

    Args:
        - **time_samples** (np.ndarray): Sampling times.
        - **params** (np.ndarray): Parameters of the propensity functions in the form requested by the Mixture Density Networks:
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **ind_param** (int): Index of the estimated Fisher Information diagonal value.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the Fisher Information 
          with MDN models. Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the Fisher Information with MDN models.
                2. **models_list** (list): List of MDN models from which to estimate the Fisher Information.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to compute the exact value of the Fisher Information. 
          Defaults to (False, None).
                
                1. **exact_value** (bool): If True, computes the exact value of the Fisher Information.
                2. **fisher_information_function** (Callable): Exact function of the Fisher Information exact value.

        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the Fisher Information with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Value such that the projection of :math:`(0, ..., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`.
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from one time window to another.

        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to :math:`200`.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file under which to save the plot. Defaults to (False, None).
        - **colors** (list, optional): Chosen colors for the bars. Defaults to ["blue", "darkorange", "forestgreen"].
        - **mean** (bool, optional): Indicates whether to compute the mean of the MDN values or to plot a bar for each MDN value. Defaults to True.
    """            
    n_rows = len(time_samples)
    preds=[]
    index_names = ('Fisher Information', 'Time')
    # compute probabilities and sensitivities with the neural networks
    if models[0]:
        model_results = []
        for j, model in enumerate(models[1]):
            probabilities_m = np.zeros((len(time_samples),up_bound))
            stv_m = np.zeros((len(time_samples),up_bound, len(params)))
            for i, t in enumerate(time_samples):
                to_pred = torch.concat((torch.tensor([t]), torch.tensor(params)))
                sens, probs = get_sensitivities.sensitivities(to_pred, model, length_output=up_bound, with_probs=True)
                probabilities_m[i,:] += probs[:,0].numpy()
                stv_m[i,:,:] += sens[:,1:].numpy()
            # compute FIM
            predicted_fi = np.zeros(n_rows)
            for i in range(n_rows):
                fim_m = get_fi.fisher_information_t(probabilities_m[i,:], stv_m[i,:,:])
                predicted_fi[i] = fim_m[ind_param, ind_param]
            model_results.append(np.round(predicted_fi, 3))
        if mean:
            pred = pd.DataFrame([list(np.mean(model_results, axis=0)), time_samples], index = index_names).transpose()
            pred['Model'] = 'MDN (mean)'
            preds.append(pred)
        else:
            for j, model in enumerate(models[1]):
                pred = pd.DataFrame([np.round(model_results[j], 3), time_samples], index = index_names).transpose()
                pred['Model'] = f'MDN {j+1}'
                preds.append(pred)
    # compute probabilities and sensitivities of probabilities with the FSP
    if plot_fsp_result[0]:
        n_time_windows = len(time_windows)
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[5],
                            n_fixed_params=plot_fsp_result[7],
                            n_control_params=plot_fsp_result[8],
                            propensities_drv=plot_fsp_result[3])
        stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=ind_param, cr=plot_fsp_result[4])
        fixed_parameters = np.stack([params[:plot_fsp_result[7]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[7]:].reshape(n_time_windows, plot_fsp_result[8])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        length = min(up_bound, plot_fsp_result[4])
        results_fsp = stv_calculator.marginal(time_samples, 
                                            time_windows, 
                                            parameters, 
                                            plot_fsp_result[6],
                                            with_stv=True)[:length,:,:]
        fsp_fi = np.zeros(n_rows)
        if ind_param < crn.n_fixed_params:
            index = 1
        elif crn.n_control_params < 2:
            index = 1 + ind_param - crn.n_fixed_params
        else:
            index = 1 + (ind_param - crn.n_fixed_params) % crn.n_control_params
        for i in range(n_rows):
            fsp_fi[i] = get_fi.fisher_information_t(results_fsp[:,i,0], results_fsp[:,i, index:])[0,0]
        pred = pd.DataFrame([np.round(fsp_fi, 3), time_samples], index = index_names).transpose()
        pred['Model'] = 'FSP estimation'
        preds.append(pred)
    if plot_exact_result[0]:
        exact_fi = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            exact_fi[i] = plot_exact_result[1](t, params)
        pred = pd.DataFrame([np.round(exact_fi,3), time_samples], index = index_names).transpose()
        pred['Model'] = 'Exact value'
        preds.append(pred)
    data = pd.concat(preds, ignore_index=True)
    #plot
    fig = seaborn.catplot(data=data, kind='bar', x=index_names[1], y=index_names[0], aspect=1.5, hue='Model',
        palette=colors)
    fig._legend.remove()
    plt.legend(loc='best')
    if save[0]:
        plt.savefig(save[1])
    plt.show()



# Plot the expectation or the gradient of the expectation
def expect_val_table(time_samples: np.ndarray, 
                    params: np.ndarray,
                    time_windows: np.ndarray,
                    models: Tuple[bool, list, int] =(False, None, 4),
                    plot_exact_result: Tuple[bool, Callable] =(False, None), 
                    plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
                    up_bound: int =200,
                    plot: Tuple[str, int] =("value", None),
                    out_of_bounds_index: int =None,
                    save: Tuple[bool, str] =(False, None)):
    r"""Plots a table of the expectation :math:`E_{\theta, \xi}[X_t]` or its gradient with respect to a specified parameter
    :math:`\frac{\partial E_{\theta, \xi}[X_t]}{\partial \theta_i}` or :math:`\frac{\partial E_{\theta, \xi}[X_t]}{\partial \xi_i}`, 
    estimated by various methods at various times.

    Args:
        - **time_samples** (np.ndarray): Sampling times.
        - **params** (np.ndarray): Parameters of the propensity functions in the form requested by the Mixture Density Networks:
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the expected value with MDN models. 
          Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the expected value with MDN models.
                2. **models_list** (list): List of MDN models from which to estimate the expected value.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to calculate the exact value of the Fisher Information. Defaults to (False, None).
                
                1. **exact_value** (bool): If True, calculates the exact expected value.
                2. **expected_value_function** (Callable): Function that computes the expected value.

        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the expected value with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray[Callable]): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Value such that the projection of :math:`(0, ..., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`.
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from a time window to another.

        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to :math:`200`.
        - **plot** (Tuple[str, int], optional): The first argument is either "value" to compute the expected value, or "gradient" to compute the
          gradient of the expected value. If it is "gradient", the second argument is the index of the parameter such that it computes the gradient
          with respect to this parameter. Defaults to ("value", None). 
        - **out_of_bounds_index** (int, optional): Index of the first time out of the training range in **time_samples**. If None, all
          times are within the training range. Defaults to None.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file under which to save the plot. Defaults to (False, None).
    """
    rows = [fr'$t={t}$' for t in time_samples]
    n_rows = len(time_samples)
    # compute probabilities and sensitivities with the neural networks
    if models[0]:
        predicted_expectation = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            to_pred = torch.concat((torch.tensor([t]), torch.tensor(params)))
            val = 0
            for model in models[1]:
                if plot[0] == 'value':
                    val += get_sensitivities.expected_val(inputs=to_pred, model=model, length_output=up_bound)
                elif plot[0] == 'gradient':
                    val += get_sensitivities.gradient_expected_val(inputs=to_pred, model=model, length_output=up_bound)[plot[1]+1]
            predicted_expectation[i] = val/len(models[1])
    # compute probabilities and sensitivities of probabilities with the FSP
    if plot_fsp_result[0]:
        n_time_windows = len(time_windows)
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[5],
                            n_fixed_params=plot_fsp_result[7],
                            n_control_params=plot_fsp_result[8],
                            propensities_drv=plot_fsp_result[3])
        stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=plot[1], cr=plot_fsp_result[4])
        fixed_parameters = np.stack([params[:plot_fsp_result[7]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[7]:].reshape(n_time_windows, plot_fsp_result[8])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        if plot[0] == 'value':
            fsp_expectation = stv_calculator.expected_val(sampling_times=time_samples, 
                                                        time_windows=time_windows, 
                                                        parameters=parameters, 
                                                        ind_species=plot_fsp_result[6])
        elif plot[0] == 'gradient':
            results_fsp = stv_calculator.gradient_expected_val(sampling_times=time_samples, 
                                                            time_windows=time_windows, 
                                                            parameters=parameters, 
                                                            ind_species=plot_fsp_result[6])
            if plot[1] < crn.n_fixed_params:
                index = 0
            elif crn.n_control_params < 2:
                index = plot[1] - crn.n_fixed_params
            else:
                index = (plot[1] - crn.n_fixed_params) % crn.n_control_params
            fsp_expectation = results_fsp[:,index]
    columns = []
    data = []
    # gathering data
    if models[0]:
        columns.append('Predicted with MDN (mean)')
        data.append(np.round(predicted_expectation,3))
    if plot_fsp_result[0]:
        columns.append('Estimated with FSP')
        data.append(np.round(fsp_expectation, 3))
    if plot_exact_result[0]:
        columns.append('Exact value')
        exact_expectation = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            exact_expectation[i] = plot_exact_result[1](t, params)
        data.append(np.round(exact_expectation,3))
    if len(data)==1:
        data = np.array(data).T
    else:
        data = np.stack(data, axis=-1)
    if out_of_bounds_index != None:
        data = np.insert(data.astype('str'), out_of_bounds_index, '...', axis=0)
        rows.insert(out_of_bounds_index, '...')
    #plot
    fig, ax = plt.subplots(figsize=(10,3))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = plt.table(cellText=data, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center', colWidths=[1.]*len(columns))
    table.set_fontsize(14)
    table.scale(0.4,1.6)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    if save[0]:
        plt.savefig(save[1])
    plt.show()


def expect_val_barplots(time_samples: np.ndarray, 
            params: np.ndarray,
            time_windows: np.ndarray,
            models: Tuple[bool, list, int] =(False, None, 4),
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            up_bound: int =200,
            plot: Tuple[str, int]=("value", None),
            save: Tuple[bool, str] =(False, None),
            colors: list =["blue", "darkorange", "forestgreen"],
            mean: bool =True):
    r"""Plots rectangular bars to visualize the expectation :math:`E_{\theta, \xi}[X_t]` or its gradient with respect to a specified parameter
    :math:`\frac{\partial E_{\theta, \xi}[X_t]}{\partial \theta_i}` or :math:`\frac{\partial E_{\theta, \xi}[X_t]}{\partial \xi_i}`, 
    estimated by various methods at various times.

    Args:
        - **time_samples** (np.ndarray): Sampling times.
        - **params** (np.ndarray): Parameters of the propensity functions in the form requested by the Mixture Density Networks:
          :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]`.
        - **time_windows** (np.ndarray): Time windows during which the parameters do not vary. Its form is :math:`[t_1, ..., t_L]`,
          such that the considered time windows are :math:`[0, t_1], [t_1, t_2], ..., [t_{L-1}, t_L]`. :math:`t_L` must match
          with the final time :math:`t_f`. If there is only one time window, **time_windows** should be defined as :math:`[t_f]`.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the expected value with MDN models. 
          Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the expected value with MDN models.
                2. **models_list** (list): List of MDN models from which to estimate the expected value.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to calculate the exact value of the Fisher Information. Defaults to (False, None).
                
                1. **exact_value** (bool): If True, calculates the exact expected value.
                2. **expected_value_function** (Callable): Function that computes the expected value.

        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the expected value with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray[Callable]): Non-parameterised propensity functions.
                4. **propensities_drv** (np.ndarray): Gradient functions of the propensities with respect to the parameters.
                   Has shape :math:`(M, M_{\theta}+M_{\xi})`. If None, it is assumed that the Chemical Reaction Network follows mass-action kinetics.
                5. :math:`C_r`: Value such that the projection of :math:`(0, ..., 0, C_r)` is the last element of the projected truncated space.
                6. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`.
                7. **ind_species** (int): Index of the species of interest.
                8. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions :math:`M_{\theta}`.
                9. **n_control_params** (int): Number of control parameters required to define the propensity functions :math:`M_{\xi}`.
                   Their values vary from a time window to another.

        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to :math:`200`.
        - **plot** (Tuple[str, int], optional): The first argument is either "value" to compute the expected value, or "gradient" to compute the
          gradient of the expected value. If it is "gradient", the second argument is the index of the parameter such that it computes the gradient
          with respect to this parameter. Defaults to ("value", None).
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file under which to save the plot. Defaults to (False, None).
        - **colors** (list, optional): Chosen colors for the bars. Defaults to ["blue", "darkorange", "forestgreen"].
        - **mean** (bool, optional): Indicates whether to compute the mean of the MDN values or to plot a bar for each MDN value. Defaults to True.
    """      
    n_rows = len(time_samples)
    preds=[]
    index_names = ('Fisher Information', 'Time')
    # compute probabilities and sensitivities with the neural networks
    if models[0]:
        predicted_expectation = np.zeros((n_rows, len(models[1])))
        for i, t in enumerate(time_samples):
            to_pred = torch.concat((torch.tensor([t]), torch.tensor(params)))
            for m, model in enumerate(models[1]):
                if plot[0] == 'value':
                    predicted_expectation[i, m] = get_sensitivities.expected_val(inputs=to_pred, model=model, length_output=up_bound)
                elif plot[0] == 'gradient':
                    predicted_expectation[i, m] = get_sensitivities.gradient_expected_val(inputs=to_pred, model=model, length_output=up_bound)[plot[1]+1]
        if mean:
            pred = pd.DataFrame([list(np.mean(predicted_expectation, axis=1)), time_samples], index = index_names).transpose()
            pred['Model'] = 'MDN (mean)'
            preds.append(pred)
        else:
            for m, model in enumerate(models[1]):
                pred = pd.DataFrame([np.round(predicted_expectation[:, m], 3), time_samples], index = index_names).transpose()
                pred['Model'] = f'MDN {m+1}'
                preds.append(pred)
    # compute probabilities and sensitivities of probabilities with the FSP
    if plot_fsp_result[0]:
        n_time_windows = len(time_windows)
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[5],
                            n_fixed_params=plot_fsp_result[7],
                            n_control_params=plot_fsp_result[8],
                            propensities_drv=plot_fsp_result[3])
        stv_calculator = fsp.SensitivitiesDerivation(crn=crn, n_time_windows=n_time_windows, index=plot[1], cr=plot_fsp_result[4])
        fixed_parameters = np.stack([params[:plot_fsp_result[7]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[7]:].reshape(n_time_windows, plot_fsp_result[8])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        if plot[0] == 'value':
            fsp_expectation = stv_calculator.expected_val(sampling_times=time_samples, 
                                                        time_windows=time_windows, 
                                                        parameters=parameters, 
                                                        ind_species=plot_fsp_result[6])
        elif plot[0] == 'gradient':
            results_fsp = stv_calculator.gradient_expected_val(sampling_times=time_samples, 
                                                            time_windows=time_windows, 
                                                            parameters=parameters, 
                                                            ind_species=plot_fsp_result[6])
            if plot[1] < crn.n_fixed_params:
                index = 0
            elif crn.n_control_params < 2:
                index = plot[1] - crn.n_fixed_params
            else:
                index = (plot[1] - crn.n_fixed_params) % crn.n_control_params
            fsp_expectation = results_fsp[:, index]
        pred = pd.DataFrame([np.round(fsp_expectation, 3), time_samples], index = index_names).transpose()
        pred['Model'] = 'FSP estimation'
        preds.append(pred)
    if plot_exact_result[0]:
        exact_fi = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            exact_fi[i] = plot_exact_result[1](t, params)
        pred = pd.DataFrame([np.round(exact_fi,3), time_samples], index = index_names).transpose()
        pred['Model'] = 'Exact value'
        preds.append(pred)
    data = pd.concat(preds, ignore_index=True)
    #plot
    fig = seaborn.catplot(data=data, kind='bar', x=index_names[1], y=index_names[0], aspect=1.5, hue='Model',
        palette=colors)
    fig._legend.remove()
    plt.legend(loc='best')
    if save[0]:
        plt.savefig(save[1])
    plt.show()
           


