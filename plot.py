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
from typing import Callable, Tuple

# Plot probability distributions or sensitivities of probabilities distributions

def plot_model(to_pred: torch.tensor, 
            models: list, 
            up_bound: int,
            time_windows: np.ndarray,
            n_comps: int,
            index_names: Tuple[str, str] =('Probabilities', r'Abundance of species $S$'), 
            plot_test_result: Tuple[bool, torch.tensor] =(False, None), 
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, int, Tuple[int], int, int, int] = (False, None),
            plot: Tuple[str, int] =('probabilities', None),
            save: Tuple[bool, str] =(False, None)):
    r"""Plots distributions estimated with various methods for a single set of time and parameters and for a specified CRN.

    Args:
        - **to_pred** (torch.tensor): Time and parameters in the form requested by the MDN model: 
          :math:`[t, \theta_1, ..., \theta_M]`.
        - **models** (list): Mixture Density Network models to compute.
        - **up_bound** (int): Upper boundary of the predicted distribution.
        - **n_comps** (int): Number of components of the predicted mixture.
        - **index_names** (Tuple[str, str], optional): Labels of x-axis and y-axis. Defaults to ('Probabilities', 'Abundance of species S').
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If the first argument is True, plots the expected results 
          from the datasets for the chosen set of parameters. The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If the first argument is True, plots the exact results 
          for the chosen set of parameters. The second argument is the function that computes the exact results. Defaults to (False, None).        
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray], optional): If the first argument is True, 
          plots the estimated results with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the distribution with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterized propensity functions.
                4. :math:`C_r`: Integer such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
                5. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`. 
                6. **ind_species** (int): Index of the species of interest.
                7. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
                8. **n_control_params** (int): Number of varying parameters required to define the propensity functions.
                   Their values vary from a time window to another.
        - **plot** (Tuple[str, int], optional): The first argument is either 'probabilities' to plot a probability distribution, or 'sensitivities' 
          to plot a sensitivities of probability mass function distribution. If it is 'sensitivities', the second argument is the index of the parameter 
          such that it plots the sensitivities with respect to this parameter. Defaults to ('probabilities', None).
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. The second argument is the name of the file 
          in which to save the plot. Defaults to (False, None).
    """            
    # prediction
    x = torch.arange(up_bound).repeat(1, n_comps,1).permute([2,0,1])
    preds = []
    for i, model in enumerate(models):
        if plot[0] == 'probabilities':
            y_pred = neuralnetwork.mix_pdf(model, to_pred, x)
        elif plot[0] == 'sensitivities':
            y_pred = get_sensitivities.sensitivities(to_pred, model, length_output=up_bound)[:, plot[1]+1]
        y_pred = y_pred.detach().numpy()
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
                            init_state=plot_fsp_result[4],
                            n_fixed_params=plot_fsp_result[6],
                            n_control_params=plot_fsp_result[7])
        stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
        # for now, time_window = [0, t], parameters has shape (n_params + 1)
        fixed_parameters = np.stack([to_pred[1:plot_fsp_result[6]+1].numpy()]*n_time_windows)
        control_parameters = to_pred[plot_fsp_result[6]+1:].numpy().reshape(n_time_windows, plot_fsp_result[7])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        if plot[0] == 'probabilities':
            results_fsp = stv_calculator.marginal(to_pred[:1].numpy(), time_windows, parameters, ind_species=plot_fsp_result[5], index=None, with_stv=False)[:,0,0]
        if plot[0] == 'sensitivities':
            results_fsp = stv_calculator.marginal(to_pred[:1].numpy(), time_windows, parameters, ind_species=plot_fsp_result[5], index=plot[1], with_stv=True)[:,0,1] 
        length = min(up_bound, plot_fsp_result[3])
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
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            plot: Tuple[str, int] =('probabilities', None),
            n_col: int =2,
            save: Tuple[bool, str] =(False, None)):
    r"""Plots distributions estimated with various methods for multiple sets of time and parameters and for a specified CRN.

    Args:
        - **to_pred** (list): List of time and parameters in the form requested by the Mixture Density Network models:
          :math:`[t, \theta_1, ..., \theta_M]`.
        - **models** (list): Mixture Density Network models to compute.
        - **up_bound** (int): Upper boundary of the predicted distributions.
        - **n_comps** (int): Number of components of the predicted mixture.
        - **index_names** (Tuple[str], optional): Labels of x-axis and y-axis. Defaults to ('Probabilities', 'Abundance of species S').
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If the first argument is True, plots the expected results 
          from the datasets for the chosen set of parameters. The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If the first argument is True, plots the exact results for the
          chosen set of parameters. The second argument is the function that computes the exact results. Defaults to (False, None).
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): If the first argument is True, 
          plots the estimated results with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the distribution with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterized propensity functions.
                4. :math:`C_r`: Value such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
                5. **init_state** (np.ndarray): Initial state.
                6. **ind_species** (int): Index of the species of interest.
                7. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
                8. **n_control_params** (int): Number of varying parameters required to define the propensity functions.
                   Their values vary from a time window to another.
        - **plot** (Tuple[str, int], optional): The first argument is either 'probabilities' to plot a probability distribution, or 'sensitivities' 
          to plot a sensitivities of probability mass function distribution. If it is 'sensitivities', second argument is the index of the parameter 
          such that it plots the sensitivities with respect to this parameter. Defaults to ('probabilities', None).
        - **n_col** (int, optional): Number of columns to plot. Defaults to 2.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. Second element is the name of the file 
          in which to save the plot. Defaults to (False, None).
    """          
    n = len(to_pred)
    if n == 1:
            plot_model(to_pred[0], models, up_bound, n_comps, index_names, plot_test_result, plot_exact_result, plot_fsp_result, plot, save)
    else:
        _, axes = plt.subplots(math.ceil(n/n_col), n_col, figsize=(3*n,3*n))
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
                                    init_state=plot_fsp_result[4],
                                    n_fixed_params=plot_fsp_result[6],
                                    n_control_params=plot_fsp_result[7])
                stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
                fixed_parameters = np.stack([to_pred_[1:plot_fsp_result[6]+1].numpy()]*n_time_windows)
                control_parameters = to_pred_[plot_fsp_result[6]+1:].numpy().reshape(n_time_windows, plot_fsp_result[7])
                parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
                if plot[0] == 'probabilities':
                    results_fsp = stv_calculator.marginal(to_pred_[:1].numpy(), time_windows, parameters, ind_species=plot_fsp_result[5], index=None, with_stv=False)[:,0,0]
                if plot[0] == 'sensitivities':
                    results_fsp = stv_calculator.marginal(to_pred_[:1].numpy(), time_windows, parameters, ind_species=plot_fsp_result[5], index=plot[1], with_stv=True)[:,0,1]
                length = min(up_bound[k], plot_fsp_result[3])
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
            axes[k//n_col, k%n_col].legend().set_title('')
        plt.subplots_adjust(hspace=0.01)
        # fig.suptitle(f'{plot[0]} plot for params {params[1:]}')
        if save[0]:
            plt.savefig(save[1])




# Plot Fisher information table

def fi_table(time_samples: np.ndarray, 
            params: np.ndarray, 
            ind_param: int, 
            time_windows: np.ndarray,
            models: Tuple[bool, list, int] =(False, None, 4),
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            up_bound: int =200,
            out_of_bounds_index: int =None,
            save: Tuple[bool, str] =(False, None)):
    r"""Plots a table of the diagonal element of the Fisher Information estimated by various methods at various times.

    Args:
        - **time_samples** (list): Times to sample.
        - **params** (list): Parameters of the propensity functions.
        - **ind_param** (int): Index of the estimated Fisher Information diagonal value.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the Fisher Information 
          with MDN models. Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the Fisher Information with MDN models.
                2. **models_list** (list): List of MDN models from which to estimate the Fisher Information.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to calculate the exact value of the Fisher Information. Defaults to (False, None).
                
                - **exact_value** (bool): If True, calculates the exact value of the Fisher Information.
                - **fisher_information_function** (Callable): Function that computes the Fisher Information value.
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the Fisher Information with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray): Non-parameterized propensity functions.
                4. :math:`C_r`: Value such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
                5. **init_state** (np.ndarray): Initial state.
                6. **ind_species** (int): Index of the species of interest.
                7. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
                8. **n_control_params** (int): Number of varying parameters required to define the propensity functions.
                   Their values vary from a time window to another.
        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to 200.
        - **out_of_bounds_index** (int, optional): Index of the first time out of the training range in **time_samples**.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file in which to save the plot. Defaults to (False, None).
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
        crn = simulation.CRN(stoichiometry_mat=plot_fsp_result[1], 
                            propensities=plot_fsp_result[2], 
                            init_state=plot_fsp_result[4],
                            n_fixed_params=plot_fsp_result[6],
                            n_control_params=plot_fsp_result[7])
        stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
        fixed_parameters = np.stack([params[:plot_fsp_result[6]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[6]:].reshape(n_time_windows, plot_fsp_result[7])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        length = min(up_bound, plot_fsp_result[3])
        results_fsp = stv_calculator.marginal(time_samples, time_windows, parameters, plot_fsp_result[5], ind_param, with_stv=True)[:length,:,:]
        fsp_fi = np.zeros(n_rows)
        for i in range(n_rows):
            fsp_fi[i] = get_fi.fisher_information_t(results_fsp[:,i,0], results_fsp[:,i,1:])[0,0]
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
    # fig.suptitle(f'Element of the Fisher Information with respect to parameter n°{ind_param} - {crn_name} with parameter values: {params}')
    plt.show()


# Plot Fisher information bars

def fi_barplots(time_samples: np.ndarray, 
            params: np.ndarray, 
            ind_param: int, 
            time_windows: np.ndarray,
            models: Tuple[bool, list, int] =(False, None, 4),
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int] = (False, None),
            up_bound: int =200,
            save: Tuple[bool, str] =(False, None),
            colors: list =['blue', 'darkorange', 'forestgreen'],
            mean: bool =True):
    """Plots rectangular bars to visualize the diagonal element of the Fisher Information estimated by various methods at various times.

    Args:
        - **time_samples** (np.ndarray): Times to sample.
        - **params** (np.ndarray): Parameters of the propensity functions.
        - **ind_param** (int): Index of the estimated Fisher Information diagonal value.
        - **models** (Tuple[bool, list, int], optional): Arguments to estimate the Fisher Information 
          with MDN models. Defaults to (False, None, 4).

                1. **model_estimation** (bool): If True, estimates the Fisher Information with MDN models.
                2. **models_list** (list): List of MDN models from which to estimate the Fisher Information.
                3. **n_comps** (int): Number of mixture components.

        - **plot_exact_result** (Tuple[bool, Callable], optional): Arguments to calculate the exact value of the Fisher Information. Defaults to (False, None).
                
                - **exact_value** (bool): If True, calculates the exact value of the Fisher Information.
                - **fisher_information_function** (Callable): Function that computes the Fisher Information value.
        - **plot_fsp_result** (Tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, int, int], optional): Arguments to estimate the Fisher Information 
          with the FSP method.
                
                1. **fsp_estimation** (bool): If True, estimates the Fisher Information with the FSP method. Defaults to False.
                2. **stoich_mat** (np.ndarray): Stoichiometry matrix.
                3. **propensities** (np.ndarray[Callable]): Non-parameterized propensity functions.
                4. :math:`C_r`: Value such that the projection of :math:`(0, .., 0, C_r)` is the last element of the projected truncated space.
                5. **init_state** (Tuple[int], optional): Initial state. If None, the initial state is set to :math:`(0,..,0)`.
                6. **ind_species** (int): Index of the species of interest.
                7. **n_fixed_params** (int): Number of fixed parameters required to define the propensity functions.
                8. **n_control_params** (int): Number of varying parameters required to define the propensity functions.
                   Their values vary from a time window to another.
        - **up_bound** (int, optional): Upper boundary of the predicted distribution. Defaults to 200.
        - **save** (Tuple[bool, str], optional): If the first argument is True, saves the file. 
          The second argument is the name of the file in which to save the plot. Defaults to (False, None).
        - **colors** (list, optional): Chosen colors for the bars. Defaults to ['blue', 'darkorange', 'forestgreen'].
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
                            init_state=plot_fsp_result[4],
                            n_fixed_params=plot_fsp_result[6],
                            n_control_params=plot_fsp_result[7])
        stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
        fixed_parameters = np.stack([params[:plot_fsp_result[6]]]*n_time_windows)
        control_parameters = params[plot_fsp_result[6]:].reshape(n_time_windows, plot_fsp_result[7])
        parameters = np.concatenate((fixed_parameters, control_parameters), axis=1)
        length = min(up_bound, plot_fsp_result[3])
        results_fsp = stv_calculator.marginal(time_samples, time_windows, parameters, plot_fsp_result[5], ind_param, with_stv=True)[:length,:,:]
        fsp_fi = np.zeros(n_rows)
        for i in range(n_rows):
            fsp_fi[i] = get_fi.fisher_information_t(results_fsp[:,i,0], results_fsp[:,i,1:])[0,0]
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
