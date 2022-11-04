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
            models: list[neuralnetwork.NeuralNetwork], 
            up_bound: int,
            n_comps: int, 
            index_names: Tuple[str, str] =('Probabilities', 'Abundance of species S'), 
            plot_test_result: Tuple[bool, torch.tensor] =(False, None), 
            plot_exact_result: Tuple[bool, Callable] =(False, None), 
            plot_fsp_result: Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]] = (False, np.zeros(1), [], 10, None),
            confidence_interval: bool =False,
            plot: Tuple[str, int] =('probabilities', None),
            save: Tuple[bool, str] =(False, None),
            crn_name: str =''):
    """Plots a single distribution of a CRN from models estimations.
    See comments in the code to adapt it to your CRN.

    Args:
        - **to_pred** (torch.tensor): Inputs for the Mixture Density Networks.
        - **models** (list[neuralnetwork.NeuralNetwork]): Mixture Density Networks to compute.
        - **up_bound** (int): Upper boundary of the estimated distribution.
        - **n_comps** (int): Number of components of the mixture.
        - **index_names** (Tuple[str, str], optional): Labels of x-axis and y-axis.. Defaults to ('Probabilities', 'Abundance of species S').
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If first element is True, plots the expected results from the datasets for the chosen set of parameters. \
        The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If first element is True, plots the exact results for the chosen set of parameters. \
        The second argument is the function that computes the exact results. Defaults to (False, None).        
        - **plot_fsp** (Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]], optional): If first element is True, plots the estimated results with the FSP method. \
            Defaults to (False, np.zeros(1), [], 10, None).
                
                1. **fsp_estimation** (bool): Indicates whether to estimate the distribution with the FSP method.
                2. **stoich_mat** (np.ndarray[int]): Stoichiometric matrix of the CRN.
                3. **propensities** (np.ndarray[Callable]): Non-parameterized propensities of the CRN.
                4. :math:`c_r`: Value such that :math:`(0, .., 0, c_r)` is the last value in the truncated space.
                5. **init_state** (np.ndarray[int], optional): If needed, initial state.        
               
        - **confidence_interval** (bool, optional): If True, plots an estime of the central tendency and a confidence interval for that estimate. If False, plots each line. Defaults to False.
        - **plot** (Tuple[str, int], optional): First element is either 'probabilities' to plot a probability distribution, or 'sensitivities' to plot probability sensitivities distribution. \
        If it is 'sensitivities', second argument is the index of the parameter such that it plots the  sensitivities of probabilities with respect to this parameter. Defaults to ('probabilities', None).
        - **save** (Tuple[bool, str], optional): If first element is True, saves the file. Second element is the name of the file in which to save the plot. Defaults to (False, None).
        - **crn_name** (str): Name of the CRN as a string, to use for the figure title.
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
        pred['model'] = f'training{i+1}'
        preds.append(pred)
    if plot_test_result[0]:
        result = plot_test_result[1]
        if torch.is_tensor(result):
            result = result.detach().numpy()
        test_result = pd.DataFrame([np.squeeze(result), np.arange(up_bound)], index = index_names).transpose()
        test_result['model'] = 'SSA simulation'
        preds.append(test_result)
    if plot_fsp_result[0]:
        crn = simulation.CRN(plot_fsp_result[1], plot_fsp_result[2], len(params))
        stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
        if plot_fsp_result[4]:
            init_state = plot_fsp_result[4]
        else:
            init_state = np.zeros(2*(plot_fsp_result[3]+1))
            init_state[1] = 1
            init_state = np.stack([init_state]*crn.n_reactions)
        probs_fsp, stv_fsp = stv_calculator.get_sensitivities(init_state, 0, to_pred[0].numpy(), params.numpy(), t_eval=to_pred[0].numpy())
        if plot[0] == 'probabilities':
            length = min(up_bound, np.shape(probs_fsp)[0])
            fsp_result = pd.DataFrame([probs_fsp[0, :up_bound], np.arange(length)], index = index_names).transpose()
        elif plot[0] == 'sensitivities':
            length = min(up_bound, np.shape(probs_fsp)[0])
            fsp_result = pd.DataFrame([stv_fsp[0, :up_bound, plot[1]], np.arange(length)], index = index_names).transpose()
        fsp_result['model'] = 'FSP estimation'
        preds.append(fsp_result)
    if plot_exact_result[0]:
        parameters = []
        for tens in to_pred:
            parameters.append(tens.numpy())
        exact_result = pd.DataFrame([[plot_exact_result[1](k, parameters) for k in range(up_bound)],
                                    np.arange(up_bound)], index = index_names).transpose()
        exact_result['model'] = 'exact result'
        preds.append(exact_result)
    data = pd.concat(preds, ignore_index=True)
    params = [np.round(param.numpy(), 2) for param in to_pred]
    if confidence_interval:
        seaborn.relplot(data=data, x=index_names[1], y=index_names[0], aspect=1.5, kind='line').set(title=fr'{plot[0]} plot for {crn_name} with $t=${params[0]}, $\theta=${params[1:]}')
    else:
        seaborn.relplot(data=data, x=index_names[1], y=index_names[0], hue='model', style='model', aspect=1.5, kind='line').set(title=fr'{plot[0]} plot for {crn_name} with $t=${params[0]}, $\theta=${params[1:]}')
    if save[0]:
        plt.savefig(save[1])

def multiple_plots(to_pred: list[torch.tensor],
            models: list[neuralnetwork.NeuralNetwork],
            up_bound: int,
            n_comps: int,
            index_names: Tuple[str] =('Probabilities', 'Abundance of species S'),
            plot_test_result: Tuple[bool, torch.tensor] =(False, None),
            plot_exact_result: Tuple[bool, Callable] =(False, None),
            plot_fsp_result: Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]] = (False, np.zeros(1), [], 10, None),
            confidence_interval: bool =False,
            plot: Tuple[str, int] =('probabilities', None),
            n_col: int =2,
            save: Tuple[bool, str] =(False, None),
            crn_name: str =''):
    """Plots multiple distributions of a CRN from models estimations.
    See comments in the code to adapt it to your CRN.

    Args:
        - **to_pred** (list[torch.tensor]): List of inputs.
        - **models** (list[neuralnetwork.NeuralNetwork]): Models to compute.
        - **up_bound** (int): Upper boundary of the estimated distribution.
        - **n_comps** (int): Number of components of the mixture.
        - **index_names** (Tuple[str], optional): Labels of x-axis and y-axis. Defaults to ('Probabilities', 'Abundance of species S').
        - **plot_test_result** (Tuple[bool, torch.tensor], optional): If first element is True, plots the expected results from the datasets for the chosen set of parameters. \
        The second argument is the expected results. Defaults to (False, None).
        - **plot_exact_result** (Tuple[bool, Callable], optional): If first element is True, plots the exact results for the chosen set of parameters. \
        The second argument is the function that computes the exact results. Defaults to (False, None).
        - **plot_fsp** (Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]], optional): If first element is True, plots the estimated results with the FSP method. \
            Defaults to (False, np.zeros(1), [], 10, None).
                
                1. **fsp_estimation** (bool): Indicates whether to estimate the distribution with the FSP method.
                2. **stoich_mat** (np.ndarray[int]): Stoichiometric matrix of the CRN.
                3. **propensities** (np.ndarray[Callable]): Non-parameterized propensities of the CRN.
                4. :math:`c_r`: Value such that :math:`(0, .., 0, c_r)` is the last value in the truncated space.
                5. **init_state** (np.ndarray[int], optional): If needed, initial state.        

        - **confidence_interval** (bool, optional): If True, plots an estime of the central tendency and a confidence interval for that estimate. If False, plots each line. Defaults to False.
        - **plot** (Tuple[str, int], optional): First element is either 'probabilities' to plot a probability distribution, or 'sensitivities' to plot probability sensitivities distribution. \
        If it is 'sensitivities', second argument is the index of the parameter such that it plots the sensitivities of probabilities with respect to this parameter. \
        Defaults to ('probabilities', None).
        - **n_col** (int, optional): Number of columns to plot. Defaults to 2.
        - **save** (Tuple[bool, str], optional): If first element is True, saves the file. Second element is the name of the file in which to save the plot. Defaults to (False, None).
        - **crn_name** (str): Name of the CRN as a string, to use for the figure title.
    """          
    n = len(to_pred)
    if n == 1:
            plot_model(to_pred[0], models, n_comps, up_bound, index_names, plot_test_result, plot_exact_result, plot_fsp_result, confidence_interval, plot, save, crn_name)
    else:
        fig, axes = plt.subplots(math.ceil(n/n_col), n_col, figsize=(12,12))
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
                pred['model'] = f'training{i+1}'
                preds.append(pred)
            if plot_test_result[0]:
                result = plot_test_result[1][k]
                if torch.is_tensor(result):
                    result = result.detach().numpy()
                test_result = pd.DataFrame([np.squeeze(result), np.arange(up_bound[k])], index = index_names).transpose()
                test_result['model'] = 'SSA simulation'
                preds.append(test_result)
            if plot_fsp_result[0]:
                crn = simulation.CRN(plot_fsp_result[1], plot_fsp_result[2], len(params))
                stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp_result[3])
                if plot_fsp_result[4]:
                    init_state = plot_fsp_result[4]
                else:
                    init_state = np.zeros(2*(plot_fsp_result[3]+1))
                    init_state[1] = 1
                    init_state = np.stack([init_state]*crn.n_reactions)
                probs_fsp, stv_fsp = stv_calculator.get_sensitivities(init_state, 0, to_pred_[0].numpy(), params.numpy(), t_eval=to_pred_[0].numpy())
                if plot[0] == 'probabilities':
                    length = min(up_bound, np.shape(probs_fsp)[0])
                    fsp_result = pd.DataFrame([probs_fsp[0, :up_bound], np.arange(length)], index = index_names).transpose()
                elif plot[0] == 'sensitivities':
                    length = min(up_bound, np.shape(probs_fsp)[0])
                    fsp_result = pd.DataFrame([stv_fsp[0, :up_bound, plot[1]], np.arange(length)], index = index_names).transpose()
                fsp_result['model'] = 'FSP estimation'
                preds.append(fsp_result)
            if plot_exact_result[0]:
                parameters = []
                for tens in to_pred_:
                    parameters.append(tens.numpy())
                exact_result = pd.DataFrame([[plot_exact_result[1](k, parameters) for k in range(up_bound[k])], 
                                            np.arange(up_bound[k])], index = index_names).transpose()
                exact_result['model'] = 'exact result'
                preds.append(exact_result)
            data = pd.concat(preds, ignore_index=True)
            params = [np.round(param.numpy(), 2) for param in to_pred]
            if confidence_interval:
                seaborn.lineplot(ax=axes[k//n_col, k%n_col], data=data, x=index_names[1], y=index_names[0])
            else:
                seaborn.lineplot(ax=axes[k//n_col, k%n_col], data=data, x=index_names[1], y=index_names[0], hue='model', style='model')
            # adapt the subtitles to your case
            axes[k//n_col, k%n_col].set_title(fr'$t=${params[k][0]}, $\theta=${params[k][1:]}')
        plt.subplots_adjust(hspace=0.01)
        # adapt the title to your case
        fig.suptitle(f'{plot[0]} plot for {crn_name}')
        if save[0]:
            plt.savefig(save[1])




# Plot Fisher information table

def fi_table(time_samples: list[float], 
            params: list[float], 
            ind_param: int, 
            models: Tuple[bool, list[neuralnetwork.NeuralNetwork], int] =(False, None, 3),
            plot_exact: Tuple[bool, Callable] =(False, None), 
            plot_fsp: Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]] = (False, np.zeros(1), [], 10, None),
            up_bound: int =100,
            crn_name: str ='',
            out_of_bounds_index: int =None):
    """Plots the fisher information table for one parameter.

    Args:
        - **time_samples** (list[float]): Sampling times.
        - **params** (list[float]): Parameters chosen for the CRN.
        - **ind_param** (int): Index of the parameter whose Fisher information is estimated.
        - **models** (Tuple[bool, list[neuralnetwork.NeuralNetwork], int], optional): Arguments to estimate the Fisher Information with neural network models. Defaults to (False, None, 3).

                1. **model_estimation** (bool): Indicates whether to estimate the Fisher Information with neural network models.
                2. **models_list** (list[neuralnetwork.NeuralNetwork]): List of models from which to estimate the Fisher Information.
                3. **n_comps** (int): Number of components.

        - **plot_exact** (Tuple[bool, Callable], optional): Arguments to estimate the Fisher Information with its exact value. Defaults to (False, None).
                
                - **exact_value** (bool): Indicates whether to calculate the exact value of the Fisher Information.
                - **fisher_information_function** (Callable): Function that computes the Fisher Information value
        - **plot_fsp** (Tuple[bool, np.ndarray[int], np.ndarray[Callable], int, np.ndarray[int]], optional): Arguments to estimate the Fisher Information with the FSP method. Defaults to (False, np.zeros(1), [], 10, None).
                
                1. **fsp_estimation** (bool): Indicates whether to estimate the Fisher Information with the FSP method.
                2. **stoich_mat** (np.ndarray[int]): Stoichiometric matrix of the CRN.
                3. **propensities** (np.ndarray[Callable]): Non-parameterized propensities of the CRN.
                4. :math:`c_r`: Value such that :math:`(0, .., 0, c_r)` is the last value in the truncated space.
                5. **init_state** (np.ndarray[int], optional): If needed, initial state.
        - **up_bound** (int, optional): Upper boundaries of the distribution to compute. Defaults to 100.
        - **crn_name** (str, optional): Name of the CRN as a string, to use for the figure title.
        - **out_of_bounds_index** (int, optional): Index of the first time out of training range in **time_samples**.
    """            
    rows = [fr'$t={t}$' for t in time_samples]
    n_rows = len(time_samples)
    # compute probabilities and sensitivities with the neural networks
    if models[0]:
        probabilities_m = np.zeros((len(time_samples),up_bound))
        stv_m = np.zeros((len(time_samples),up_bound, len(params)))
        x = torch.arange(up_bound).repeat(1, models[2],1).permute([2,0,1])
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
    if plot_fsp[0]:
        crn = simulation.CRN(plot_fsp[1], plot_fsp[2], len(params))
        stv_calculator = fsp.SensitivitiesDerivation(crn, plot_fsp[3])
        if plot_fsp[4]:
            init_state = plot_fsp[4]
        else:
            init_state = np.zeros(2*(plot_fsp[3]+1))
            init_state[1] = 1
            init_state = np.stack([init_state]*crn.n_reactions)
        probs_fsp, stv_fsp = stv_calculator.get_sensitivities(init_state, 0, time_samples[-1], params.numpy(), t_eval=time_samples)
        fsp_fi = np.zeros(n_rows)
        for i in range(n_rows):
            fim = get_fi.fisher_information_t(probs_fsp[i,:], stv_fsp[i,:,:])
            fsp_fi[i] = fim[ind_param, ind_param]
    # add condition for models and fsp
    columns = []
    data = []
    # gathering data
    if models[0]:
        columns.append('Predicted with NN (mean)')
        data.append(np.round(predicted_fi,3))
    if plot_fsp[0]:
        columns.append('Estimated with FSP')
        data.append(np.round(fsp_fi, 3))
    if plot_exact[0]:
        columns.append('Exact')
        exact_fi = np.zeros(n_rows)
        for i, t in enumerate(time_samples):
            exact_fi[i] = plot_exact[1](t, params)
        data.append(np.round(exact_fi,3))
    if len(data)==1:
        data = np.array(data).T
    else:
        data = np.stack(data, axis=-1)
    if out_of_bounds_index != None:
        data = np.insert(data.astype('str'), out_of_bounds_index, '...', axis=0)
    #plot
    fig, ax = plt.subplots(figsize=(10,3))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = plt.table(cellText=data, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center', colWidths=[1.]*len(columns))
    table.set_fontsize(14)
    table.scale(0.4,1.6)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    fig.suptitle(f'Element of the Fisher Information with respect to parameter n°{ind_param} - {crn_name} with parameter values: {params.tolist()}')
    plt.show()