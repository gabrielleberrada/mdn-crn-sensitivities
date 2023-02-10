Workflow
========

The complete, step-by-step approach to compute the Fisher Information for Chemical Reaction Networks and Optimal Control experiments 
using Mixture Density Networks is introduced below.

Some useful notations are gathered in the following table:

+-------------------------------------------------------------------------------------------------+-------------------------------+
| Denomination                                                                                    | Notation                      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Fisher Information at time :math:`t`                                                            | :math:`I_t^{\theta,\xi}`      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of control parameters of the propensity functions :math:`\xi`                            | :math:`M_{\xi}`               |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of elements in the case of a finite state-space                                          | :math:`N_{\max}`              |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of fixed parameters of the propensity functions :math:`\theta`                           | :math:`M_{\theta}`            |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of reactions                                                                             | :math:`M`                     |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of species                                                                               | :math:`N``                    |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Number of time windows                                                                          | :math:`L`                     |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Probability mass function evaluated at the :math:`\ell`-th element of the enumerated state-space| :math:`p_\ell(t, \theta, \xi)`|
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Sensitivity of the likelihood matrix                                                            | :math:`S_t^{\theta,\xi}`      |
+-------------------------------------------------------------------------------------------------+-------------------------------+
| Total number of parameters of the propensity functions :math:`M_{\theta} + M_{\xi}\times L`     | :math:`M_{\text{tot}}`        |
+-------------------------------------------------------------------------------------------------+-------------------------------+


Step 1: Specify the Chemical Reaction Network to work on
--------------------------------------------------------

To fully define the Chemical Reaction Network to work on, define:

1. The stoichiometry matrix. It should have shape :math:`(N, M)`.
2. The propensity functions. Their first argument is a list of parameters, their second argument is the state.
   For instance, for a Production and Degradation Chemical Reaction Network:

.. code-block:: python

    def propensity_production(params, x):
        return params[0]

    def propensity_degradation(params, x):
        return params[1]*x[0]

3. The initial state.
4. The number of fixed parameters :math:`M_{\theta}` and the number of control parameters :math:`M_{\xi}`.
5. The index of the species to study (only one).

The Chemical Reaction Network is then completely defined.

If it does not follow mass-action kinetics, it is also necessary to specify the propensity derivatives with respect to each parameter for the Finite State Projection method. 
These derivatives should be stored in an array of shape :math:`(M, M_{\theta}+M_{\xi})`.

In the demos of the repository, the information for the Chemical Reaction Networks used as examples are stored in files called :meth:`propensities_[CRN_name].py`.

Step 2: Generate the Datasets
-----------------------------

Datasets :math:`\mathcal{D}_t^{\theta,\xi}` are generated from Stochastic Simulations, using the Stochastic Simulation Algorithm as outlined in :cite:`gillespie1976general`. 
However other simulation algorithms can be implemented.
The simulations are generated using Sobol sequences to choose the parameters over specified intervals.

To perform a single simulation, run the `simulation.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/simulation.py>`_ file.
To generate multiple simulations and estimate mass functions, use the `generate_data.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/generate_data.py>`_ file.  
The method :meth:`generate_csv.generate_csv_datasets` in the `generate_csv.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/generate_csv.py>`_ can be used to generate multiple datasets at once and store them in CSV files.

These datasets can be used for training, validation or testing purposes. 
When training one or multiple Mixture Density Networks for a single Chemical Reaction Network,
it is recommended to generate all data at once and then split the initial dataset into smaller ones
to avoid overlap in the various training, validation and testing datasets.

Step 3: Import data
-------------------

To load the datasets saved in CSV files as tensors (PyTorch) or arrays (NumPy), use the file `convert_csv.py <https://github.com/gabrielleberrada/DL_based_control_of_CRNs/blob/main/convert_csv.py>`_.

Here is an example:

.. code-block:: python

    FILE_NAME = 'file_name'
    CRN_NAME = 'crn_name'
    NUM_PARAMS = num_params

    # loading data
    X_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_train.csv')
    y_train1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_train.csv')

    X_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_valid.csv')
    y_valid1 = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_valid.csv')

    X_test = convert_csv.csv_to_tensor(f'{FILE_NAME}/X_{CRN_NAME}_test.csv')
    y_test = convert_csv.csv_to_tensor(f'{FILE_NAME}/y_{CRN_NAME}_test.csv')

    train_data = [X_train, y_train]
    valid_data = [X_valid, y_valid]
    test_data = [X_test, y_test]


Step 4: Train the Mixture Density Network
-----------------------------------------

Once the hyperparameters are defined, a Mixture Density Network can be trained on the simulated data:

.. code-block:: python

    model = neuralnetwork.NeuralNetwork(n_comps=N_COMPS, n_params=NUM_PARAMS, n_hidden=N_HIDDEN, mixture=mixture)
    train_losses, valid_losses = neuralnetwork.train_NN(model, train_data, valid_data, loss=neuralnetwork.loss_kldivergence, max_rounds=N_ITER, lr=LR, batchsize=BATCHSIZE)

To print the computed losses:

.. code-block:: python

    print("Training dataset")
    print(f"KLD : {neuralnetwork.mean_loss(X_train, y_train, model, loss=neuralnetwork.loss_kldivergence)}")
    print(f'Hellinger : {neuralnetwork.mean_loss(X_train, y_train, model, loss=neuralnetwork.loss_hellinger)}')

    print("\nTest dataset")
    print(f"KLD : {neuralnetwork.mean_loss(X_test, y_test, model, loss=neuralnetwork.loss_kldivergence)}")
    print(f'Hellinger : {neuralnetwork.mean_loss(X_test, y_test, model, loss=neuralnetwork.loss_hellinger)}')

Note that the validation dataset is not involved in the training process. However, if you use an early stopping method, the training process will stop based on the validation loss.

The file `training.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/training.py>`_ provides an example of the code implementation used for training and evaluating the model's performance with loss calculation.

Step 5: Estimate the probability mass functions and sensitivities of the likelihood
-----------------------------------------------------------------------------------

A trained Mixture Density Network can predict probability distributions.

- Inputs of the Mixture Density Network: :math:`[t, \theta_1, ..., \theta_{M_{\theta}}, \xi_1^1, \xi_1^2, ..., \xi_1^{M_{\xi}}, \xi_2^1, ..., \xi_L^{M_{\xi}}]` of type ``torch.tensor``.

- Outputs of the Mixture Density Network: The mixture parameters :math:`[w, r, q]` where :math:`w` are the mixture weights, :math:`r` and :math:`q` are the Negative Binomials parameters 
  (numbers of successes before the experiment is stopped and probabilities of success for each experiment). In case of a Poisson Mixture, it will only return parameters :math:`[w, r]`.
  The outputs are of type ``torch.tensor`` and are linked to the computational graph.

Predict probability mass function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To predict a probability mass function, use the function :meth:`get_sensitivities.probabilities`:

.. code-block:: python

    up_bound = 500 # to choose the upper boundary of the predicted distribution
    to_pred = torch.tensor([t, theta_1, .., theta_M_theta, xi_1_1, ..., xi_L_{M_xi}])

    y_pred = get_sensitivities.probabilities(to_pred, model, up_bound)
    y_pred = y_pred.detach().numpy() # to get the prediction as a NumPy array

Predict the sensitivity of the likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To predict the gradient of the likelihood with respect to time :math:`t` and to all parameters,
use :meth:`get_sensitivities.sensitivities`:

.. code-block:: python

    up_bound = 500 # to choose the upper boundary of the predicted distribution
    to_pred = torch.tensor([t, theta_1, .., theta_{M_theta}, xi_1_1, ..., xi_L_{q_1+q_2}])

    y_pred = get_sensitivities.sensitivities(to_pred, model, up_bound)

To get the sensitivity of the likelihood distribution for the :math:`i^{th}` parameter:

.. code-block:: python

    y_pred_i = y_pred[:, i+1]

Step 6A: Estimate the Fisher Information
----------------------------------------

The Fisher Information can be computed from the probability mass functions and the sensitivity of the likelihood (see section :ref:`Background on the Fisher Information<Background on the Fisher information>`).

To estimate the Fisher Information at a single time point :math:`t`, use the function :meth:`get_fi.fisher_information_t`. 
For multiple time points, ie to compute :math:`\sum\limits_{k=1}^L I_{t_k}^{\theta, \xi}`, use the function :meth:`get_fi.fisher_information`.


Step 6B: Estimate the expectation and its gradient
--------------------------------------------------

As for the Fisher Information, the expectation and its gradient with respect to the parameters can be computed using the functions 
:meth:`get_sensitivities.expected_val` and :meth:`get_sensitivities.gradient_expected_val`.

If a loss function :math:`\mathcal{L}` is specified in input, these functions can compute :math:`\mathcal{L}\big(E_{\theta, \xi}[X_t]\big)` and 
:math:`\nabla_{t, \theta, \xi} \mathcal{L} \big(E_{\theta, \xi}[X_t]\big) = \frac{dL(x)}{dx} \nabla_{t, \theta, \xi} E_{\theta, \xi}[X_t]`.

Step 7: Find the optimal control parameters
-------------------------------------------

The gradient of the expectation can be used in a Projected Gradient Descent algorithm to find the optimal control parameters that produce a specific species abundance at a given time.

The implementation of this method is in the `projected_gradient_descent.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/projected_gradient_descent.py>`_ file.
specifically in the class :class:`projected_gradient_descent.ProjectedGradientDescent_MDN` to use with Mixture Density Networks.

For a convenient way to run the algorithm, monitor progress and save results, check out the `training_pgd.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/training_pgd.py>`_ file.
The function :meth:`training_pgd.pgdMDN` does the following:

- Computes the gradient descent.
- Plots various aspects of the algorithm progress, including the optimal control values, loss values as estimated by the model,
  control parameter values, sensitivity values, SSA-estimated loss values over iterations and SSA-estimated abundances over time.
- Saves the hyperparameters, parameters and results of the algorithm in a ``.txt`` file for future reference.

Optional steps
--------------

Compare with Finite State Projection results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can compare the predictions of the Mixture Density Network to those made using the Finite State Projection (FSP) method.

If the propensity derivatives are not specified, it is assumed that the Chemical Reaction Network follows mass-action kinetics. 
However if this is not the case, make sure to specify the derivative functions to compute the sensitivities.

To truncate the state-space, set a value :math:`C_r \in \mathbb{N}` such that the :math:`N_{\max}`-th element in the enumerated state space corresponds to :math:`(0,...,0,C_r) \in \mathbb{N}^N`.
Note that :math:`\Phi_N(0,...,0,C_r) = N_{\max}-1`. In the case of :math:`N=2` species, this means that :math:`\frac{C_r(C_r+3)}{2}+1 = N_{\max}`.

The Finite State Projection method is implemented in the `fsp.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/fsp.py>`_ file. 
The method :meth:`fsp.SensitivitiesDerivation.solve_multiple_odes` computes the probability mass functions and the sensitivity of the likelihood. 

Keep in mind that the Finite State Projection method estimates global probability mass functions, while Mixture Density Networks predict marginal probability mass functions.
To make a fair comparison, compute the marginal probability mass functions using either the method :meth:`fsp.SensitivitiesDerivation.marginal` for a single 
mass function or the method :meth:`fsp.SensitivitiesDerivation.marginals` for multiple mass functions.

To compute the expectation and its gradient, use the methods :meth:`fsp.SensitivitiesDerivation.expected_val` and :meth:`fsp.SensitivitiesDerivation.gradient_expected_val`, respectively.

To run the Projected Gradient Descent using the Finite State Projection Method, use the class :class:`projected_gradient_descent.ProjectedGradientDescent_FSP`. Call the function :meth:`training_pgd.pgdFSP` 
to perform the gradient descent and save the results. 

Use regularisation methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

In all computed examples, we have not come across overfitting. If it arises, the regularisation methods provided in the code can be applied to address it:

- Add a :math:`\ell_2`-regularisation term.

- Use an early stopping method. You need to specify a tolerance threshold :math:`\delta` as well as a patience level :math:`n_p`. 
  See section :ref:`How to deal with overfitting? <Some advice on the implementation of the approach>` for more details on this method.

Tune hyperparameters
^^^^^^^^^^^^^^^^^^^^

To optimise the results, it is important to find the best hyperparameters.

Here are some examples of hyperparameters to tune:

- Batchsize
- Learning rate
- Mixture type
- Number of components
- Number of hidden layer neurons
- Number of samples in the training dataset
- Number of training rounds
- Patience and delta in case of early stopping

To tune them, train models for each parameters combination. Use the validation dataset to estimate the optimal set of parameters.
To speed up the process, consider using multiprocessing.

In our demos, we tuned the learning rate, number of training rounds, number of hidden layer neurons and batchsize. 
We kept the number of mixture components separate.

The implemented code is available in the `tuning.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/tuning.py>`_ file. It is easily adaptable to any Chemical Reaction Network. This file
uses the function :meth:`hyperparameters_tuning.test_multiple_combs`  which trains one or several models and
saves the results of all parameter combinations in a CSV file. It calls the function :meth:`hyperparameters_test.test_comb` to test each combination.

A similar method can be used to tune the Projected Gradient Descent algorithm hyperparameters.

Plot probability distributions and sensitivities distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluating the accuracy of a model based only on loss values can be challenging. A quicker and more intuitive way to assess model performance is by visualizing the predicted distributions and comparing them to other distributions.
This can include a known exact distribution if the Chemical Reaction Network has a known sampled mass function, simulated distributions from Stochastic Algorithms, or even distirbutions estimated by the Finite State Projection method.

To plot the results, you can use the functions in the `plot.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/plot.py>`_ file. For a single plot, call the function
:meth:`plot.plot_model`. To compare multiple distributions, call the function :meth:`plot.multiple_plots`. 

These functions allow you to plot both probability distributions and sensitivity distributions, giving a comprehensive view of the model performance.

Examples can be found in the notebooks on the `GitHub repository <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs>`_.

Plot the Fisher Information values in a table or in barplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compare and see the Fisher Information results from different sources, a comprehensive table can be created by calling the function :meth:`plot.fi_table`.
It brings together the predictions from the Mixture Density Network, the calculations from the Finite State Projection, and, when available, the exact results.

For a more visual representation, you can use bar plots, generated by calling the function :meth:`fi_barplots`.

Examples can be found in all the notebooks on the `GitHub repository <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs>`_.

Plot the expectation and its gradient in a table or in barplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just as for the Fisher Information values, the results of the expectation and its gradient values, evaluated using different methods, can be compared in tables or barplots. 
To do so, call the functions :meth:`plot.expect_val_table` and :meth:`plot.expect_val_barplots`.

Examples can be found in all the notebooks on the `GitHub repository <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs>`_.

Store and load Mixture Density Networks weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trained Mixture Density Network model can be saved and loaded at any time.

To do so, use the `save_load_MDN.py <https://github.com/gabrielleberrada/DL_based_Control_of_CRNs/blob/main/save_load_MDN.py>`_ file.
To save all needed information to define the Mixture Density Network in a `.pt` file, call the function :meth:`save_MDN_model`.
To load a Mixture Density Network from a `.pt` file, call the function :meth:`save_load_MDN.load_MDN_model`.

Estimate the loss of the Projected Gradient Descent based on Stochastic Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a parameter configuration :math:`\theta` and :math:`\xi`, the true loss value (as estimated by Stochastic Simulations)
can be computed as following:

.. code-block:: python

    sim = generate_data.CRN_Simulations(crn=crn,
                                        time_windows=time_windows,
                                        n_trajectories=10**4,
                                        ind_species=ind_species,
                                        complete_trajectory=False,
                                        sampling_times=time_windows)
    parameters = np.concatenate((fixed_parameters, control_parameters))
    samples, _ = sim.run_simulations(parameters)
    expect = np.mean(samples, axis=0)
    res = 0
    for j in range(n_time_windows):
        res += weights[j] * loss_function[j](expect[j])
    print(res)

The method to compute the real loss evolution based on Stochastic Simulations also is implemented in
:meth:`projected_gradient_descent.ProjectedGradientDescent_CRN.plot_performance_index`.









